import numpy as np
import torch
from collections import OrderedDict
from opt_einsum import contract
from scipy.special import logit
from tqdm import tqdm
from scipy.special import xlogy
from scipy.special import softmax
from Spectra import Spectra_util
import torch.nn as nn
import scipy
import pandas as pd
from pyvis.network import Network
import random

from torch.distributions.normal import Normal
from torch.distributions.log_normal import LogNormal
from torch.distributions.dirichlet import Dirichlet

### Class for SPECTRA model
from Spectra.initialization import *


class SPECTRA(nn.Module):
    """

    Parameters
        ----------
        X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used for initialization of self.n and self.p but not stored as an attribute
        labels : np.ndarray or NoneType
            the ``(n, )`` -shaped array containing cell type labels. If use_cell_types == False, then should be set to None

        L : dict or OrderedDict [if use_cell_types == False, then int]
            ``number of cell types + 1``-shaped dictionary. Must have "global" as a key, indicating the number of global factors
            {
                "global": 15,
                "CD8": 5
                ...
            }
            > Format matches output of K_est.py to estimate the number of
            > Must match cell type labels provided during training
            > Recommended practice is to assign at minimum 2 factors per cell type
            > Note that L contains the number of factors that describe the graph.
        adj_matrix :  dict or OrderedDict
            ``a dictionary of adjacency matrices, one for every cell type + a "global"
            {
                "global": ``(p, p)``-shaped binary np.ndarray
                "CD8": ...

            }
        weights : dict or OrderedDict or NoneType [if use_cell_types == False, then ``(p, p)``-shaped array]
            the ``(p, p)``-shaped set of edge weights per . If weight[i,j] is non-zero when adj_matrix[i,j] = 0
            this weight is ignored.

            if weights == None, no weights are used
        lam : float
            lambda parameter of the model, which controls the relative influence of the graph vs expression loss functions. This term multiplies the expression loss, so smaller values of lambda upweight the prior information
        delta : float
            delta parameter of the model, which controls a lower bound for gene scaling factors. If delta is small then the maximum ratio between gene scaling factors is larger and lowly expressed genes can be put on the same scale as highly expressed genes.
        kappa : float or NoneType
            kappa controls the background rate of edges in the graph. if kappa is a float, kappa is fixed to the given float value. If kappa == None, then kappa is a parameter that is estimated from the data.
        rho : float or NoneType
            rho controls the bakcground rate of non-edges in the graph. if rho is a float, rho is fixed to the given float value. If rho == None, then rho is a parameter that is estimated from the data.
        use_cell_types: bool
            use_cell_types is a Boolean variable that determines whether cell type labels are used to fit the model. If False, then parameters are initialized as nn.Parameter rather than as nn.ParameterDict with cell type keys that index nn.Parameter values
        determinant_penalty : float
            determinant penalty affects the selection parameters that are fit when L[cell_type] > K[cell_type]. A determinantally regularized selection parameter is fit with determinant penalty that encourages sparsity and diversity.
    Attributes
        ----------
        model.delta : delta parameter of the model

        model.lam : lambda parameter of the model

        model.determinant_penalty : determinant penalty of the model

        model.L : L parameter, either int, dict or OrderedDict()

        model.p : number of genes

        model.n : number of cells

        model.use_cell_types : if True then cell types are considered, else cell types ignored. Affects the dimensions of the initialized parameters.

        model.kappa : if not kappa, nn.ParameterDict() if use_cell_types, else nn.Parameter(). If kappa is a float, it is fixed throughout training

        model.rho : if not rho, nn.ParamterDict() if use_cell_types, else nn.Parameter. If rho is a float it is fixed throughout training

        model.adj_matrix : adjacency matrix with diagonal removed. dict containing torch.Tensors

        model.adj_matrix_1m : 1 - adjacency matrix with diagonal removed. dict containing torch.Tensors

        model.weights : contains edge weights. format matches adj_matrix

        model.cell_types : np.ndarray containing array of unique cell types

        model.cell_type_counts : dict {key = cell type, values = number of cells}

        model.theta : nn.ParameterDict() or nn.Parameter() containing the factor weights

        model.alpha : nn.ParameterDict() or nn.Parameter() containing the cell loadings

        model.eta : nn.ParameterDict() or nn.Parameter() containing the interaction matrix between factors

        model.gene_scaling : nn.ParameterDict() or nn.Parameter() containing the gene scale factors

        model.selection :  nn.ParameterDict() or nn.Parameter() containing the attention weights. Only initialized when L[cell_type] > K[cell_type] for some cell type or when L > K and use_cell_types == False

        model.kgeql_flag : dict or bool. dictionary of boolean values indicating whether K >= L.  When use_cell_types == False, it is a boolean value

    Methods
        ----------

        model.loss(self, X, labels) : called by fit if use_cell_types = True. Evalutes the loss of the model

        model.loss_no_cell_types(self,X) : called by fit if use_cell_types = False. Evalutes the loss of the model

        model.initialize(self, gene_sets,val) : initialize the model based on given dictionary of gene sets. val is a float that determines the strength of the initialization.

        model.initialize_no_celltypes(self, gs_list, val) : initialize the model based on given list of gene sets. val is a float that determines the strength of the initialization.


    To do:
        __________

        > Alternative initialization functions

        > comment SPECTRA-EM code

        > test lower bound constraint [see pyspade_global.py implementation]

        > Overlap threshold test statistic


    """

    def __init__(
        self,
        X,
        labels,
        adj_matrix,
        L,
        weights=None,
        lam=0.01,
        delta=0.001,
        kappa=None,
        rho=0.001,
        use_cell_types=True,
    ):
        super(SPECTRA, self).__init__()

        # hyperparameters
        self.delta = delta
        self.lam = lam
        self.L = L
        # for memory efficiency we don't store X in the object attributes, but require X dimensions to be known at initialization
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.use_cell_types = use_cell_types

        if not use_cell_types:
            # check that L is an int
            assert isinstance(self.L, int)

            # trust the user to input a np.ndarray for adj_matrix
            self.adj_matrix = torch.Tensor(adj_matrix) - torch.Tensor(
                np.diag(np.diag(adj_matrix))
            )
            adj_matrix_1m = 1.0 - adj_matrix
            self.adj_matrix_1m = torch.Tensor(
                adj_matrix_1m - np.diag(np.diag(adj_matrix_1m))
            )
            if weights is not None:
                self.weights = torch.Tensor(weights) - torch.Tensor(
                    np.diag(np.diag(adj_matrix))
                )
            else:
                self.weights = self.adj_matrix

            self.theta = nn.Parameter(Normal(0.0, 1.0).sample([self.p, self.L]))
            self.alpha = nn.Parameter(Normal(0.0, 1.0).sample([self.n, self.L]))
            self.eta = nn.Parameter(Normal(0.0, 1.0).sample([self.L, self.L]))
            self.gene_scaling = nn.Parameter(Normal(0.0, 1.0).sample([self.p]))

            if kappa == None:
                self.kappa = nn.Parameter(Normal(0.0, 1.0).sample())
            else:
                self.kappa = torch.tensor(np.log(kappa / (1 - kappa)))  #
            if rho == None:
                self.rho = nn.Parameter(Normal(0.0, 1.0).sample())
            else:
                self.rho = torch.tensor(np.log(rho / (1 - rho)))

        if use_cell_types:
            # convert adjacency matrices to pytorch tensors to make optimization easier later
            self.adj_matrix = {
                cell_type: (
                    torch.Tensor(adj_matrix[cell_type])
                    - torch.Tensor(np.diag(np.diag(adj_matrix[cell_type])))
                    if len(adj_matrix[cell_type]) > 0
                    else []
                )
                for cell_type in adj_matrix.keys()
            }
            # for convenience store 1 - adjacency matrix elements [except on diagonal, where we store 0]
            adj_matrix_1m = {
                cell_type: (
                    1.0 - adj_matrix[cell_type]
                    if len(adj_matrix[cell_type]) > 0
                    else []
                )
                for cell_type in adj_matrix.keys()
            }  # one adj_matrix per cell type
            self.adj_matrix_1m = {
                cell_type: (
                    torch.Tensor(
                        adj_matrix_1m[cell_type]
                        - np.diag(np.diag(adj_matrix_1m[cell_type]))
                    )
                    if len(adj_matrix_1m[cell_type]) > 0
                    else []
                )
                for cell_type in adj_matrix_1m.keys()
            }  # one adj_matrix per cell type

            # if weights are provided, convert these to tensors, else set weights = to adjacency matrices
            if weights:
                self.weights = {
                    cell_type: (
                        torch.Tensor(weights[cell_type])
                        - torch.Tensor(np.diag(np.diag(weights[cell_type])))
                        if len(weights[cell_type]) > 0
                        else []
                    )
                    for cell_type in weights.keys()
                }
            else:
                self.weights = self.adj_matrix

            self.cell_types = np.unique(
                labels
            )  # cell types are the unique labels, again require knowledge of labels at initialization but do not store them

            # store a dictionary containing the counts of each cell type
            self.cell_type_counts = {}
            for cell_type in self.cell_types:
                n_c = sum(labels == cell_type)
                self.cell_type_counts[cell_type] = n_c

            # initialize parameters randomly, we use torch's ParameterDict() for storage for intuitive accessing cell type specific parameters
            self.theta = nn.ParameterDict()
            self.alpha = nn.ParameterDict()
            self.eta = nn.ParameterDict()
            self.gene_scaling = nn.ParameterDict()

            if kappa == None:
                self.kappa = nn.ParameterDict()
            if rho == None:
                self.rho = nn.ParameterDict()
            # initialize global params
            self.theta["global"] = nn.Parameter(
                Normal(0.0, 1.0).sample([self.p, self.L["global"]])
            )
            self.eta["global"] = nn.Parameter(
                Normal(0.0, 1.0).sample([self.L["global"], self.L["global"]])
            )
            self.gene_scaling["global"] = nn.Parameter(
                Normal(0.0, 1.0).sample([self.p])
            )
            if kappa == None:
                self.kappa["global"] = nn.Parameter(Normal(0.0, 1.0).sample())
            if rho == None:
                self.rho["global"] = nn.Parameter(Normal(0.0, 1.0).sample())

            # initialize all cell type specific params
            for cell_type in self.cell_types:
                self.theta[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([self.p, self.L[cell_type]])
                )
                self.eta[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([self.L[cell_type], self.L[cell_type]])
                )
                n_c = sum(labels == cell_type)
                self.alpha[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([n_c, self.L["global"] + self.L[cell_type]])
                )
                self.gene_scaling[cell_type] = nn.Parameter(
                    Normal(0.0, 1.0).sample([self.p])
                )

                if kappa == None:
                    self.kappa[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample())

                if rho == None:
                    self.rho[cell_type] = nn.Parameter(Normal(0.0, 1.0).sample())

            # if kappa and rho are provided, hold these fixed during training, else fit as free parameters
            # to unify the cases, we put this in the same format
            if kappa != None:
                self.kappa = {}
                self.kappa["global"] = torch.tensor(np.log(kappa / (1 - kappa)))
                for cell_type in self.cell_types:
                    self.kappa[cell_type] = torch.tensor(np.log(kappa / (1 - kappa)))
                self.kappa = nn.ParameterDict(self.kappa)

            if rho != None:
                self.rho = {}
                self.rho["global"] = torch.tensor(np.log(rho / (1 - rho)))
                for cell_type in self.cell_types:
                    self.rho[cell_type] = torch.tensor(np.log(rho / (1 - rho)))
                self.rho = nn.ParameterDict(self.rho)

    def loss(self, X, labels):
        assert (
            self.use_cell_types
        )  # if this is False, fail because model has not been initialized to use cell types

        # convert inputs to torch.Tensors
        X = torch.Tensor(X)
        # labels = torch.Tensor(labels)

        # initialize loss and fetch global parameters
        loss = 0.0
        theta_global = torch.softmax(self.theta["global"], dim=1)
        eta_global = (self.eta["global"]).exp() / (1.0 + (self.eta["global"]).exp())
        eta_global = 0.5 * (eta_global + eta_global.T)
        gene_scaling_global = self.gene_scaling["global"].exp() / (
            1.0 + self.gene_scaling["global"].exp()
        )
        kappa_global = self.kappa["global"].exp() / (1 + self.kappa["global"].exp())
        rho_global = self.rho["global"].exp() / (1 + self.rho["global"].exp())

        # loop through cell types and evaluate loss at every cell type
        for cell_type in self.cell_types:
            kappa = self.kappa[cell_type].exp() / (1 + self.kappa[cell_type].exp())
            rho = self.rho[cell_type].exp() / (1 + self.rho[cell_type].exp())
            gene_scaling_ct = self.gene_scaling[cell_type].exp() / (
                1.0 + self.gene_scaling[cell_type].exp()
            )
            X_c = X[labels == cell_type]
            adj_matrix = self.adj_matrix[cell_type]
            weights = self.weights[cell_type]
            adj_matrix_1m = self.adj_matrix_1m[cell_type]
            theta_ct = torch.softmax(self.theta[cell_type], dim=1)
            eta_ct = (self.eta[cell_type]).exp() / (1.0 + (self.eta[cell_type]).exp())
            eta_ct = 0.5 * (eta_ct + eta_ct.T)
            theta_global_ = contract(
                "jk,j->jk", theta_global, gene_scaling_global + self.delta
            )
            theta_ct_ = contract("jk,j->jk", theta_ct, gene_scaling_ct + self.delta)
            theta = torch.cat((theta_global_, theta_ct_), 1)
            alpha = torch.exp(self.alpha[cell_type])
            recon = contract("ik,jk->ij", alpha, theta)
            term1 = -1.0 * (torch.xlogy(X_c, recon) - recon).sum()
            if len(adj_matrix) > 0:
                mat = contract("il,lj,kj->ik", theta_ct, eta_ct, theta_ct)
                term2 = (
                    -1.0
                    * (
                        torch.xlogy(
                            adj_matrix * weights,
                            (1.0 - rho) * (1.0 - kappa) * mat + (1.0 - rho) * kappa,
                        )
                    ).sum()
                )
                term3 = (
                    -1.0
                    * (
                        torch.xlogy(
                            adj_matrix_1m,
                            (1.0 - kappa) * (1.0 - rho) * (1.0 - mat) + rho,
                        )
                    ).sum()
                )
            else:
                term2 = 0.0
                term3 = 0.0
            # the magnitude of lambda is proportional to the number of cells [todo: simpler to just take the mean instead of sum in term 1]
            loss = (
                loss
                + self.lam * term1
                + (self.cell_type_counts[cell_type] / float(self.n)) * (term2 + term3)
            )

        # compute loss associated with global graph
        adj_matrix = self.adj_matrix["global"]
        adj_matrix_1m = self.adj_matrix_1m["global"]
        weights = self.weights["global"]
        if len(adj_matrix) > 0:
            mat = contract("il,lj,kj->ik", theta_global, eta_global, theta_global)
            term2 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix * weights,
                        (1.0 - rho_global) * (1.0 - kappa_global) * mat
                        + (1.0 - rho_global) * kappa_global,
                    )
                ).sum()
            )
            term3 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix_1m,
                        (1.0 - kappa_global) * (1.0 - rho_global) * (1.0 - mat)
                        + rho_global,
                    )
                ).sum()
            )
            loss = loss + term2 + term3
        return loss

    def loss_no_cell_types(self, X):
        assert self.use_cell_types == False  # if this is True, just fail
        X = torch.Tensor(X)

        theta = torch.softmax(self.theta, dim=1)
        eta = self.eta.exp() / (1.0 + (self.eta).exp())
        eta = 0.5 * (eta + eta.T)
        gene_scaling = self.gene_scaling.exp() / (1.0 + self.gene_scaling.exp())
        kappa = self.kappa.exp() / (1 + self.kappa.exp())
        rho = self.rho.exp() / (1 + self.rho.exp())
        alpha = torch.exp(self.alpha)
        adj_matrix = self.adj_matrix
        weights = self.weights
        adj_matrix_1m = self.adj_matrix_1m
        theta_ = contract("jk,j->jk", theta, gene_scaling + self.delta)
        recon = contract("ik,jk->ij", alpha, theta_)
        term1 = -1.0 * (torch.xlogy(X, recon) - recon).sum()

        if len(adj_matrix) > 0:
            mat = contract("il,lj,kj->ik", theta, eta, theta)
            term2 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix * weights,
                        (1.0 - rho) * (1.0 - kappa) * mat + (1.0 - rho) * kappa,
                    )
                ).sum()
            )
            term3 = (
                -1.0
                * (
                    torch.xlogy(
                        adj_matrix_1m, (1.0 - kappa) * (1.0 - rho) * (1.0 - mat) + rho
                    )
                ).sum()
            )
        else:
            term2 = 0.0
            term3 = 0.0

        return self.lam * term1 + term2 + term3

    def initialize(self, gene_sets, val):
        """
        form of gene_sets:

        cell_type (inc. global) : set of sets of idxs
        """

        for ct in self.cell_types:
            assert self.L[ct] >= len(gene_sets[ct])
            count = 0
            if self.L[ct] > 0:
                if len(self.adj_matrix[ct]) > 0:
                    for gene_set in gene_sets[ct]:
                        self.theta[ct].data[:, count][gene_set] = val
                        count = count + 1
                    for i in range(self.L[ct]):
                        self.eta[ct].data[i, -1] = -val
                        self.eta[ct].data[-1, i] = -val
                    self.theta[ct].data[:, -1][self.adj_matrix[ct].sum(axis=1) == 0] = (
                        val
                    )
                    self.theta[ct].data[:, -1][self.adj_matrix[ct].sum(axis=1) != 0] = (
                        -val
                    )

        assert self.L["global"] >= len(gene_sets["global"])
        count = 0
        for gene_set in gene_sets["global"]:
            self.theta["global"].data[:, count][gene_set] = val
            count = count + 1
        for i in range(self.L["global"]):
            self.eta["global"].data[i, -1] = -val
            self.eta["global"].data[-1, i] = -val
        self.theta["global"].data[:, -1][self.adj_matrix["global"].sum(axis=1) == 0] = (
            val
        )
        self.theta["global"].data[:, -1][self.adj_matrix["global"].sum(axis=1) != 0] = (
            -val
        )

    def initialize_no_celltypes(self, gs_list, val):
        assert self.L >= len(gs_list)
        count = 0
        for gene_set in gs_list:
            self.theta.data[:, count][gene_set] = val
            count = count + 1
        for i in range(self.L):
            self.eta.data[i, -1] = -val
            self.eta.data[-1, i] = -val
        self.theta.data[:, -1][self.adj_matrix.sum(axis=1) == 0] = val
        self.theta.data[:, -1][self.adj_matrix.sum(axis=1) != 0] = -val


class spectra_attn(nn.Module):
    """
    Spectra v2: (1) attention based or (2) remove MMSB constraint allowing true multi membership for gene nodes (3) VI uncertainty
    > tradeoff between noisy gene sets and getting lowly expressed genes
    > tradeoff between simplex constraint and adding new genes - simplex allows using background factors
    > how to add new factors w/ attention based method, if you allow dropping
    """

    def __init__(self, X, K, gene_set_matrix, lambda_=1.0, d=10, lam=0.01):
        super(spectra_attn, self).__init__()
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.K = K
        self.d = d
        self.X = X
        self.gene_set_matrix = gene_set_matrix
        self.L = gene_set_matrix.shape[0]
        self.lambda_ = lambda_

        self.theta = nn.Parameter(Normal(0.0, 1.0).sample([self.n, self.K]))
        self.gamma = nn.Parameter(Normal(0.0, 1.0).sample([self.p, self.K]))
        self.query = nn.Parameter(Normal(0.0, 1.0).sample([self.K, self.d]))
        self.keys = nn.Parameter(Normal(0.0, 1.0).sample([self.L, self.d]))

    def loss(self):
        # compute the Poisson reconstruction loss
        gamma = self.gamma.exp()
        theta = self.theta.exp()
        reconstruction = contract("ij,kj->ik", theta, gamma)
        loss_1 = (-1 * torch.xlogy(self.X, reconstruction) + reconstruction).sum()

        # compute the attention dot products, K x L similarities
        similarities = contract("ij,kj->ik", self.query, self.keys) / torch.sqrt(self.d)
        similarities = similarities.softmax(dim=1)

        # compute assignments, K x p
        assignments = contract("ij,jk->ik", similarities, self.gene_set_matrix)

        # compute loss per assignments
        loss_2 = (-1 * torch.xlogy(self.assignments, self.gamma.T) + self.gamma.T).sum()
        return loss_1 + self.lambda_ * loss_2


class SPECTRA_Model:
    """

    Parameters
        ----------
        X : np.ndarray or torch.Tensor
            the ``(n, p)`` -shaped matrix containing logged expression count data. Used for initialization of self.n and self.p but not stored as an attribute
        labels : np.ndarray or NoneType
            the ``(n, )`` -shaped array containing cell type labels. If use_cell_types == False, then should be set to None

        L : dict or OrderedDict [if use_cell_types == False, then int]
            ``number of cell types + 1``-shaped dictionary. Must have "global" as a key, indicating the number of global factors
            {
                "global": 15,
                "CD8": 5
                ...
            }
            > Format matches output of K_est.py to estimate the number of
            > Must match cell type labels provided during training
            > Recommended practice is to assign at minimum 2 factors per cell type
            > L contains the number of factors that describe the graph.
        adj_matrix :  dict or OrderedDict
            ``a dictionary of adjacency matrices, one for every cell type + a "global"
            {
                "global": ``(p, p)``-shaped binary np.ndarray
                "CD8": ...

            }
        weights : dict or OrderedDict or NoneType [if use_cell_types == False, then ``(p, p)``-shaped array]
            the ``(p, p)``-shaped set of edge weights per . If weight[i,j] is non-zero when adj_matrix[i,j] = 0
            this weight is ignored.

            if weights == None, no weights are used
        lam : float
            lambda parameter of the model, which controls the relative influence of the graph vs expression loss functions. This term multiplies the expression loss, so smaller values of lambda upweight the prior information
        delta : float
            delta parameter of the model, which controls a lower bound for gene scaling factors. If delta is small then the maximum ratio between gene scaling factors is larger and lowly expressed genes can be put on the same scale as highly expressed genes.
        kappa : float or NoneType
            kappa controls the background rate of edges in the graph. if kappa is a float, kappa is fixed to the given float value. If kappa == None, then kappa is a parameter that is estimated from the data.
        rho : float or NoneType
            rho controls the bakcground rate of non-edges in the graph. if rho is a float, rho is fixed to the given float value. If rho == None, then rho is a parameter that is estimated from the data.
        use_cell_types: bool
            use_cell_types is a Boolean variable that determines whether cell type labels are used to fit the model. If False, then parameters are initialized as nn.Parameter rather than as nn.ParameterDict with cell type keys that index nn.Parameter values
        determinant_penalty : float
            determinant penalty affects the selection parameters that are fit when L[cell_type] > K[cell_type]. A determinantally regularized selection parameter is fit with determinant penalty that encourages sparsity and diversity.
    Attributes
        ----------
        model.delta : delta parameter of the model

        model.lam : lambda parameter of the model

        model.determinant_penalty : determinant penalty of the model

        model.L : L parameter, either int, dict or OrderedDict()

        model.p : number of genes

        model.n : number of cells

        model.use_cell_types : if True then cell types are considered, else cell types ignored. Affects the dimensions of the initialized parameters.

        model.kappa : if not kappa, nn.ParameterDict() if use_cell_types, else nn.Parameter(). If kappa is a float, it is fixed throughout training

        model.rho : if not rho, nn.ParamterDict() if use_cell_types, else nn.Parameter. If rho is a float it is fixed throughout training

        model.adj_matrix : adjacency matrix with diagonal removed. dict containing torch.Tensors

        model.adj_matrix_1m : 1 - adjacency matrix with diagonal removed. dict containing torch.Tensors

        model.weights : contains edge weights. format matches adj_matrix

        model.cell_types : np.ndarray containing array of unique cell types

        model.cell_type_counts : dict {key = cell type, values = number of cells}

        model.factors : nn.ParameterDict() or nn.Parameter() containing the factor weights

        model.cell_scores : nn.ParameterDict() or nn.Parameter() containing the cell loadings

        model.eta : nn.ParameterDict() or nn.Parameter() containing the interaction matrix between factors

        model.gene_scaling : nn.ParameterDict() or nn.Parameter() containing the gene scale factors

        model.selection :  nn.ParameterDict() or nn.Parameter() containing the attention weights. Only initialized when L[cell_type] > K[cell_type] for some cell type or when L > K and use_cell_types == False



    Methods
        ----------

        model.train(self, X, labels, lr_schedule,num_epochs, verbose) :
        model.save()
        model.load()
        model.initialize
        model.return_selection()
        model.return_eta_diag()
        model.return_cell_scores()
        model.return_factors()
        model.return_eta()
        model.return_rho()
        model.return_kappa()
        model.return_gene_scalings()
        model.return_graph(ct = "global") :
        model.matching(markers, gene_names_dict, threshold = 0.4):

    """

    def __init__(
        self,
        X,
        labels,
        L,
        vocab=None,
        gs_dict=None,
        use_weights=True,
        adj_matrix=None,
        weights=None,
        lam=0.01,
        delta=0.001,
        kappa=None,
        rho=0.001,
        use_cell_types=True,
    ):
        self.L = L
        self.lam = lam
        self.delta = delta
        self.kappa = kappa
        self.rho = rho
        self.use_cell_types = use_cell_types

        # if gs_dict is provided instead of adj_matrix, convert to adj_matrix, overrides adj_matrix and weights
        if gs_dict is not None:
            gene2id = dict((v, idx) for idx, v in enumerate(vocab))

            if use_cell_types:
                adj_matrix, weights = Spectra_util.process_gene_sets(
                    gs_dict=gs_dict, gene2id=gene2id, weighted=use_weights
                )
            else:
                adj_matrix, weights = Spectra_util.process_gene_sets_no_celltypes(
                    gs_dict=gs_dict, gene2id=gene2id, weighted=use_weights
                )

        self.internal_model = SPECTRA(
            X=X,
            labels=labels,
            adj_matrix=adj_matrix,
            L=L,
            weights=weights,
            lam=lam,
            delta=delta,
            kappa=kappa,
            rho=rho,
            use_cell_types=use_cell_types,
        )

        self.cell_scores = None
        self.factors = None
        self.B_diag = None
        self.eta_matrices = None
        self.gene_scalings = None
        self.rho = None
        self.kappa = None

    def train(
        self,
        X,
        labels=None,
        lr_schedule=[1.0, 0.5, 0.1, 0.01, 0.001, 0.0001],
        num_epochs=10000,
        verbose=False,
    ):
        opt = torch.optim.Adam(self.internal_model.parameters(), lr=lr_schedule[0])
        counter = 0
        last = np.inf

        for i in tqdm(range(num_epochs)):
            # print(counter)
            opt.zero_grad()
            if self.internal_model.use_cell_types:
                assert len(labels) == X.shape[0]
                loss = self.internal_model.loss(X, labels)
            elif self.internal_model.use_cell_types == False:
                loss = self.internal_model.loss_no_cell_types(X)

            loss.backward()
            opt.step()

            if loss.item() >= last:
                counter += 1
                if int(counter / 3) >= len(lr_schedule):
                    break
                if counter % 3 == 0:
                    opt = torch.optim.Adam(
                        self.internal_model.parameters(),
                        lr=lr_schedule[int(counter / 3)],
                    )
                    if verbose:
                        print("UPDATING LR TO " + str(lr_schedule[int(counter / 3)]))
            last = loss.item()

        # add all model parameters as attributes

        if self.use_cell_types:
            self.__store_parameters(labels)
        else:
            self.__store_parameters_no_celltypes()

    def save(self, fp):
        torch.save(self.internal_model.state_dict(), fp)

    def load(self, fp, labels=None):
        self.internal_model.load_state_dict(torch.load(fp))
        if self.use_cell_types:
            assert labels is not None
            self.__store_parameters(labels)
        else:
            self.__store_parameters_no_celltypes()

    def __store_parameters(self, labels):
        """
        Replaces __cell_scores() and __compute factors() and __compute_theta()
        store parameters after fitting the model:
            cell scores
            factors
            eta
            scalings
            gene scalings
            kappa
            rho
        """

        model = self.internal_model

        # compute the loading matrix

        k = sum(list(model.L.values()))
        out = np.zeros((model.n, k))

        global_idx = model.L["global"]

        tot = global_idx
        f = ["global"] * model.L["global"]
        for cell_type in model.cell_types:
            alpha = torch.exp(model.alpha[cell_type]).detach().numpy()
            out[labels == cell_type, :global_idx] = alpha[:, :global_idx]
            out[labels == cell_type, tot : tot + model.L[cell_type]] = alpha[
                :, global_idx:
            ]

            tot += model.L[cell_type]

            f = f + [cell_type] * model.L[cell_type]

        out2 = np.zeros((k, model.p))

        theta_ct = torch.softmax(model.theta["global"], dim=1)
        theta = theta_ct.detach().numpy().T
        tot = 0
        out2[0 : theta.shape[0], :] = theta
        tot += theta.shape[0]

        for cell_type in model.cell_types:
            theta_ct = torch.softmax(model.theta[cell_type], dim=1)
            theta = theta_ct.detach().numpy().T
            out2[tot : tot + theta.shape[0], :] = theta
            tot += theta.shape[0]

        factors = out2
        lst = []
        for i in range(len(f)):
            ct = f[i]
            scaled = (
                factors[i, :]
                * (
                    model.gene_scaling[ct].exp().detach()
                    / (1.0 + model.gene_scaling[ct].exp().detach())
                    + model.delta
                ).numpy()
            )

            lst.append(scaled)
        scaled = np.array(lst)
        new_factors = scaled / (scaled.sum(axis=0, keepdims=True) + 1.0)
        cell_scores = out * scaled.mean(axis=1).reshape(1, -1)
        self.cell_scores = cell_scores
        self.factors = new_factors
        self.B_diag = self.__B_diag()
        self.eta_matrices = self.__eta_matrices()
        self.gene_scalings = {
            ct: model.gene_scaling[ct].exp().detach().numpy()
            / (1.0 + model.gene_scaling[ct].exp().detach().numpy())
            for ct in model.gene_scaling.keys()
        }
        self.rho = {
            ct: model.rho[ct].exp().detach().numpy()
            / (1.0 + model.rho[ct].exp().detach().numpy())
            for ct in model.rho.keys()
        }
        self.kappa = {
            ct: model.kappa[ct].exp().detach().numpy()
            / (1.0 + model.kappa[ct].exp().detach().numpy())
            for ct in model.kappa.keys()
        }

    def __B_diag(self):
        model = self.internal_model
        k = sum(list(model.L.values()))
        out = np.zeros(k)

        Bg = model.eta["global"].exp() / (1.0 + model.eta["global"].exp())
        Bg = 0.5 * (Bg + Bg.T)
        B = torch.diag(Bg).detach().numpy()
        tot = 0
        out[0 : B.shape[0]] = B
        tot += B.shape[0]

        for cell_type in model.cell_types:
            Bg = model.eta[cell_type].exp() / (1.0 + model.eta[cell_type].exp())
            Bg = 0.5 * (Bg + Bg.T)
            B = torch.diag(Bg).detach().numpy()
            out[tot : tot + B.shape[0]] = B

            tot += B.shape[0]

        return out

    def __eta_matrices(self):
        model = self.internal_model
        eta = OrderedDict()
        Bg = model.eta["global"].exp() / (1.0 + model.eta["global"].exp())
        Bg = 0.5 * (Bg + Bg.T)
        eta["global"] = Bg.detach().numpy()

        for cell_type in model.cell_types:
            Bg = model.eta[cell_type].exp() / (1.0 + model.eta[cell_type].exp())
            Bg = 0.5 * (Bg + Bg.T)
            eta[cell_type] = Bg.detach().numpy()
        return eta

    def __store_parameters_no_celltypes(self):
        """
        store parameters after fitting the model:
            cell scores
            factors
            eta
            scalings
            gene scalings
            kappa
            rho
        """
        model = self.internal_model
        theta_ct = torch.softmax(model.theta, dim=1)
        theta = theta_ct.detach().numpy().T
        alpha = torch.exp(model.alpha).detach().numpy()
        out = alpha
        factors = theta

        scaled = factors * (
            model.gene_scaling.exp().detach()
            / (1.0 + model.gene_scaling.exp().detach())
            + model.delta
        ).numpy().reshape(1, -1)
        new_factors = scaled / (scaled.sum(axis=0, keepdims=True) + 1.0)

        self.factors = new_factors
        self.cell_scores = out * scaled.mean(axis=1).reshape(1, -1)
        Bg = model.eta.exp() / (1.0 + model.eta.exp())
        Bg = 0.5 * (Bg + Bg.T)
        self.B_diag = torch.diag(Bg).detach().numpy()
        self.eta_matrices = Bg.detach().numpy()
        self.gene_scalings = (
            model.gene_scaling.exp().detach()
            / (1.0 + model.gene_scaling.exp().detach())
        ).numpy()
        self.rho = (model.rho.exp().detach() / (1.0 + model.rho.exp().detach())).numpy()
        self.kappa = (
            model.kappa.exp().detach() / (1.0 + model.kappa.exp().detach())
        ).numpy()

    def initialize(self, annotations, word2id, W, init_scores, val=25):
        """
        self.use_cell_types must be True
        create form of gene_sets:

        cell_type (inc. global) : set of sets of idxs

        filter based on L_ct
        """
        if self.use_cell_types:
            if init_scores == None:
                init_scores = compute_init_scores(annotations, word2id, torch.Tensor(W))
            gs_dict = OrderedDict()
            for ct in annotations.keys():
                mval = max(self.L[ct] - 1, 0)
                sorted_init_scores = sorted(init_scores[ct].items(), key=lambda x: x[1])
                sorted_init_scores = sorted_init_scores[-1 * mval :]
                names = set([k[0] for k in sorted_init_scores])
                lst_ct = []
                for key in annotations[ct].keys():
                    if key in names:
                        words = annotations[ct][key]
                        idxs = []
                        for word in words:
                            if word in word2id:
                                idxs.append(word2id[word])
                        lst_ct.append(idxs)
                gs_dict[ct] = lst_ct
            self.internal_model.initialize(gene_sets=gs_dict, val=val)
        else:
            if init_scores == None:
                init_scores = compute_init_scores_noct(
                    annotations, word2id, torch.Tensor(W)
                )
            lst = []
            mval = max(self.L - 1, 0)
            sorted_init_scores = sorted(init_scores.items(), key=lambda x: x[1])
            sorted_init_scores = sorted_init_scores[-1 * mval :]
            names = set([k[0] for k in sorted_init_scores])
            for key in annotations.keys():
                if key in names:
                    words = annotations[key]
                    idxs = []
                    for word in words:
                        if word in word2id:
                            idxs.append(word2id[word])
                    lst.append(idxs)
            self.internal_model.initialize_no_celltypes(gs_list=lst, val=val)

    def return_eta_diag(self):
        return self.B_diag

    def return_cell_scores(self):
        return self.cell_scores

    def return_factors(self):
        return self.factors

    def return_eta(self):
        return self.eta_matrices

    def return_rho(self):
        return self.rho

    def return_kappa(self):
        return self.kappa

    def return_gene_scalings(self):
        return self.gene_scalings

    def return_graph(self, ct="global"):
        model = self.internal_model
        if self.use_cell_types:
            eta = (model.eta[ct]).exp() / (1.0 + (model.eta[ct]).exp())
            eta = 0.5 * (eta + eta.T)
            theta = torch.softmax(model.theta[ct], dim=1)
            mat = contract("il,lj,kj->ik", theta, eta, theta).detach().numpy()
        else:
            eta = model.eta.exp() / (1.0 + model.eta.exp())
            eta = 0.5 * (eta + eta.T)
            theta = torch.softmax(model.theta, dim=1)
            mat = contract("il,lj,kj->ik", theta, eta, theta).detach().numpy()
        return mat

    def matching(self, markers, gene_names_dict, threshold=0.4):
        """
        best match based on overlap coefficient
        """
        markers = pd.DataFrame(markers)
        if self.use_cell_types:
            matches = []
            jaccards = []
            for i in range(markers.shape[0]):
                max_jacc = 0.0
                best = ""
                for key in gene_names_dict.keys():
                    for gs in gene_names_dict[key].keys():
                        t = gene_names_dict[key][gs]

                        jacc = Spectra_util.overlap_coefficient(
                            list(markers.iloc[i, :]), t
                        )
                        if jacc > max_jacc:
                            max_jacc = jacc
                            best = gs
                matches.append(best)
                jaccards.append(max_jacc)

        else:
            matches = []
            jaccards = []
            for i in range(markers.shape[0]):
                max_jacc = 0.0
                best = ""
                for key in gene_names_dict.keys():
                    t = gene_names_dict[key]

                    jacc = Spectra_util.overlap_coefficient(list(markers.iloc[i, :]), t)
                    if jacc > max_jacc:
                        max_jacc = jacc
                        best = key
                matches.append(best)
                jaccards.append(max_jacc)
        output = []
        for j in range(markers.shape[0]):
            if jaccards[j] > threshold:
                output.append(matches[j])
            else:
                output.append("0")
        return np.array(output)


class SPECTRA_EM:
    """
    non-integrative SPECTRA -- EM Routine

    Usage:

    model = EMSPADE(X = X, A=A, K = K, lam = 0.5, T= 15, kappa = 0)
    model.fit(n_epochs = 200)


    Notes
    _________________

    > EM is recommended for problems where K is smaller [as the memory scales with K**2, as opposed to K for GD]
        e.g. when we are only interested in 1 cell type

    > EM does not currently support estimation of rho and kappa (this was added after the paper)

    > We notice more stable estimates from EM in general

    """

    def __init__(
        self,
        X,
        A,
        weights=None,
        K=10,
        delta=0.001,
        kappa=None,
        rho=0.001,
        lam=0.01,
        T=3,
    ):
        self.EPS = 0.0
        # fixed constants
        self.K = K
        self.delta = delta
        self.kappa = kappa
        self.lam = lam
        self.rho = rho
        self.T = T
        self.n = X.shape[0]
        self.p = X.shape[1]

        # data
        self.X = X
        self.A = A - np.diag(np.diag(A))
        A1m = 1 - A
        self.A1m = A1m - np.diag(np.diag(A1m))
        if weights is not None:
            self.weights = weights
        else:
            self.weights = self.A

        # random initializations
        self.theta = np.random.normal(size=(self.p, self.K))
        self.theta = softmax(self.theta, axis=1)
        self.alpha = np.exp(np.random.normal(size=(self.n, self.K)))
        self.g = np.random.normal(size=self.p)
        self.g = np.exp(self.g) / (1.0 + np.exp(self.g))
        self.B = np.random.normal(size=(self.K, self.K))
        self.B = np.exp(self.B) / (1.0 + np.exp(self.B))
        self.B = 0.5 * (self.B + (self.B).T)  # make symmetric

        # initialize pseudo-data
        self.phi = np.exp(np.random.normal(size=(self.n, self.p, self.K)))
        self.phi_tilde = np.exp(np.random.normal(size=(self.p, self.p, self.K, self.K)))
        for j in range(self.p):
            self.phi_tilde[j, j, :, :] = np.zeros((self.K, self.K))

    def M_step(self):
        for t in range(self.T):
            self.update_theta_newton()
        self.update_B()
        self.update_alpha()
        self.update_g()

        return

    def E_step(self):
        """
        need to do all computations on the ln scale

        get up to proportionality then normalize - custom softmax for phi_tilde

        keep self.phi_tilde[i,i,:,:] = 0s  <-- ** really important **
        """

        # phi update
        alpha_theta = contract("ik,jk->ijk", self.alpha, self.theta)
        multinomial_parameter = alpha_theta / (
            alpha_theta.sum(axis=2, keepdims=True) + self.EPS
        )
        multinomial_parameter[~np.isfinite(multinomial_parameter)] = 0
        self.phi = contract("ij,ijk->ijk", self.X, multinomial_parameter)

        # phi_tilde update
        theta_prods = contract("ik,jl->ijkl", self.theta, self.theta)
        log_term2 = contract(
            "ij,ij,kl->ijkl",
            self.A,
            self.weights,
            np.log((1 - self.kappa) * self.B + self.kappa),
        )
        log_term2[~np.isfinite(log_term2)] = 0  # A should be 0 anyways, so
        log_term3 = contract(
            "ij,kl->ijkl",
            self.A1m,
            np.log((1 - self.kappa) * (1 - self.rho) * (1.0 - self.B) + self.rho),
        )
        log_term3[~np.isfinite(log_term3)] = 0
        log_term4 = log_term2 + log_term3
        theta_prods = theta_prods * np.exp(log_term4)
        self.phi_tilde = theta_prods / (
            theta_prods.sum(axis=(2, 3), keepdims=True) + self.EPS
        )
        self.phi_tilde[~np.isfinite(self.phi_tilde)] = 0
        for j in range(self.p):
            self.phi_tilde[j, j, :, :] = np.zeros((self.K, self.K))

        return

    def update_theta_newton(self):
        """
        Runs one step of a simplex constrained newton raphson algorithm to update theta
        """
        """
        evidence = self.phi.sum(axis = 0) + self.lam*(self.phi_tilde.sum(axis = (1,3)) + self.phi_tilde.sum(axis = (0,2))) # P x K
        grad_ = evidence/(self.theta + self.EPS) + (self.g + self.delta).reshape(-1,1)*(self.alpha.sum(axis = 0).reshape(1,-1))  #analytic gradient
        #grad_[~np.isfinite(grad_)] = 0
        inverse_hessian = -1.0*(self.theta**2)/(evidence + self.EPS)#analytic second derivative
        #inverse_hessian[~np.isfinite(inverse_hessian)] = 0
        gh_inv = (inverse_hessian*grad_).sum(axis = 1) #P
        h_inv = inverse_hessian.sum(axis = 1)
        #h_inv[~np.isfinite(h_inv)] = 0
        lagrange_multiplier = -1.0*gh_inv/(h_inv + self.EPS)#enforces simplex constraint
        #lagrange_multiplier[~np.isfinite(lagrange_multiplier)] = 0
        theta = self.theta - inverse_hessian*(lagrange_multiplier.reshape(-1,1) + grad_)
        self.theta = theta
        self.theta = self.theta/self.theta.sum(axis = 1, keepdims = True) #ensure sum to 1 constraint. - without numerical error does nothing
        """

        evidence = self.phi.sum(axis=0) + self.lam * (
            self.phi_tilde.sum(axis=(1, 3)) + self.phi_tilde.sum(axis=(0, 2))
        )  # P x K
        inverse_hessian = (
            -1.0 * (self.theta**2) / (evidence + self.EPS)
        )  # analytic second derivative
        inverse_hessian[~np.isfinite(inverse_hessian)] = 0  # control instability
        gh = -1.0 * (
            self.theta
            - (self.theta**2)
            / evidence
            * (self.g + self.delta).reshape(-1, 1)
            * (self.alpha.sum(axis=0).reshape(1, -1))
        )
        gh[~np.isfinite(gh)] = 0  # control instability
        gh_inv = gh.sum(axis=1)
        h_inv = inverse_hessian.sum(axis=1)
        lagrange_multiplier = (
            -1.0 * gh_inv / (h_inv + self.EPS)
        )  # enforces simplex constraint
        theta = self.theta - inverse_hessian * lagrange_multiplier.reshape(-1, 1) - gh
        self.theta = theta
        self.theta[self.theta < 0] = 0.0
        self.theta = self.theta / self.theta.sum(
            axis=1, keepdims=True
        )  # ensure sum to 1 constraint. - without numerical error does nothing
        return

    def update_B(self):
        """
        last step must truncate for constraints
        """
        evidence_num = contract("ijkl,ij,ij->kl", self.phi_tilde, self.weights, self.A)
        evidence_denom = contract("ijkl,ij->kl", self.phi_tilde, self.A1m)
        # evidence = evidence_num/(evidence_denom + self.EPS)
        # evidence[~np.isfinite(evidence)] = 0
        self.B = (
            (self.rho / (1.0 - self.rho) + 1.0 - self.kappa) * evidence_num
            - self.kappa * evidence_denom
        ) / ((1.0 - self.kappa) * (evidence_num + evidence_denom) + self.EPS)
        self.B[~np.isfinite(self.B)] = 0
        self.B[self.B > 1] = 1.0
        self.B[self.B < 0] = 0.0
        return

    def update_g(self):
        """
        last step must truncate for constraints
        """

        num_ = self.phi.sum(axis=(0, 2))  # P
        denom_ = contract("ik,jk->j", self.alpha, self.theta)  # P
        self.g = num_ / (denom_ + self.EPS) - self.delta
        # self.g[~np.isfinite(self.g)] = 0
        # enforce constraints - strictly convex--> suffices to truncate
        self.g[self.g > 1] = 1.0
        self.g[self.g < 0] = 0.0
        return

    def update_alpha(self):
        num_ = self.phi.sum(axis=1)  # N x K
        denom_ = contract("jk,j->k", self.theta, self.g + self.delta).reshape(
            1, -1
        )  # 1 x K
        self.alpha = num_ / (denom_ + self.EPS)  # N x K
        # self.alpha[~np.isfinite(self.alpha)] = 0
        return

    def NLL(self):
        recon = contract("ik,jk,j->ij", self.alpha, self.theta, self.g + self.delta)
        term1 = -1.0 * (xlogy(self.X, recon) - recon).sum()
        mat = contract("il,lj,kj->ik", self.theta, self.B, self.theta)
        term2 = (
            -1.0
            * (
                np.log((1.0 - self.kappa) * mat + self.kappa) * self.A * self.weights
            ).sum()
        )
        term3 = (
            -1.0
            * (
                np.log((1.0 - self.kappa) * (1.0 - self.rho) * (1.0 - mat) + self.rho)
                * self.A1m
            ).sum()
        )
        return term1 + self.lam * (term2 + term3)

    def fit(self, n_epochs=10000, thres=1, suppress=True):
        """
        must run E step first
        """
        prev_nll = np.inf
        for t in range(n_epochs):
            # print("E_step")
            self.E_step()
            # print("M_step")
            self.M_step()
            # print(model.B.max())

            if t % 1 == 0:
                nll = self.NLL()
                if not suppress:
                    print("NLL: ", round(nll, 3))
                if prev_nll - nll < thres:
                    return
                else:
                    prev_nll = nll
        return

    def initialize(self, gene_sets, val=10):
        ct = 0
        for gene_set in gene_sets:
            self.theta[np.array(gene_set), ct] = val
            self.B[ct, ct] = 1.0
            ct = ct + 1

        self.theta = self.theta / (self.theta.sum(axis=1, keepdims=True))

    def markers(self, offset, id2gene, n_top):
        # compute marker weights
        marker_weights = self.theta * (self.g + self.delta).reshape(-1, 1) + offset
        marker_weights = marker_weights / marker_weights.sum(axis=1, keepdims=True)

        # print genes per factors
        df = pd.DataFrame()
        for j in range(self.K):
            df["factor" + str(j)] = [
                id2gene[i] for i in np.argsort(marker_weights[:, j])[::-1][:n_top]
            ]
        return df


""" 
Public Functions 


    est_spectra():

    matching(): 

    graph_network():  

    markers


"""


def get_factor_celltypes(adata, obs_key, cellscore):
    """
    Assigns Spectra factors to cell types by analyzing the factor cell scores.
    Cell type specific factors will have zero cell scores except in their respective cell type

    adata: AnnData , object containing the Spectra output
    obs_key: str , column name in adata.obs containing the cell type annotations
    cellscore_obsm_key: str , key for adata.obsm containing the Spectra cell scores

    returns: dict , dictionary of {factor index : 'cell type'}
    """

    # get cellscores
    cell_scores_df = pd.DataFrame(cellscore)
    cell_scores_df["celltype"] = list(adata.obs[obs_key])

    # find global and cell type specific fators
    global_factors_series = (cell_scores_df.groupby("celltype").mean() != 0).all()
    global_factors = [
        factor
        for factor in global_factors_series.index
        if global_factors_series[factor]
    ]

    specific_factors = {}
    if len(global_factors) != (len(cell_scores_df.columns) - 1):
        specific_cell_scores = (
            (cell_scores_df.groupby("celltype").mean()).T[~global_factors_series].T
        )

        for i in set(cell_scores_df["celltype"]):
            specific_factors[i] = [
                factor
                for factor in specific_cell_scores.loc[i].index
                if specific_cell_scores.loc[i, factor]
            ]

        # inverse dict factor:celltype
        factors_inv = {}
        for i, v in specific_factors.items():
            for factor in v:
                factors_inv[factor] = i

        # add global

        for factor in global_factors:
            factors_inv[factor] = "global"

    else:
        factors_inv = {}
        for factor in global_factors:
            factors_inv[factor] = "global"

    return factors_inv


def est_spectra(
    adata,
    gene_set_dictionary,
    L=None,
    use_highly_variable=True,
    cell_type_key="cell_type_annotations",
    use_weights=True,
    lam=0.01,
    delta=0.001,
    kappa=None,
    rho=0.001,
    use_cell_types=True,
    n_top_vals=50,
    filter_sets=True,
    label_factors=True,
    clean_gs=True,
    min_gs_num=3,
    overlap_threshold=0.2,
    **kwargs,
):
    """

    Parameters
        ----------
        adata : AnnData object
            containing cell_type_key with log count data stored in .X
        gene_set_dictionary : dict or OrderedDict()
            maps cell types to gene set names to gene sets ; if use_cell_types == False then maps gene set names to gene sets ;
            must contain "global" key in addition to every unique cell type under .obs.<cell_type_key>
        L : dict, OrderedDict(), int , NoneType
            number of factors per cell type ; if use_cell_types == False then int. Else dictionary. If None then match factors
            to number of gene sets (recommended)
        use_highly_variable : bool
            if True, then uses highly_variable_genes
        cell_type_key : str
            cell type key, must be under adata.obs.<cell_type_key> . If use_cell_types == False, this is ignored
        use_weights : bool
            if True, edge weights are estimated based on graph structure and used throughout training
        lam : float
            lambda parameter of the model. weighs relative contribution of graph and expression loss functions
        delta : float
            delta parameter of the model. lower bounds possible gene scaling factors so that maximum ratio of gene scalings
            cannot be too large
        kappa : float or None
            if None, estimate background rate of 1s in the graph from data
        rho : float or None
            if None, estimate background rate of 0s in the graph from data
        use_cell_types : bool
            if True then cell type label is used to fit cell type specific factors. If false then cell types are ignored
        n_top_vals : int
            number of top markers to return in markers dataframe
        determinant_penalty : float
            determinant penalty of the attention mechanism. If set higher than 0 then sparse solutions of the attention weights
            and diverse attention weights are encouraged. However, tuning is crucial as setting too high reduces the selection
            accuracy because convergence to a hard selection occurs early during training [todo: annealing strategy]
        filter_sets : bool
            whether to filter the gene sets based on coherence
        label_factors : bool
            whether to label the factors by their cell type specificity and their overlap coefficient with the input marker genes
        clean_gs : bool
            if True cleans up the gene set dictionary to:
                1. checks that annotations dictionary cell type keys and adata cell types are identical.
                2. to contain only genes contained in the adata
                3. to contain only gene sets greater length min_gs_num
        min_gs_num : int
            only use if clean_gs True, minimum number of genes per gene set expressed in adata, other gene sets will be filtered out
        overlap_threshold: float
            minimum overlap coefficient to assign an input gene set label to a factor

        **kwargs : (num_epochs = 10000, lr_schedule = [...], verbose = False)
            arguments to .train(), maximum number of training epochs, learning rate schedule and whether to print changes in
            learning rate

     Returns: SPECTRA_Model object [after training]

     In place: adds 1. factors, 2. cell scores, 3. vocabulary, 4. markers, 5. overlap coefficient of markers vs input gene sets as attributes in .obsm, .var, .uns


    """

    # filter gene set dictionary
    if clean_gs:
        gene_set_dictionary = Spectra_util.check_gene_set_dictionary(
            adata,
            gene_set_dictionary,
            obs_key=cell_type_key,
            global_key="global",
            return_dict=True,
            min_len=min_gs_num,
            use_cell_types=use_cell_types,
        )
    if L == None:
        init_flag = True
        if use_cell_types:
            L = {}
            for key in gene_set_dictionary.keys():
                length = len(list(gene_set_dictionary[key].values()))
                L[key] = length + 1
        else:
            length = len(list(gene_set_dictionary.values()))
            L = length + 1
    # create vocab list from gene_set_dictionary
    lst = []
    if use_cell_types:
        for key in gene_set_dictionary:
            for key2 in gene_set_dictionary[key]:
                gene_list = gene_set_dictionary[key][key2]
                lst += gene_list
    else:
        for key in gene_set_dictionary:
            gene_list = gene_set_dictionary[key]
            lst += gene_list

    # lst contains all of the genes that are in the gene sets --> convert to boolean array
    bools = []
    for gene in adata.var_names:
        if gene in lst:
            bools.append(True)
        else:
            bools.append(False)
    bools = np.array(bools)

    if use_highly_variable:
        idx_to_use = (
            bools | adata.var.highly_variable
        )  # take union of highly variable and gene set genes (todo: add option to change this at some point)
        X = adata.X[:, idx_to_use]
        vocab = adata.var_names[idx_to_use]
        adata.var["spectra_vocab"] = idx_to_use
    else:
        X = adata.X
        vocab = adata.var_names

    if use_cell_types:
        labels = adata.obs[cell_type_key].values
        for label in np.unique(labels):
            if label not in gene_set_dictionary:
                gene_set_dictionary[label] = {}
            if label not in L:
                L[label] = 1
    else:
        labels = None
    if type(X) == scipy.sparse.csr.csr_matrix:
        X = np.array(X.todense())
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    if filter_sets:
        if use_cell_types:
            new_gs_dict = {}
            init_scores = compute_init_scores(
                gene_set_dictionary, word2id, torch.Tensor(X)
            )
            for ct in gene_set_dictionary.keys():
                new_gs_dict[ct] = {}
                mval = max(L[ct] - 1, 0)
                sorted_init_scores = sorted(init_scores[ct].items(), key=lambda x: x[1])
                sorted_init_scores = sorted_init_scores[-1 * mval :]
                names = set([k[0] for k in sorted_init_scores])
                for key in gene_set_dictionary[ct].keys():
                    if key in names:
                        new_gs_dict[ct][key] = gene_set_dictionary[ct][key]
        else:
            init_scores = compute_init_scores_noct(
                gene_set_dictionary, word2id, torch.Tensor(X)
            )
            new_gs_dict = {}
            mval = max(L - 1, 0)
            sorted_init_scores = sorted(init_scores.items(), key=lambda x: x[1])
            sorted_init_scores = sorted_init_scores[-1 * mval :]
            names = set([k[0] for k in sorted_init_scores])
            for key in gene_set_dictionary.keys():
                if key in names:
                    new_gs_dict[key] = gene_set_dictionary[key]
        gene_set_dictionary = new_gs_dict
    else:
        init_scores = None

    spectra = SPECTRA_Model(
        X=X,
        labels=labels,
        L=L,
        vocab=vocab,
        gs_dict=gene_set_dictionary,
        use_weights=use_weights,
        lam=lam,
        delta=delta,
        kappa=kappa,
        rho=rho,
        use_cell_types=use_cell_types,
    )

    spectra.initialize(gene_set_dictionary, word2id, X, init_scores)

    spectra.train(X=X, labels=labels, **kwargs)

    adata.uns["SPECTRA_factors"] = spectra.factors
    adata.uns["SPECTRA_markers"] = return_markers(
        factor_matrix=spectra.factors, id2word=id2word, n_top_vals=n_top_vals
    )
    adata.uns["SPECTRA_L"] = L

    # label factors

    # transform input nested dictionary into a flat dictionary
    gene_set_dictionary_flat = {}

    for k, v in gene_set_dictionary.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                gene_set_dictionary_flat[k2] = v2
            is_global = False
        else:
            gene_set_dictionary_flat[k] = v
            is_global = True

    # labeling function
    if label_factors:
        # get cell type specificity of every factor
        if is_global == False:
            celltype_dict = get_factor_celltypes(
                adata, cell_type_key, cellscore=spectra.cell_scores
            )
            max_celltype = [
                celltype_dict[x] for x in range(spectra.cell_scores.shape[1])
            ]
        else:
            max_celltype = ["global"] * (spectra.cell_scores.shape[1])
        # get gene set with maximum overlap coefficient with top marker genes
        overlap_df = Spectra_util.label_marker_genes(
            adata.uns["SPECTRA_markers"],
            gene_set_dictionary_flat,
            threshold=overlap_threshold,
        )

        # create new column labels
        column_labels = []
        for i in range(len(spectra.cell_scores.T)):
            column_labels.append(
                str(i)
                + "-X-"
                + str(max_celltype[i])
                + "-X-"
                + str(list(overlap_df.index)[i])
            )

        overlap_df.index = column_labels
        adata.uns["SPECTRA_overlap"] = overlap_df

    adata.obsm["SPECTRA_cell_scores"] = spectra.cell_scores

    return spectra


def return_markers(factor_matrix, id2word, n_top_vals=100):
    idx_matrix = np.argsort(factor_matrix, axis=1)[:, ::-1][:, :n_top_vals]
    df = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df.iloc[i, j] = id2word[idx_matrix[i, j]]
    return df.values


def load_from_pickle(fp, adata, gs_dict, cell_type_key):
    model = SPECTRA_Model(
        X=adata[:, adata.var["spectra_vocab"]].X,
        labels=np.array(adata.obs[cell_type_key]),
        L=adata.uns["SPECTRA_L"],
        vocab=adata.var_names[adata.var["spectra_vocab"]],
        gs_dict=gs_dict,
    )
    model.load(fp, labels=np.array(adata.obs[cell_type_key]))
    return model


def graph_network(adata, mat, gene_set, thres=0.20, N=50):

    vocab = adata.var_names[adata.var["spectra_vocab"]]
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=True,
    )
    net.barnes_hut()

    idxs = []
    for term in gene_set:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs, :].sum(axis=0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0
    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label=id2word[est], color="#00ff1e")
        else:
            net.add_node(count, label=id2word[est], color="#162347")
        count += 1

    inferred_mat = mat[ests, :][:, ests]
    for i in range(len(inferred_mat)):
        for j in range(i + 1, len(inferred_mat)):
            if inferred_mat[i, j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    return net


def graph_network_multiple(adata, mat, gene_sets, thres=0.20, N=50):
    gene_set = []
    for gs in gene_sets:
        gene_set += gs

    vocab = adata.var_names[adata.var["spectra_vocab"]]
    word2id = dict((v, idx) for idx, v in enumerate(vocab))
    id2word = dict((idx, v) for idx, v in enumerate(vocab))

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=True,
    )
    net.barnes_hut()
    idxs = []
    for term in gene_set:
        idxs.append(word2id[term])
    ests = list(set(list(mat[idxs, :].sum(axis=0).argsort()[::-1][:N]) + idxs))
    ests_names = []
    count = 0

    color_map = []
    for gene_set in gene_sets:
        random_color = [
            "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
        ]
        color_map.append(random_color[0])

    for est in ests:
        ests_names.append(id2word[est])
        if est not in idxs:
            net.add_node(count, label=id2word[est], color="#00ff1e")
        else:
            for i in range(len(gene_sets)):
                if id2word[est] in gene_sets[i]:
                    color = color_map[i]
                    break
            net.add_node(count, label=id2word[est], color=color)
        count += 1

    inferred_mat = mat[ests, :][:, ests]
    for i in range(len(inferred_mat)):
        for j in range(i + 1, len(inferred_mat)):
            if inferred_mat[i, j] > thres:
                net.add_edge(i, j)
    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    return net


def gene_set_graph(gene_sets):
    """
    input
    [
    ["a","b", ... ],
    ["b", "d"],

    ...
    ]
    """

    net = Network(
        height="750px",
        width="100%",
        bgcolor="#FFFFFF",
        font_color="black",
        notebook=True,
    )
    net.barnes_hut()
    count = 0
    # create nodes
    genes = []
    for gene_set in gene_sets:
        genes += gene_set

    color_map = []
    for gene_set in gene_sets:
        random_color = [
            "#" + "".join([random.choice("ABCDEF0123456789") for i in range(6)])
        ]
        color_map.append(random_color[0])

    for gene in genes:
        for i in range(len(gene_sets)):
            if gene in gene_sets[i]:
                color = color_map[i]
                break
        net.add_node(gene, label=gene, color=color)

    for gene_set in gene_sets:
        for i in range(len(gene_set)):
            for j in range(i + 1, len(gene_set)):
                net.add_edge(gene_set[i], gene_set[j])

    neighbor_map = net.get_adj_list()
    for node in net.nodes:
        node["value"] = len(neighbor_map[node["id"]])

    return net
