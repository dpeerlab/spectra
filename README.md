![alt text](img/spectra_schema.png?raw=true)


# Quickstart

SPECTRA takes in a single cell gene expression matrix, cell type annotations, and gene sets for cellular processes to fit the data to. 

If you use Spectra please cite our preprint on [bioRxiv](https://doi.org/10.1101/2022.12.20.521311).

We start by importing spectra. The easiest way to run spectra is to use the `est_spectra` function, as shown below. The default behavior is to set the number of factors equal to the number of gene sets plus one. However, this can be modified by passing an integer e.g. `L = 20` as an argument to the function or a dictionary that maps cell type to an integer per cell type. We provide a method for estimating the number of factors directly from the data by bulk eigenvalue matching analysis, which is detailed further below. 

```
# This script requires 12GB of memory or more

import Spectra

annotations = Spectra.default_gene_sets.load()
adata = Spectra.sample_data.load()

#run the model with default values
model = Spectra.est_spectra(
    adata=adata, 
    gene_set_dictionary=annotations, 
    use_highly_variable=True,
    cell_type_key="cell_type_annotations", 
    use_weights=True,
    lam=0.1, #varies depending on data and gene sets, try between 0.5 and 0.001
    delta=0.001, 
    kappa=None,
    rho=0.001, 
    use_cell_types=True,
    n_top_vals=50,
    label_factors=True, 
    overlap_threshold=0.2,
    clean_gs = True, 
    min_gs_num = 3,
    num_epochs=10000
)

```

This function stores four important quantities in the AnnData, in addition to returning a fitted model object. Factors are the scores that tell you how much each gene contributes to each factor, while markers is an array of genes with top scores for every factor. Cell scores are similarly the score of each factor for every cell. Finally, vocab is a boolean array that is `True` for genes that were used while fitting the model - note that this quantity is only added to the AnnData when `highly_variable` is set to `True`.

```
factors = adata.uns['SPECTRA_factors'] # factors x genes matrix that tells you how important each gene is to the resulting factors
markers = adata.uns['SPECTRA_markers'] # factors x n_top_vals list of n_top_vals top markers per factor
cell_scores = adata.obsm['SPECTRA_cell_scores'] # cells x factors matrix of cell scores
vocab = adata.var['spectra_vocab'] # boolean matrix of size # of genes that indicates the set of genes used to fit spectra 
```

# Installation 

A pypi package will be available soon. For installation, you can add spectra to pip:  

```
pip install scSpectra
```

*Requires Python 3.7 or later.*

# scRNAseq Knowledge Base

Check out our scRNAseq knowledge base [Cytopus :octopus:]([https://github.com/wallet-maker/cytopus](https://github.com/dpeerlab/cytopus)) to retrieve *Spectra* input gene sets adapted to the cell type composition in your data.

# Spectra GUI

A Graphical User Interface (GUI) to sit over top SPECTRA, a factor analysis tool developed in Dana Pe'er's Lab at Memorial Sloan Kettering Cancer Research Center (MSKCC) can be found [here](https://github.com/HMC-Clinic-MSKCC-22-23/SpectraGUI).

# Interactive Tutorial

We provide a full tutorial how to run the basic Spectra model here: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/dpeerlab/spectra/blob/main/notebooks/Spectra_Colaboratory_tutorial.ipynb) (12 GB RAM required)

You can use log1p transformed median library size normalized data. For leukocyte data, we recommend using [scran normalization](https://doi.org/10.1186/s13059-016-0947-7). We provide a [tutorial](https://github.com/dpeerlab/spectra/blob/main/notebooks/scran_preprocessing.ipynb).

# Advanced

## Accessing model parameters
Run the model as indicated above. To access finer grained information about the model fit, we can look at the attributes of the model object directly. Model parameters can be accessed with functions associated with the model object

```
model.return_eta_diag()
model.return_cell_scores()
model.return_factors() 
model.return_eta()
model.return_rho()
model.return_kappa()
model.return_gene_scalings()
```

Apart from cell scores and factors, we can also retrive a number of other parameters this way that are not by default added to the AnnData. Eta diag is the diagonal of the fitted factor-factor interaction matrix; however, its interpretation is that it measures the extent to which each factor is influenced by the prior information. In practice many of these values are zero, indicating that they are estimated without bias introduced by the annotation set. Eta is the full set of factor-factor interaction matrices, whose off diagonals measure the extent to which factors share the same genes. Rho and kappa are parameters that control the background rate of non-edges and edges respectively. These can be fixed throughout training or estimated from the data by providing `rho = None` or `kappa = None` to the `est_spectra()` function  or to `model.train()`. Finally gene scalings are correction factors that normalize each gene based on its mean expression value. 

## Estimating the number of factors
For most datasets you want to select the number of factors based on the number of gene sets and prior knowledge as well as the granularity of the expected gene programs. However, we also provide a method to estimate the number of factors. To estimate the number of factors first, run:
```
from Spectra import K_est as kst
L = kst.estimate_L(adata, attribute = "cell_type", highly_variable = True)
```

## Fitting via EM 
For smaller problems we can use a memory intensive EM algorithm instead
```
X = adata.X 
A = binary adjacency matrix g
model = Spectra.SPECTRA_EM(X = X, A= A, T = 4)
model.fit()
```

## Parameters 

`adata` : AnnData object containing cell_type_key with log count data stored in .X

`gene_set_dictionary` : dict or OrderedDict() maps cell types to gene set names to gene sets ; if use_cell_types == False then maps gene set names to gene sets ; must contain "global" key in addition to every unique cell type under .obs.<cell_type_key>

`L` : dict, OrderedDict(), int , NoneType number of factors per cell type ; if use_cell_types == False then int. Else dictionary. If None then match factors to number of gene sets (recommended)

`use_highly_variable` : bool if True, then uses highly_variable_genes

`cell_type_key` : str cell type key, must be under adata.obs.<cell_type_key> . If use_cell_types == False, this is ignored

`use_weights` : bool if True, edge weights are estimated based on graph structure and used throughout training

`lam` : float lambda parameter of the model. weighs relative contribution of graph and expression loss functions

`delta` : float delta parameter of the model. lower bounds possible gene scaling factors so that maximum ratio of gene scalings cannot be too large

`kappa` : float or None if None, estimate background rate of 1s in the graph from data

`rho` : float or None if None, estimate background rate of 0s in the graph from data

`use_cell_types` : bool if True then cell type label is used to fit cell type specific factors. If false then cell types are ignored

`n_top_vals` : int number of top markers to return in markers dataframe

`determinant_penalty` : float determinant penalty of the attention mechanism. If set higher than 0 then sparse solutions of the attention weights and diverse attention weights are encouraged. However, tuning is crucial as setting too high reduces the selection accuracy because convergence to a hard selection occurs early during training [todo: annealing strategy]

`filter_sets` : bool whether to filter the gene sets based on coherence

`clean_gs` : bool if True cleans up the gene set dictionary to:
                1. checks that annotations dictionary cell type keys and adata cell types are identical.
                2. to contain only genes contained in the adata
                3. to contain only gene sets greater length min_gs_num

`min_gs_num` : int only use if clean_gs True, minimum number of genes per gene set expressed in adata, other gene sets will be filtered out

`label_factors` : bool whether to label the factors by their cell type specificity and their Szymkiewicz–Simpson overlap coefficient with the input marker genes

`overlap_threshold`: float minimum overlap coefficient to assign an input gene set label to a factor

`num_epochs` : int number of epochs to fit the model. We recommend 10,000 epochs which works for most datasets although many models converge earlier

``**kwargs`` : (num_epochs = 10000, lr_schedule = [...], verbose = False) arguments to .train(), maximum number of training epochs, learning rate schedule and whether to print changes in learning rate

**Returns**: SPECTRA_Model object [after training]

**In place**: adds 1. factors, 2. cell scores, 3. vocabulary, and 4. markers as attributes in .obsm, .var, .uns

**default parameters:** `est_spectra(adata, gene_set_dictionary, L = None,use_highly_variable = True, cell_type_key = None, use_weights = True, lam = 0.01, delta=0.001,kappa = None, rho = 0.001, use_cell_types = True, n_top_vals = 50, 
filter_sets = True, clean_gs = True, min_gs_num = 3, label_factors=True, overlap_threshold= 0.2, **kwargs)`

## labeling factors

We also provide an approach to label the factors by their Szymkiewicz–Simpson overlap coefficient with the input gene sets. Each factors receives the label of the input gene set with the highest overlap coefficient, given that it the overlap coefficient is greater than the threshold defined in 'overlap_threshold'. Ties in the overlap coefficient by gene set size, selecting the label of the bigger gene set (because smaller gene sets might get bigger overlap coefficients by chance).

We provide a pandas.DataFrame indicating the overlap coefficients for each input gene set with each factor's marker genes. The index of this dataframe contains the *index* of each factor, *assigned label* as well as the *cell type specificity* for each factor in the format:

`['index' + '-X-' + 'cell type specificity' + '-X-' + 'assigned label', ...]`

We use `'-X-'` as a unique seperator to make string splitting and retrieval of the different components of the index easier.

```
adata.uns['SPECTRA_overlap']
```
