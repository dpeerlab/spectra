
![alt text](img/spectra_img.png?raw=true)

SPECTRA takes in a single cell gene expression matrix and a set of pathway annotations to fit the data to. 

```
gene_set_annotations = {
"global": {"global_ifn_II_response" : ["CIITA", "CXCL10", ...] , "global_MHCI": ["HLA-A". "HLA-B", "HLA-C"] },
"CD8_T": {"CD8_exhaustion" : ["CXCL13", "LAG3", "CD39", ...]}
... 

}
```

Alternatively, SPECTRA can take in a set of adjacency matrices representing networks for each cell type:

```
gene_set_annotations = {
"global": adjacency_matrix1,
"CD8_T": adjacency_matrix2,
...
}
```


# Installation 
A pypi package will be available soon. For installation, you can add spectra to pip:  
 
```
git clone https://github.com/dpeerlab/spectra
cd spectra
pip install . 

```
# Example Data Download:
Data used for the examples can be found here: [dropbox link to h5ad file](https://www.dropbox.com/s/jm49egj7lhe83fe/Bassez_10k_subset_scran.h5ad?dl=0) . This is a subsetted version of the Bassez et al. Nature Medicine (https://doi.org/10.1038/s41591-021-01323-8) preprocessed by scran and emptydrops with annotated cell types at various resolutions.


# Usage
We start by importing spectra and a helper package that estimates the number of factors to use for spectra by a version of bulk eigenvalue matching analysis. In some cases, we can run spectra with number of factors equal to number of gene sets instead of estimating the number of factors. There are multiple ways to fit the model, but the simplest is to use the `est_spectra()` function in the `spectra` module. After determining the number of factors per cell type with `estimate_L` we can run this. The function modifies the AnnData object in place as well as returning a spectra model object that contains the entire set of fitted model parameters. 
```
from spectra import spectra as spc

model = spc.est_spectra(adata = adata,  gene_set_dictionary = gene_set_dictionary, cell_type_key = "cell_type", use_highly_variable = True, lam = 0.01)
```
This latter function stores three important quantities in the AnnData. Factors are the scores that tell you how much each gene contributes to each factor, while markers is an array of genes with top scores for every factor. Cell scores are similarly the score of each factor for every cell. Finally, vocab is a boolean array that is `True` for genes that were used while fitting the model - note that this quantity is only added to the AnnData when `highly_variable` is set to `True`  
```
factors = adata.uns['SPECTRA_factors'] # factors x genes matrix that tells you how important each gene is to the resulting factors
markers = adata.uns['SPECTRA_markers'] # factors x K [where K is a user specified number] list of K top markers per factor
cell_scores = adata.obsm['SPECTRA_cell_scores'] # cells x factors matrix of cell scores
vocab = adata.var['spectra_vocab'] # boolean matrix of size # of genes that indicates the set of genes used to fit spectra 
```

## Accessing model parameters
To access finer grained information about the model fit, we can look at the attributes of the model object directly. Model parameters can be accessed with functions associated with the model object
```
model.return_eta_diag()
model.return_cell_scores()
model.return_factors() 
model.return_eta()
model.return_rho()
model.return_kappa()
model.return_gene_scalings()
```
Apart from cell scores and factors, we can also retrive a number of other parameters this way that are not by default added to the AnnData. Eta diag is the diagonal of the fitted factor-factor interaction matrix; however, its interpretation is that it measures the extent to which each factor is influenced by the prior information. In practice many of these values are zero, indicating that they are estimated without bias introduced by the annotation set. Eta is the full set of factor-factor interaction matrices, whose off diagonals measure the extent to which factors share the same genes. Rho and kappa are parameters that control the background rate of non-edges and edges respectively. These can be fixed throughout training (default) or estimated from the data by providing `rho = None` or `kappa = None` to the `est_spectra()` function  or to `model.train()`. Finally gene scalings are correction factors that normalize each gene based on its mean expression value. 

## Interpreting marker lists with respect to the input annotations
To interpret factor weight based marker lists with respect to the input annotations, we take the overlap coefficient with the input sets. The spectra model object has a function to do this `matching(markers, annotations,threshold)`
```
model.matching(adata.uns["SPECTRA_markers"], annotations, threshold = 0.4)
```


## Accessing the fitted gene-gene graph
One outcome of fitting spectra is to fit a gene-gene graph where edges represent similarities between latent variables associated with each gene (a smoothed version of transcriptional similarity) 
To access this for say, TNK cells; use the following
```
soft_graph = model.return_graph(ct = "TNK")
```
for large numbers of genes its clumsy to visualize the whole graph - to visualize a subgraph formed around a particular list of genes, use:
```
out = spc.graph_network(adata, soft_graph, gene_set)
out.show("test_graph.html")
```
this will take N closest genes to your gene set and only visualize this subgraph. The interactive graph file gets saved as an html. To visualize multiple gene sets at the same time, we have a different version of the function that assigns a random color to each gene set: 
```
#gene sets is a list of lists
out = spc.graph_network_multiple(adata,soft_graph, gene_sets)
out.show("test_graph.html")
``` 
![alt text](img/graph_img.png?raw=true)

## Save and load the model
```
#save trained model
model.save("test_model")

#initialize a model and load trained model
model = spc.SPECTRA_Model(X = X, labels = labels,  L = L, vocab = vocab, gs_dict = gene_set_dictionary)
model.load("test_model",labels = labels) 
```

## Fitting the model without AnnData
Instead of passing an AnnData object to `est_spectra` one can pass `np.ndarray` objects directly. The `**kwargs` contains arguments to the training function, `lr_schedule = [1.0,.5,.1,.01,.001,.0001],num_epochs = 10000, verbose = False`. To do this, initialize a model:
```
model = spc.SPECTRA_Model(X = X, labels = labels,  L = L, vocab = vocab, gs_dict = gene_set_dictionary, use_weights = use_weights, lam = lam, delta=delta,kappa = kappa, rho = rho, use_cell_types = use_cell_types)
model.train(X = X, labels = labels,**kwargs)
```
It is also required to run the model this way if you want to input arbitrary adjacency matrices instead of a dictionary of gene sets. The gene set dictionary is used to create an adjacency matrix when it is not `None`. 
```
model = spc.SPECTRA_Model(X = X, labels = labels,  L = L, adj_matrix = adj_matrix, weights = weights, lam = lam, delta=delta,kappa = kappa, rho = rho, use_cell_types = use_cell_types)
model.train(X = X, labels = labels,**kwargs)
```

## Estimating the number of factors
To estimate the number of factors first, run
```
from spectra import K_est as kst
L = kst.estimate_L(adata, attribute = "cell_type", highly_variable = True)
```
## Fitting via EM 
For smaller problems we can use a memory intensive EM algorithm instead
```
X = adata.X 
A = binary adjacency matrix 
model = spc.SPECTRA_EM(X = X, A= A, T = 4)
model.fit()
```

## Examine parameters of underlying model
```
model.internal_model

SPECTRA(
  (theta): ParameterDict(
      (global): Parameter containing: [torch.FloatTensor of size 4324x20]
      (B): Parameter containing: [torch.FloatTensor of size 4324x31]
      (M): Parameter containing: [torch.FloatTensor of size 4324x15]
      (Neutrophil): Parameter containing: [torch.FloatTensor of size 4324x3]
      (TNK): Parameter containing: [torch.FloatTensor of size 4324x13]
  )
  (alpha): ParameterDict(
      (B): Parameter containing: [torch.FloatTensor of size 6762x51]
      (M): Parameter containing: [torch.FloatTensor of size 29328x35]
      (Neutrophil): Parameter containing: [torch.FloatTensor of size 3492x23]
      (TNK): Parameter containing: [torch.FloatTensor of size 77495x33]
  )
  (eta): ParameterDict(
      (global): Parameter containing: [torch.FloatTensor of size 20x20]
      (B): Parameter containing: [torch.FloatTensor of size 31x31]
      (M): Parameter containing: [torch.FloatTensor of size 15x15]
      (Neutrophil): Parameter containing: [torch.FloatTensor of size 3x3]
      (TNK): Parameter containing: [torch.FloatTensor of size 13x13]
  )
  (gene_scaling): ParameterDict(
      (global): Parameter containing: [torch.FloatTensor of size 4324]
      (B): Parameter containing: [torch.FloatTensor of size 4324]
      (M): Parameter containing: [torch.FloatTensor of size 4324]
      (Neutrophil): Parameter containing: [torch.FloatTensor of size 4324]
      (TNK): Parameter containing: [torch.FloatTensor of size 4324]
  )
)

```
