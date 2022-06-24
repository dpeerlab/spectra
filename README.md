# SPECTRA: Supervised Pathway DEConvolution of InTerpretable Gene ProgRAms
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


# Usage
```
from spectra import SPECTRA as spc
from spectra import K_est as kst 

L = kst.estimate_L(adata, attribute = "cell_type", highly_variable = True)
model = spc.est_spectra(adata = adata, L = L, gene_set_dictionary = gene_set_dictionary)
```
