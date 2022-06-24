# SPECTRA: Supervised Pathway DEConvolution of InTerpretable Gene ProgRAms

![alt text](https://github.com/dpeerlab/spectra/img/spectra_img.png?raw=true)

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
We recommend setting up a new conda environment for spectra:
 
```
git clone https://github.com/dpeerlab/spectra
cd spectra
conda env create -f environment.yml
conda activate spectra
pip install .
```

# Usage
```
from spectra import SPECTRA as spc
from spectra import K_est as kst 

L = kst.estimate_L(adata, attribute = "cell_type", highly_variable = True)
model = spc.est_spectra(adata = adata, L = L, gene_set_dictionary = gene_set_dictionary)
```

## Fitting via EM 
For smaller problems we can use a memory intensive EM algorithm instead
```
X = adata.X 
A = binary adjacency matrix 
model = SPECTRA_EM(X = X, A= A, T = 4)
model.fit()
```

