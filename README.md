# SPECTRA: Supervised Pathway DEConvolution of InTerpretable Gene ProgRAms
SPECTRA takes in a single cell gene expression matrix and a set of pathway annotations to fit the data to. 

```
gene_set_annotations = {
"global": {"global_ifn_II_response" : ["CIITA", "CXCL10", ...] , "global_TNF_response": ["". ""] },
"CD8_T": { ... }
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
