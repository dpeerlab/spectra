import pkg_resources
import scanpy as sc

def load():
    """
    Load sample data from Spectra package
    """
    path = pkg_resources.resource_filename('Spectra', 'data/sample_data.h5ad')
    adata = sc.read_h5ad(path)
    print(f"Loaded sample data: {adata.shape}")
    return adata