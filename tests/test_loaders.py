import Spectra

adata = Spectra.sample_data.load()


def test_gene_set():
    gene_set_read = Spectra.default_gene_sets.load()
    gene_set_true = [
        "B_GC",
        "B_memory",
        "B_naive",
        "CD4_T",
        "CD8_T",
        "DC",
        "ILC3",
        "MDC",
        "NK",
        "Treg",
        "gdT",
        "mast",
        "pDC",
        "plasma",
        "global",
    ]
    assert set(gene_set_read) == set(gene_set_true)


def test_adata_sample():
    adata_read = Spectra.sample_data.load()

    adata_shape = (1000, 6397)
    obsm_keys = ["X_diffmap", "X_draw_graph_fa", "X_pca", "X_tsne", "X_umap"]
    obs_cols = ["cell_type_annotations"]
    var_cols = ["n_cells_by_counts", "highly_variable"]

    assert adata_read.shape == adata_shape
    assert set(adata_read.obsm.keys()) == set(obsm_keys)
    assert set(adata_read.obs.columns) == set(obs_cols)
    assert set(adata_read.var.columns) == set(var_cols)
