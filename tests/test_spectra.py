import pytest
from copy import deepcopy
import numpy as np
import Spectra

@pytest.fixture
def annotations():
    return Spectra.default_gene_sets.load()

@pytest.fixture
def adata():
    return Spectra.sample_data.load()


def test_model_default(adata, annotations):
    model = Spectra.est_spectra(
        adata=adata, 
        gene_set_dictionary=annotations, 
        num_epochs=3
    )

    # Check cell type specific info matrix (subset)
    inspect_values_subset = {"B_GC": 1, "B_memory": 5, "B_naive": 1, "gdT": 1}
    for key, value in inspect_values_subset.items():
        assert key in model.L, f"Key {key} not in model.L"
        assert model.L[key] == value, f"Value for key {key} is not correct. Should be {value}, but is {model.L[key]}"
        assert key in model.gene_scalings, f"Key {key} not in model.gene_scalings"
        assert len(model.gene_scalings[key]) == 6392, f"Gene scalings shape is not correct. Should be 6392, but is {len(model.gene_scalings[key])}"
    assert "global" in model.gene_scalings, f"Gene scalings does not contain 'global'"
    assert len(model.gene_scalings["global"]) == 6392, f"Gene scalings shape is not correct. Should be 6392, but is {len(model.gene_scalings['global'])}"

    # Check shapes
    assert model.factors.shape == (196, 6392), f"Factors shape is not correct. Should be (196, 6392), but is {model.factors.shape}"
    assert model.cell_scores.shape == (1000, 196), f"Cell scores shape is not correct. Should be (1000, 196), but is {model.cell_scores.shape}"


def test_train(adata, annotations):
    model = Spectra.est_spectra(
        adata=adata, 
        gene_set_dictionary=annotations, 
        num_epochs=1
    )

    model_save = deepcopy(model)
    
    X = np.array(adata[:, adata.var["spectra_vocab"]].X.todense())
    labels = adata.obs["cell_type_annotations"].values
    model.train(X, labels=labels, num_epochs=1)
    
    assert not np.allclose(model.factors, model_save.factors), "Model factors did not change after training"
    assert not np.allclose(model.cell_scores, model_save.cell_scores), "Model cell scores did not change after training"
    assert not np.allclose(model.gene_scalings["global"], model_save.gene_scalings["global"]), "Model global gene scalings did not change after training"
    assert not np.allclose(model.gene_scalings["B_GC"], model_save.gene_scalings["B_GC"]), "Model B_GC gene scalings did not change after training"
    assert not np.allclose(list(model.kappa.values()), list(model_save.kappa.values())), "Model kappa values did not change after training"


