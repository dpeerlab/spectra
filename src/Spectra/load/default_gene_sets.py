import json
import pkg_resources


def load():
    """
    load default gene set dictionary
    """
    path = pkg_resources.resource_filename("Spectra", "data/default_gene_sets.json")
    with open(path, "r") as f:
        default_gene_sets = json.loads(f.read())
    return default_gene_sets
