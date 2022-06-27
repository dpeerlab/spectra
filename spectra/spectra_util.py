import numpy as np 
from collections import OrderedDict 
"""
methods 
_______

amatrix(gene_set_list, gene2id) 

amatrix_weighted(gene_set_list, gene2id)

unravel_dict(dict_)

process_gene_sets()

process_gene_sets_no_celltypes()

overlap_coefficient()


"""

def amatrix(gene_set_list,gene2id):
    """ 
    creates adjacency matrix from gene set list
    """ 
    n = len(gene2id)
    adj_matrix = np.zeros((n,n))
    for gene_set in gene_set_list:
        for i in range(len(gene_set)):
            for j in range(len(gene_set)):
                g1 = gene_set[i]
                g2 = gene_set[j]
                if (g1 in gene2id)&(g2 in gene2id):
                    adj_matrix[gene2id[g1],gene2id[g2]] = 1
    return adj_matrix

def amatrix_weighted(gene_set_list, gene2id):
    """
    Creates weighted adjacency matrix from gene sets
    uses 1/{n choose 2} as edge weights - edge weights accumulate additively  
    """
    n = len(gene2id)
    adj_matrix = np.zeros((n,n))
    ws = []
    for gene_set in gene_set_list:
        if len(gene_set) > 1:
            w = 1.0/(len(gene_set)*(len(gene_set)-1)/2.0)
        else:
            w = 1.0
        ws.append(w)
        for i in range(len(gene_set)):
            for j in range(len(gene_set)):
                g1 = gene_set[i]
                g2 = gene_set[j]
                if (g1 in gene2id)&(g2 in gene2id):
                    adj_matrix[gene2id[g1],gene2id[g2]] += w
    med = np.median(np.array(ws))
    return adj_matrix/float(med)

def unravel_dict(dict_):
    lst = []
    for key in dict_.keys():
        lst.append(dict_[key])
    return lst 
def process_gene_sets(gs_dict, gene2id, weighted = True):
    """ 
    { "global": {"<gene set name>": [<gene 1>, <gene 2>, ...]}
    }
    """ 
    adict = OrderedDict()
    adict["global"] = amatrix(unravel_dict(gs_dict["global"]), gene2id)
    weights = None 

    if weighted:
        weights = OrderedDict()
        weights["global"] = amatrix_weighted(unravel_dict(gs_dict["global"]), gene2id)

    for key in gs_dict.keys():
        if len(gs_dict[key]) > 0:
            adict[key] = amatrix(unravel_dict(gs_dict[key]),gene2id) 
            if weighted:
                weights[key] = amatrix_weighted(unravel_dict(gs_dict[key]), gene2id)
        else: 
            adict[key] = []
            if weighted:
                weights[key] = []

    return adict, weights



def process_gene_sets_no_celltypes(gs_dict, gene2id, weighted = True):
    """ 
    input: {"<gene set name>": [<gene 1>, <gene 2>, ...]}
    }
    gene2id {gene name: index in vocab} 

    weighted: whether to return NoneType or weighted adjacency matrix 
    """
    adict = amatrix(unravel_dict(gs_dict), gene2id)
    weights = None 
    if weighted:
        weights = amatrix_weighted(unravel_dict(gs_dict) , gene2id)
    return adict, weights

def overlap_coefficient(list1, list2): 
    """ 
    Computes overlap coefficient between two lists
    """ 
    intersection = len(list(set(list1).intersection(set(list2))))
    union = min(len(list1),len(list2))# + len(list2)) - intersection
    return float(intersection) / union