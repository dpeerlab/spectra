import torch 
import numpy as np 
from collections import OrderedDict

def mimno_coherence_single(w1,w2,W):
    eps = 0.01
    dw1 = W[:, w1] > 0
    dw2 = W[:, w2] > 0
    N = W.shape[0]

    dw1w2 = (dw1 & dw2).float().sum() 
    dw1 = dw1.float().sum() 
    dw2 = dw2.float().sum() 

    return ((dw1w2 + 1)/(dw2)).log()

def mimno_coherence_2011(words, W): 
    score = 0
    V= len(words)

    for j1 in range(1, V):
        for j2 in range(j1):
            score += mimno_coherence_single(words[j1], words[j2], W)
    denom = V*(V-1)/2
    return(score/denom)

def compute_init_scores_noct(gs_dict, word2id, W):
    init_scores = OrderedDict()
    keys = list(gs_dict.keys()) 
    for key in keys: 
        gs = gs_dict[key] 
        idxs = [] 
        for word in gs:
            if word in word2id:
                idxs.append(word2id[word])
        #idxs = [word2id[word] for word in gs] 
        coh = mimno_coherence_2011(idxs,W)
        init_scores[key] = coh.item() 
    return init_scores
        
def compute_init_scores(gs_dict, word2id, W):
    keys = list(gs_dict.keys())
    init_scores = OrderedDict()
    for key in keys:
        if len(gs_dict[key]) > 0:
            inner_keys = list(gs_dict[key].keys())
            init_scores[key] = OrderedDict() 
            for inner_key in inner_keys:
                gs = gs_dict[key][inner_key]
                idxs = [] 
                for word in gs: 
                    if word in word2id:
                        idxs.append(word2id[word])
                #idxs = [word2id[word] for word in gs]
                coh = mimno_coherence_2011(idxs, W)
                init_scores[key][inner_key] = coh.item() 
        else: 
            init_scores[key] = {} 
            
        
    return init_scores 

