"""

"""

class attend(nn.Module):
    """
    per (i,j): 
    
    X \approx lam* (g_j + delta) <Theta_i, Eps_proj * Gamma_j>
    
    Penalize KL(A, Gamma) / lam
    
    Parameters:
    
    g
    Theta
    Eps
    Gamma
    
    -----
    
    Usage : 
    bin_matrix = # binary gene set matrix 
    
    bin_matrix = (bin_matrix + 0.001)/(bin_matrix + 0.001).sum(axis = 0)
    model = attend(X = X, K = 30, gene_set_matrix = bin_matrix, delta = 0.0, lam = 10e-5, beta = 10e-1)
    epsilon = model.assignments.softmax(dim = 1).detach().numpy()
    """
    def __init__(self, X, K, gene_set_matrix, delta = 0.001, lam = 10e-4,beta = 3.0):
        super(attend, self).__init__()
        self.p = X.shape[1]
        self.n = X.shape[0]
        self.K = K 
        self.X = torch.Tensor(X) 
        self.gene_set_matrix = torch.Tensor(gene_set_matrix)
        self.L = gene_set_matrix.shape[0]
        self.lambda_ = lam 
        self.delta = delta
        self.beta = beta
        self.theta = nn.Parameter(Normal(0.,1.).sample([self.n, self.K]))
        self.gamma = nn.Parameter(Normal(0.,1.).sample([self.p, self.L]))
        self.gene_scalings = nn.Parameter(Normal(0.,1.).sample([self.p]))
        self.assignments = nn.Parameter(Normal(0.,1.).sample([self.K,self.L]))
    
    def loss(self): 
        #compute the Poisson reconstruction loss
        gamma = self.gamma.softmax(dim = 1)
        theta = self.theta.exp()
        gene_scaling = self.gene_scalings.exp()/(1 + self.gene_scalings.exp())
        epsilon = self.assignments.softmax(dim = 1)
        reconstruction  = contract('il,lj,kj,k->ik',theta,epsilon,gamma,gene_scaling + self.delta)
        loss_1 = (-1*torch.xlogy(self.X, reconstruction) + reconstruction).sum()
        #compute loss between gamma and gene set matrix
        loss_2 = -1*torch.xlogy(self.gene_set_matrix,gamma.T).sum()
        loss_3 = -1*torch.logdet(contract('ij,kj->ik',epsilon, epsilon))
        
        #loss_2 = -1*torch.corrcoef(self.gene_set_matrix,gamma.T)
        return self.lambda_*loss_1 + loss_2 + self.beta*loss_3
    def store_parameters(self):
        self.cell_scores = 0
        self.factors = 0
        
        
def return_factors(model):
    gamma = model.gamma.softmax(dim = 1)
    gene_scaling = model.gene_scalings.exp()/(1 + model.gene_scalings.exp())
    epsilon = model.assignments.softmax(dim = 1)
    factor_unscaled  = contract('lj,kj,k->lk',epsilon,gamma,gene_scaling + model.delta).detach().numpy()
    factor_scaled = factor_unscaled/(factor_unscaled.sum(axis = 0) + 0.01)
    return factor_scaled

def return_markers(factor_matrix, id2word,n_top_vals = 100):
    idx_matrix = np.argsort(factor_matrix,axis = 1)[:,::-1][:,:n_top_vals]
    df = pd.DataFrame(np.zeros(idx_matrix.shape))
    for i in range(idx_matrix.shape[0]):
        for j in range(idx_matrix.shape[1]):
            df.iloc[i,j] = id2word[idx_matrix[i,j]]
    return df

def matching(markers, gene_names_dict, threshold = 0.4):
        """
        best match based on overlap coefficient
        """
        
        matches = []
        jaccards = []
        for i in range(markers.shape[0]):
            max_jacc = 0.0 
            best = ""
            for key in gene_names_dict.keys():
                t = gene_names_dict[key]

                jacc = spectra_util.overlap_coefficient(list(markers.iloc[i,:]),t)
                if jacc > max_jacc:
                    max_jacc = jacc
                    best = key 
            matches.append(best)
            jaccards.append(max_jacc)
        
        return matches,jaccards
        
        
def train(model, lr_schedule = [1.0,.5,.1,.01,.001,.0001],num_epochs = 10000, verbose = False): 
        opt = torch.optim.Adam(model.parameters(), lr=lr_schedule[0])
        counter = 0
        last = np.inf

        for i in tqdm(range(num_epochs)):
            #print(counter)
            opt.zero_grad()
            
            loss = model.loss()
            
            loss.backward()
            opt.step()
        
            if loss.item() >= last:
                counter += 1
                if int(counter/3) >= len(lr_schedule):
                    break
                if counter % 3 == 0:
                    opt = torch.optim.Adam(model.parameters(), lr=lr_schedule[int(counter/3)])
                    if verbose:
                        print("UPDATING LR TO " + str(lr_schedule[int(counter/3)]))
            last = loss.item() 