import numpy as np
from opt_einsum import contract
import scipy 
from tqdm import tqdm

class BEMA: 
    """ 
    Performs bulk eigenvalue matching analysis.
    @Russell ZK
    
    [1] Algorithm 2 in : https://arxiv.org/abs/2006.00436 
    
    Parameters
    __________
    eigenvalues: np.ndarray, ascending order
        the ``(min(n,p),)``-shaped vector of eigenvalues. Must correspond to min(n,p)
    B: int 
        the number of bootstrapped replicates for estimating the best fit null distribution; default is 10
    M: int 
        the number of bootstrapped replicates for determining the K estimation threshold; default is 100 
    alpha: float 
        the 
    beta: float
    
    B: int 
    
    Output
    __________
    
    BEMA.estimate_background(B) ; 
        stores gamma parameters self.theta and self.sigma_sq
        returns None 
    
    BEMA.estimate_K(M) ; 
        stores results of Monte Carlo simulations in a ``(min(n,p), M)``-shaped np.ndarray, sorted from 
        smallest to largest eigenvalue. Can be used to check model fit 
        
        
        
    
    Examples
    _________
    
    Here we fit a ``(n,p)``-shaped dataset X with BEMA :: 
        # X a ``(n,p)``-shaped np.ndarray; get covariance matrix
        n = X.shape[0]
        p = X.shape[1]
        cov = X.T.dot(X)/n
        
        # get eigenvalues of covariance matrix
        l,_=np.linalg.eigh(cov)
        
        #initialize BEMA class
        model = BEMA(n, p, l)
        K = model.forward()
    
    
    
    """
    def __init__(self, n, p, eigenvalues, alpha = 0.2, beta = 0.1, M = 100, B = 10): 
        #assert(min(n,p) == len(eigenvalues))

        self.eigenvalues = np.array(sorted(eigenvalues))#cast to np.array and sort
        self.p = p
        self.n = n 
        self.p_tilde = min(n,p)
        self.gamma_n = p/n 
        self.alpha = alpha
        self.M = M 
        self.beta = beta
        self.B = B 
        self.eigenvalues = self.eigenvalues[-1*self.p_tilde:]
    def getQT(self, theta, B):
        """ 
        getQT function gets the quantiles
        """
        lambda_b = np.zeros((self.p_tilde,B)) #
        
        for b in range(B): 
            cov = np.random.gamma(theta,1/theta,self.p)
            Z = np.random.normal(size = (self.n,self.p))
            Z_s = np.sqrt(cov).reshape(1,-1)*Z

            # if n is larger than p, we compute the regular covariance matrix - otherwise for computational efficiency
            #       compute the n x n matrix with identical non-zero eigenvalues
            if self.n >= self.p:
                S_b = 1/self.n * contract('ij,ik->jk', Z_s, Z_s)
            else:
                S_b = 1/self.n * contract('ji,ki->jk', Z_s, Z_s)
            l_sim =np.linalg.eigvalsh(S_b)
            
            lambda_b[:,b] = l_sim 
        
        return lambda_b.mean(axis = 1)
    
    def est_sigma(self,theta,qt):
        eigvals = self.eigenvalues[int(self.alpha*self.p_tilde):int((1-self.alpha)*self.p_tilde)]
        qt = qt[int(self.alpha*self.p_tilde):int((1-self.alpha)*self.p_tilde)]
        sigma_sq = np.dot(qt, eigvals) / np.dot(qt, qt)
        return sigma_sq

    def loss(self, theta, sigma_sq, qt):
        eigvals = self.eigenvalues[int(self.alpha*self.p_tilde):int((1-self.alpha)*self.p_tilde)]
        qt = qt[int(self.alpha*self.p_tilde):int((1-self.alpha)*self.p_tilde)]
        loss_ = ((sigma_sq * qt - eigvals)**2).sum()
        return loss_ 

    def estimate_background(self, B):
        bound = [0.1,6]
        while bound[1] - bound[0] > 0.001: # todo: make these hyperparameters tunable
            print("Minimization gap... ", bound[1] - bound[0])
            points = np.linspace(bound[0], bound[1], 4)
            losses = []
            for point in points:
                qt = self.getQT(point, B)
                sigma_sq = self.est_sigma(point,qt)
                loss_ = self.loss(point,sigma_sq,qt)
                losses.append(loss_)

            ind = np.argmin(losses)
            if ind==0:
                bound=[points[ind],points[ind+1]]
            elif ind==3:
                bound=[points[ind-1],points[ind]]
            else:
                bound=[points[ind-1],points[ind+1]]

        self.theta = points[ind]
        qt = self.getQT(self.theta, B)
        self.sigma_sq = self.est_sigma(self.theta, qt)
        return

    
    def estimate_K(self, M):
        """
        Runs step 2 of the algorithm - estimating K 
        """
        #store entire results matrix
        res_matrix = np.zeros((self.p_tilde,M)) 
        for m in tqdm(range(M)):
            d = np.random.gamma(self.theta,1/self.theta, self.p) 
            Z = np.random.normal(size = (self.n,self.p))
            Z_s = np.sqrt(d*self.sigma_sq).reshape(1,-1)*Z
            
            if self.n >= self.p:
                S_b = 1/self.n * contract('ij,ik->jk', Z_s, Z_s)
            else:
                S_b = 1/self.n * contract('ji,ki->jk', Z_s, Z_s)
            l_sim =np.linalg.eigvalsh(S_b)
            res_matrix[:,m] = l_sim
            
        #update global parameters
        self.res_matrix = res_matrix
        self.threshold = np.array(sorted(res_matrix[-1,:]))[int((1-self.beta)*M)] 
        return sum(self.eigenvalues > self.threshold)

    def forward(self):
        print("Step 1; estimating best fit null distribution... ")
        self.estimate_background(self.B)
        print("Step 2; estimating K...")
        return self.estimate_K(self.M)
    
    def confidence_interval(self, m_0):
        upper = self.res_matrix[-1,int((1-m_0/2)*self.M)]
        lower = self.res_matrix[-1,int((m_0/2)*self.M)]
        return (sum(self.eigenvalues > lower),sum(self.eigenvalues > upper))


def estimate_L(adata, attribute, highly_variable = False, **kwargs):
    """ 

    Parameters
    ----------
    adata : sc.AnnData object

    attribute : must be column of adata.obs
    
    highly_variable : if True, only uses adata.var.highly_variable to generate eigenvalues
    
    **kwargs: arguments for BEMA class
    
    """ 
    
    L_dict = {}
    cell_types = np.unique(adata.obs[attribute])
    init_Ls = np.zeros(len(cell_types) + 1)
    if highly_variable:
        X = adata.X[:, adata.var.highly_variable]
    else:
        X = adata.X
    if type(X) == scipy.sparse.csr.csr_matrix:
        X = np.array(X.todense())
    
    data = X - X.mean(axis=0, keepdims = True)
    sample_cov = contract('ij,ik->jk',data,data)/data.shape[0]
    l = np.linalg.eigvalsh(sample_cov)
    
    model = BEMA(n = data.shape[0], p = data.shape[1], eigenvalues = l, **kwargs)
    K = model.forward()

    init_Ls[0] = K 
    
    ct = 1
    for cell_type in cell_types:
        data_ct = data[adata.obs[attribute] == cell_type]
        sample_cov = contract('ij,ik->jk',data_ct,data_ct)/data_ct.shape[0]
        l = np.linalg.eigvalsh(sample_cov)
        model = BEMA(n = data_ct.shape[0], p = data_ct.shape[1], eigenvalues = l, **kwargs)
        K_out = model.forward()
        init_Ls[ct] = K_out
        ct += 1 
        
    """ 
    L_dict["global"] = (sum(init_Ls[1:]) - init_Ls[0])/(len(cell_types) - 1)
    ct = 1
    for cell_type in cell_types:
        L_dict[cell_type] = init_Ls[ct] - L_dict["global"] 
        ct += 1 
    """ 
    L_dict["global"] = int(init_Ls[0] + 1)
    ct = 1
    for cell_type in cell_types:
        L_dict[cell_type] = int(init_Ls[ct] + 1)
        ct +=1
    return L_dict
    
    
