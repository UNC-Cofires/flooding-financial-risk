import numpy as np
import scipy.stats as stats
import scipy.spatial as spatial
import scipy.sparse as sparse
import scipy.optimize as so
import pandas as pd
import geopandas as gpd
import time

### *** COVARIANCE FUNCTION BUILDING BLOCKS *** ###

class CovFun:
    """
    Parent class used to define shared methods of difference covariance models
    """
    
    def __init__(self):
        self.determine_param_status()
        
    def determine_param_status(self):
        """
        Determine whether parameters are fixed or need to be fitted
        """
        self.fixed_params = {}
        self.free_params = []
        self.all_fixed = True
        
        for param in self.params:
            value = getattr(self,param)
            if value is not None:
                self.fixed_params[param] = value
            else:
                self.free_params.append(param)
                self.all_fixed = False
                
        return None
    
    def get_bounds(self):
        """
        Get bounds on free parameters 
        """
        bounds = []
        
        for param in self.free_params:
            bounds.append(getattr(self,param + '_bounds'))        
        return(bounds)
    
    def update_params(self,**kwargs):
        """
        Update values of model parameters
        """
        for key,value in kwargs.items():
            setattr(self,key,value)
            
        self.determine_param_status()
        
        return None
    
    def verify_all_fixed(self):
        if not self.all_fixed:
            free_params = ', '.join(self.free_params)
            raise ValueError(f'Please specify the following model parameters: {free_params}')
            
        return None
        
    def vgm(self,h):
        """
        Return value of the variogram function
        param: h: value of spatial or temporal distance between points
        """
        v = self.cov(0) - self.cov(h)
        return v
    
class SphCovFun(CovFun):
    """
    Spherical covariance function class (child class of CovFun)
    """
    def __init__(self,a=None,a_bounds=(0,10)):
        """
        param: a: scale parameter determining speed at which spatial / temporal dependence decays
        param: a_bounds: bounds on "a" if fitting as a free parameter 
        """
        self.a = a
        self.a_bounds = a_bounds
        self.params = ['a']
        self.name = 'SphCovFun'
        super().__init__()
    
    def cov(self,h):
        """
        Return value of covariance function
        param: h: value of spatial or temporal distance between points
        """
        self.verify_all_fixed()
        c = (1-np.heaviside(h - self.a,0))*(1 - 1.5*h/self.a + 0.5*h**3/self.a**3)
        return c
    
class ExpCovFun(CovFun):
    """
    Exponential covariance function class (child class of CovFun)
    """
    def __init__(self,a=None,a_bounds=(0,10)):
        """
        param: a: scale parameter determining speed at which spatial / temporal dependence decays
        param: a_bounds: bounds on "a" if fitting as a free parameter 
        """
        self.a = a
        self.a_bounds = a_bounds
        self.params = ['a']
        self.name = 'ExpCovFun'
        super().__init__()
    
    def cov(self,h):
        """
        Return value of covariance function
        param: h: value of spatial or temporal distance between points
        """
        self.verify_all_fixed()
        c = np.exp(-h/self.a)
        return c
    
class GauCovFun(CovFun):
    """
    Gaussian covariance function class (child class of CovFun)
    """
    def __init__(self,a=None,a_bounds=(0,10)):
        """
        param: a: scale parameter determining speed at which spatial / temporal dependence decays
        param: a_bounds: bounds on "a" if fitting as a free parameter 
        """
        self.a = a
        self.a_bounds = a_bounds
        self.params = ['a']
        self.name = 'GauCovFun'
        super().__init__()
    
    def cov(self,h):
        """
        Return value of covariance function
        param: h: value of spatial or temporal distance between points
        """
        self.verify_all_fixed()
        c = np.exp(-h**2/self.a**2)
        return c
    
class CauCovFun(CovFun):
    """
    Cauchy covariance function class (child class of CovFun)
    """
    def __init__(self,a=None,a_bounds=(0,10),beta=None,beta_bounds=(0,10),alpha=2,alpha_bounds=(0,2)):
        """
        param: a: scale parameter determining speed at which spatial / temporal dependence decays
        param: a_bounds: bounds on a if fitting as a free parameter
        param: beta: parameter affecting dependence at large distances
        param: beta_bounds: bounds on beta if fitting as a free parameter
        param: alpha: shape parameter (usually fixed at value of 2)
        param: alpha_bounds: bounds on alpha if fitting as free parameter. Must always be between [0,2]. 
        """
        self.a = a
        self.a_bounds = a_bounds
        self.beta = beta
        self.beta_bounds = beta_bounds
        self.alpha = alpha
        self.alpha_bounds = alpha_bounds
        self.params = ['a','beta','alpha']
        self.name = 'CauCovFun'
        super().__init__()
    
    def cov(self,h):
        """
        Return value of covariance function
        param: h: value of spatial or temporal distance between points
        """
        self.verify_all_fixed()
        c = (1 + h**self.alpha/self.a**self.alpha)**(-self.beta/self.alpha)
        return c
    
### *** SPACE-TIME COVARIANCE MODELS *** ###
    
class STCovFun:
    """
    Parent class used to define shared methods of difference space-time covariance models
    """
    
    def __init__(self,Cs,Ct):
        """
        param: Cs: spatial component of covariance function
        param: Ct: temporal component of covariance function 
        """
        self.Cs = Cs
        self.Ct = Ct
        self.determine_param_status()
        
    def determine_param_status(self):
        """
        Determine whether parameters are fixed or need to be fitted
        """
        self.spacetime_fixed_params = {}
        self.spacetime_free_params = []
        self.spacetime_all_fixed = True
        
        for param in self.params:
            value = getattr(self,param)
            if value is not None:
                self.spacetime_fixed_params[param] = value
            else:
                self.spacetime_free_params.append(param)
                self.spacetime_all_fixed = False
                
        self.spatial_all_fixed = self.Cs.all_fixed
        self.temporal_all_fixed = self.Ct.all_fixed
        
        self.all_fixed = self.spacetime_all_fixed and self.spatial_all_fixed and self.temporal_all_fixed
                
        return None
    
    def get_bounds(self):
        """
        Get bounds and initial guesses for free parameters 
        """
        bounds = []        
        
        for param in self.spacetime_free_params:
            bounds.append(getattr(self,param + '_bounds'))
        
        return(bounds)
    
    def update_spacetime_params(self,**kwargs):
        """
        Update coefficients of Cs and Ct
        """
        for key,value in kwargs.items():
            setattr(self,key,value)
            
        self.determine_param_status()
            
        return None
    
    def update_spatial_params(self,**kwargs):
        """
        Update parameters of Cs
        """
        self.Cs.update_params(**kwargs)
        self.determine_param_status()
        return None
    
    def update_temporal_params(self,**kwargs):
        """
        Update parameters of Cs
        """
        self.Ct.update_params(**kwargs)
        self.determine_param_status()
        return None
    
    def verify_all_fixed(self):
        
        if not self.all_fixed:
            spacetime_free_params = ', '.join(self.spacetime_free_params)
            spatial_free_params = ', '.join(self.Cs.free_params)
            temporal_free_params = ', '.join(self.Ct.free_params)
            message = 'Please specify the following model parameters:\n'
            message += f'Space-time covariance model ({self.name}): {spacetime_free_params}\n'
            message += f'Spatial component ({self.Cs.name}): {spatial_free_params}\n'
            message += f'Temporal component ({self.Ct.name}): {temporal_free_params}'
            raise ValueError(message)
            
        return None
    
    def vgm(self,hs,ht):
        """
        Return value of the variogram function
        param: hs: value of spatial distance between points
        param: ht: value of temporal distance between points
        """
        v = self.cov(0,0) - self.cov(hs,ht)
        return v
    
class ProductSTCovFun(STCovFun):
    """
    Product (separable) space-time covariance model (child class of STCovFun). 
    """
    
    def __init__(self,Cs,Ct,k=None,nugget=None,k_bounds=(0,10),nugget_bounds=(0,10)):
        """
        param: Cs: spatial component of covariance function
        param: Ct: temporal component of covariance function
        param: k: value of sill
        param: nugget: value of nugget effect
        param: k_bounds: bounds on k1 if fitting as a free parameter
        param: nugget_bounds: bounds on nugget effect if fitting as a free parameter
        """
        
        self.k = k
        self.nugget = nugget
        self.k_bounds = k_bounds
        self.nugget_bounds = nugget_bounds
        self.params = ['k','nugget']
        self.name = 'ProductSTCovFun'
        super().__init__(Cs,Ct)
    
    def cov(self,hs,ht):
        """
        Return value of covariance function
        param: hs: value of spatial distance between points
        param: ht: value of temporal distance between points
        """
        self.verify_all_fixed()
        nugget_effect = self.nugget*(1-np.abs(np.sign(hs)))*(1-np.abs(np.sign(ht)))
        c = nugget_effect + self.k*self.Cs.cov(hs)*self.Ct.cov(ht)
        return c
    
class SumSTCovFun(STCovFun):
    """
    Sum (linear) space-time covariance model (child class of STCovFun).
    """
    
    def __init__(self,Cs,Ct,
                 k1=None,k2=None,nugget=None,
                 k1_bounds=(0,10),k2_bounds=(0,10),nugget_bounds=(0,10)):
        """
        param: Cs: spatial component of covariance function
        param: Ct: temporal component of covariance function
        param: k1: coefficient of spatial term
        param: k2: coefficient of temporal term
        param: nugget: value of nugget effect
        param: k1_bounds: bounds on k1 if fitting as a free parameter
        param: k2_bounds: bounds on k2 if fitting as a free parameter
        param: nugget_bounds: bounds on nugget effect if fitting as a free parameter
        """
        
        self.k1 = k1
        self.k2 = k2
        self.nugget = nugget
        self.k1_bounds = k1_bounds
        self.k2_bounds = k2_bounds
        self.nugget_bounds = nugget_bounds
        self.params = ['k1','k2','nugget']
        self.name = 'SumSTCovFun'
        super().__init__(Cs,Ct)
    
    def cov(self,hs,ht):
        """
        Return value of covariance function
        param: hs: value of spatial distance between points
        param: ht: value of temporal distance between points
        """
        self.verify_all_fixed()
        nugget_effect = self.nugget*(1-np.abs(np.sign(hs)))*(1-np.abs(np.sign(ht)))
        c = nugget_effect + self.k1*self.Cs.cov(hs) + self.k2*self.Ct.cov(ht)
        return c
    
class ProductSumSTCovFun(STCovFun):
    """
    Product-sum space-time covariance model (child class of STCovFun). 
    Described in detail by De Iaco, Myers, and Posa (doi:10.1016/S0167-7152(00)00200-5).
    """
    
    def __init__(self,Cs,Ct,
                 k1=None,k2=None,k3=None,nugget=None,
                 k1_bounds=(0,10),k2_bounds=(0,10),k3_bounds=(0,10),nugget_bounds=(0,10)):
        """
        param: Cs: spatial component of covariance function
        param: Ct: temporal component of covariance function
        param: k1: coefficient of product term
        param: k2: coefficient of spatial term
        param: k3: coefficient of temporal term
        param: nugget: value of nugget effect
        param: k1_bounds: bounds on k1 if fitting as a free parameter
        param: k2_bounds: bounds on k2 if fitting as a free parameter
        param: k3_bounds: bounds on k3 if fitting as a free parameter
        param: nugget_bounds: bounds on nugget effect if fitting as a free parameter
        """
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.nugget = nugget
        self.k1_bounds = k1_bounds
        self.k2_bounds = k2_bounds
        self.k3_bounds = k3_bounds
        self.nugget_bounds = nugget_bounds
        self.params = ['k1','k2','k3','nugget']
        self.name = 'ProductSumSTCovFun'
        super().__init__(Cs,Ct)
    
    def cov(self,hs,ht):
        """
        Return value of covariance function
        param: hs: value of spatial distance between points
        param: ht: value of temporal distance between points
        """
        self.verify_all_fixed()
        nugget_effect = self.nugget*(1-np.abs(np.sign(hs)))*(1-np.abs(np.sign(ht)))
        c = nugget_effect + self.k1*self.Cs.cov(hs)*self.Ct.cov(ht) + self.k2*self.Cs.cov(hs) + self.k3*self.Ct.cov(ht)
        return c

### *** VARIOGRAM PARAMETER ESTIMATION FUNCTIONS *** ###
    
def fit_covariance_model(Cst,hs,ht,gamma,w=None,options={}):
    """
    Fit free parameters to empirical variogram data using weighted sum of squares.
    
    Based on section 2.6.2 of "Geostatistics: Modeling Spatial Uncertainty" by J.P Chiles. 

    param: Cst: space-time covariance model that includes free parameters (e.g., instance of ProductSumSTCovFun class)
    param: hs: numpy array of spatial distances (vector of length n)
    param: ht: numpy array of temporal distances (vector of length n)
    param: gamma: numpy array of empirical variogram values (vector of length n)
    param: w: numpy array of weights (vector of length n)
    param: options: optional dict of kwargs to pass to optimizer (see scipy.optimize.differential_evolution documentation)
    """

    # Empirical variogram function returns 2-D arrays, 
    # so flatten everything before beginning in case the user forgets
    hs = hs.flatten()
    ht = ht.flatten()
    gamma = gamma.flatten()
    
    n = len(hs)
    
    if w is None: 
        w = np.ones(n)
    else:
        w = w.flatten()
        
    sum_w = np.sum(w)
        
    spacetime_free_params = Cst.spacetime_free_params
    spatial_free_params = Cst.Cs.free_params
    temporal_free_params = Cst.Ct.free_params
    
    nstp = len(spacetime_free_params)
    nsp = len(spatial_free_params)
    ntp = len(temporal_free_params)
    
    if nstp > 0:
        stp_string = ', '.join(spacetime_free_params)
    else:
        stp_string = '--'
    if nsp > 0:
        sp_string = ', '.join(spatial_free_params)
    else:
        sp_string = '--'
    if ntp > 0:
        tp_string = ', '.join(temporal_free_params)
    else:
        tp_string = '--'
    
    message = 'Estimating the following variogram parameters via least squares:\n'
    message += f'    Space-time covariance model ({Cst.name}): {stp_string}\n'
    message += f'    Spatial component ({Cst.Cs.name}): {sp_string}\n'
    message += f'    Temporal component ({Cst.Ct.name}): {tp_string}'
    
    print(message,flush=True)
    
    param_idx_dict = {}
    param_idx_dict['spacetime_idx'] = {value:i for i,value in enumerate(spacetime_free_params)}
    param_idx_dict['spatial_idx'] = {value:i+nstp for i,value in enumerate(spatial_free_params)}
    param_idx_dict['temporal_idx'] = {value:i+nstp+nsp for i,value in enumerate(temporal_free_params)}
    
    # Set up bounds on parameters
    spacetime_bounds = Cst.get_bounds()
    spatial_bounds = Cst.Cs.get_bounds()
    temporal_bounds = Cst.Ct.get_bounds()
    
    bounds = spacetime_bounds + spatial_bounds + temporal_bounds
    
    # Define objective function
    def objective_function(theta):
        
        # Update model parameters with latest value tried by optimizer
        spacetime_kwargs = {key:theta[idx] for key,idx in param_idx_dict['spacetime_idx'].items()}
        spatial_kwargs = {key:theta[idx] for key,idx in param_idx_dict['spatial_idx'].items()}
        temporal_kwargs = {key:theta[idx] for key,idx in param_idx_dict['temporal_idx'].items()}
        Cst.update_spacetime_params(**spacetime_kwargs)
        Cst.update_spatial_params(**spatial_kwargs)
        Cst.update_temporal_params(**temporal_kwargs)
        
        # Calculate model-predicted variogram
        gamma_hat = Cst.vgm(hs,ht)
        
        # Get weighted squared error vs empirical variogram
        weighted_squared_error = np.sum(w*(gamma - gamma_hat)**2)/sum_w
        
        return(weighted_squared_error)
    
    res = so.differential_evolution(objective_function,bounds,**options)
    
    if res.success:
        theta_hat = res.x
        obj_val = res.fun
        
        spacetime_kwargs = {key:theta_hat[idx] for key,idx in param_idx_dict['spacetime_idx'].items()}
        spatial_kwargs = {key:theta_hat[idx] for key,idx in param_idx_dict['spatial_idx'].items()}
        temporal_kwargs = {key:theta_hat[idx] for key,idx in param_idx_dict['temporal_idx'].items()}
        Cst.update_spacetime_params(**spacetime_kwargs)
        Cst.update_spatial_params(**spatial_kwargs)
        Cst.update_temporal_params(**temporal_kwargs)
        
        if nstp > 0:
            stp_result = ', '.join([f'{key}={np.round(value,6)}' for key,value in spacetime_kwargs.items()])
        else:
            stp_result = '--'
        if nsp > 0:
            sp_result = ', '.join([f'{key}={np.round(value,6)}' for key,value in spatial_kwargs.items()])
        else:
            sp_result = '--'
        if ntp > 0:
            tp_result = ', '.join([f'{key}={np.round(value,6)}' for key,value in temporal_kwargs.items()])
        else:
            tp_result = '--'
        
        message = '\nOptimization converged: Returning the fit model:\n'
        message += f'    Space-time covariance model ({Cst.name}): {stp_result}\n'
        message += f'    Spatial component ({Cst.Cs.name}): {sp_result}\n'
        message += f'    Temporal component ({Cst.Ct.name}): {tp_result}\n\n'
        message += f'Objective function value: {obj_val}\n'
        
        if obj_val > 0:
            message += f'Log10(objective function value): {np.log10(obj_val)}\n'
        
        return_object = Cst
        status = 'converged'
        
    else:
        message = '\nOptimization failed: Returning the associated scipy OptimizeResult object.'
        
        return_object = res
        status = 'failed'
        
    print(message,flush=True)
    
    return(return_object,status)

# Helper function for working with sparse distance matrices
def apply_covariance_sparse(Cst,spatial_dmat,temporal_dmat,adjmat):
    """
    Calculate the values of the covariance model given sparse distance matrices. 
    
    param: Cst: fitted space-time covariance function (instance of STCovFun)
    param: spatial_dmat: sparse spatial distance matrix (n x m)
    param: temporal_dmat: sparse temporal distance matrix (n x m)
    param: adjmat: sparse binary adjacency matrix. Values of 1 denote pairs within spatial / temporal cutoff range. 
    returns: cmat: sparse matrix of covariance function values 
    """
    rows,cols = adjmat.nonzero()
    hs = spatial_dmat[rows,cols]
    ht = temporal_dmat[rows,cols].flatten()
    
    values = Cst.cov(hs,ht)
    cmat = sparse.csr_array((values,(rows,cols)),shape=adjmat.shape)
    return(cmat)

# Helper function to format elapsed time in seconds
def format_elapsed_time(seconds):
    seconds = int(np.round(seconds))
    hours = seconds // 3600
    seconds = seconds - hours*3600
    minutes = seconds // 60
    seconds = seconds - minutes*60
    return(f'{hours}h:{minutes:02d}m:{seconds:02d}s')


### *** KRIGING CLASSES *** ###
class Kriging:
    """
    Parent class used to define shared methods of different kriging approaches
    """
    def __init__(self,hard_points,krig_points,response_variable,spatial_column='geometry',temporal_column='time_val'):
        """
        param: hard_points: geodataframe of "hard" observations (i.e., realizations of space-time random field)
        param: krig_points: geodataframe of points to be interpolated via kriging
        param: response_variable: column name corresponding to quantity of interest (z)
        param: spatial_column: column name corresponding to pandas geoseries of point coordinates
        param: temporal_column: column name corresponding to time values. Must be numeric (e.g., # days since 2000).  
        """
        
        self.n_h = len(hard_points)
        self.n_k = len(krig_points)
        
        self.xy_h = np.zeros((self.n_h,2))
        self.xy_h[:,0] = hard_points[spatial_column].x
        self.xy_h[:,1] = hard_points[spatial_column].y
        
        self.xy_k = np.zeros((self.n_k,2))
        self.xy_k[:,0] = krig_points[spatial_column].x
        self.xy_k[:,1] = krig_points[spatial_column].y
        
        self.t_h = hard_points[temporal_column].to_numpy()
        self.t_k = krig_points[temporal_column].to_numpy()
        
        self.z_h = hard_points[response_variable].to_numpy()
                
        self.Cst = None
        
    def build_distance_matrices(self,spatial_cutoff,temporal_cutoff):
        """
        Create KDTrees for fast spatial indexing and construct spatial / temporal distance matrices. These
        include a spatial / temporal cutoff to reduce the computational burden of kriging a large number of points. 
        
        Please think carefully about the process you are modeling when selecting cutoff values! 
        
        For processes with persistent spatial autocorrelation (e.g., home prices), you will want to select 
        a long or infinite temporal cutoff. For processes with transient spatial autocorrelation (e.g., precipitation) 
        you can probably get away with a shorter temporal cutoff. 
        
        It is worth doing sensitivity analysis on these cutoffs to determine where they should be set.  
        
        param: spatial_cutoff: spatial distance beyond which correlation is assumed to be zero
        param: temporal_cutoff: temporal distance beyond which correlation is assumed to be zero
        """
        
        t1 = time.time()
        
        print('\nBuilding spatial and temporal distance matrices:',flush=True)
        
        self.spatial_cutoff = spatial_cutoff
        self.temporal_cutoff = temporal_cutoff
        
        # Build k-d tree for fast spatial lookups
        xy = np.concatenate((self.xy_k,self.xy_h))
        kd_tree = spatial.cKDTree(xy)
        
        t2 = time.time()
        elapsed_time = format_elapsed_time(t2-t1)
        print(f'    Finished building k-d tree: {elapsed_time} elapsed',flush=True)
        
        # Find spatial pairs that are within cutoff distance
        pairs = kd_tree.query_pairs(r=self.spatial_cutoff,output_type='ndarray')

        hh_mask = (pairs[:,0] >= self.n_k)&(pairs[:,1] >= self.n_k)
        kh_mask = (pairs[:,0] < self.n_k)&(pairs[:,1] >= self.n_k)

        self.pairs_hh = pairs[hh_mask]
        self.pairs_kh = pairs[kh_mask]
        self.pairs_hh = self.pairs_hh - self.n_k
        self.pairs_kh[:,1] = self.pairs_kh[:,1] - self.n_k
        
        t2 = time.time()
        elapsed_time = format_elapsed_time(t2-t1)
        print(f'    Finished querying pairs: {elapsed_time} elapsed',flush=True)
    
        # Compute temporal distance for potential pairs
        self.pairs_hh_temporal_distance = np.abs(self.t_h[self.pairs_hh[:,0]] - self.t_h[self.pairs_hh[:,1]])
        self.pairs_kh_temporal_distance = np.abs(self.t_k[self.pairs_kh[:,0]] - self.t_h[self.pairs_kh[:,1]])
        
        # Drop pairs that do not fall within temporal cutoff
        hh_mask = (self.pairs_hh_temporal_distance < self.temporal_cutoff)
        kh_mask = (self.pairs_kh_temporal_distance < self.temporal_cutoff)
        self.pairs_hh = self.pairs_hh[hh_mask]
        self.pairs_hh_temporal_distance = self.pairs_hh_temporal_distance[hh_mask]
        self.pairs_kh = self.pairs_kh[kh_mask]
        self.pairs_kh_temporal_distance = self.pairs_kh_temporal_distance[kh_mask]
        
        left_hh = self.pairs_hh[:,0]
        right_hh = self.pairs_hh[:,1]
        left_kh = self.pairs_kh[:,0]
        right_kh = self.pairs_kh[:,1]
        
        # Compute spatial distance between remaining pairs
        self.pairs_hh_spatial_distance = np.sum((self.xy_h[left_hh] - self.xy_h[right_hh])**2,axis=1)**0.5
        self.pairs_kh_spatial_distance = np.sum((self.xy_k[left_kh] - self.xy_h[right_kh])**2,axis=1)**0.5
        
        # Create sparse adjacency and distance matrices
        
        # H-H matrices
        self.adjmat_hh = sparse.csr_array((np.ones(len(left_hh),dtype=np.int64),(left_hh,right_hh)),shape=(self.n_h,self.n_h))
        self.adjmat_hh += self.adjmat_hh.T
        self.adjmat_hh += sparse.eye(self.n_h)
        
        self.spatial_dmat_hh = sparse.csr_array((self.pairs_hh_spatial_distance,(left_hh,right_hh)),shape=(self.n_h,self.n_h))
        self.spatial_dmat_hh += self.spatial_dmat_hh.T
        
        self.temporal_dmat_hh = sparse.csr_array((self.pairs_hh_temporal_distance,(left_hh,right_hh)),shape=(self.n_h,self.n_h))
        self.temporal_dmat_hh += self.temporal_dmat_hh.T
        
        # K-H matrices
        self.adjmat_kh = sparse.csr_array((np.ones(len(left_kh),dtype=np.int64),(left_kh,right_kh)),shape=(self.n_k,self.n_h))
        
        self.spatial_dmat_kh = sparse.csr_array((self.pairs_kh_spatial_distance,(left_kh,right_kh)),shape=(self.n_k,self.n_h))
        
        self.temporal_dmat_kh = sparse.csr_array((self.pairs_kh_temporal_distance,(left_kh,right_kh)),shape=(self.n_k,self.n_h))
        
        t2 = time.time()
        elapsed_time = format_elapsed_time(t2-t1)
        print(f'    Finished building sparse distance matrices: {elapsed_time} elapsed\n',flush=True)
        
        return(None)
    
    def estimate_variogram(self,Cst,spatial_lags,temporal_lags,options={}):
        """
        param: Cst: space-time covariance model that includes free parameters (e.g., instance of ProductSumSTCovFun class)
        param: spatial_lags: edges of spatial lag bins (vector of length ns). Max value is used as spatial cutoff.
        param: temporal_lags: edges of temporal lag bins (vector of length nt). Max value is used as temporal cutoff.
        param: options: optional dict of kwargs to pass to optimizer 
        (see scipy.optimize.differential_evolution documentation)
        """
        
        left = self.pairs_hh[:,0]
        right = self.pairs_hh[:,1]
        hs = self.pairs_hh_spatial_distance
        ht = self.pairs_hh_temporal_distance
        gamma_vector = 0.5*(self.z_h[left] - self.z_h[right])**2
        
        # Ensure lags are sorted and all positive. 
        spatial_lags = np.sort(spatial_lags[(spatial_lags >= 0)&(spatial_lags <= self.spatial_cutoff)])
        temporal_lags = np.sort(temporal_lags[(temporal_lags >= 0)&(temporal_lags <= self.temporal_cutoff)])
        
        # Get spatial and temporal bin of distances
        hs_bin = np.digitize(hs,spatial_lags)
        ht_bin = np.digitize(ht,temporal_lags)

        num_spatial_lags = len(spatial_lags)
        num_temporal_lags = len(temporal_lags)
        num_sbin = num_spatial_lags - 1
        num_tbin = num_temporal_lags - 1

        # Pre-allocate arrays used to store output describing semivariogram
        gamma_bin = np.zeros((num_sbin,num_tbin))
        num_pairs = np.zeros((num_sbin,num_tbin))
        hs_center = np.zeros((num_sbin,num_tbin))
        hs_mean = np.zeros((num_sbin,num_tbin))
        ht_center = np.zeros((num_sbin,num_tbin))
        ht_mean = np.zeros((num_sbin,num_tbin))

        for i,sbin in enumerate(np.arange(1,num_spatial_lags)):
            for j,tbin in enumerate(np.arange(1,num_temporal_lags)):

                # Select pairs that fall within space-time bin
                m = (hs_bin == sbin)&(ht_bin == tbin)
                num_pairs[i,j] = np.sum(m)

                # Calculate value of semivariogram within bin 
                gamma_bin[i,j] = np.sum(gamma_vector[m])/(num_pairs[i,j])

                # Record bin centers
                hs_center[i,j] = (spatial_lags[i] + spatial_lags[i+1])/2
                ht_center[i,j] = (temporal_lags[j] + temporal_lags[j+1])/2

                # Record mean spatial and temporal distance separating pairs within bin
                hs_mean[i,j] = np.mean(hs[m])
                ht_mean[i,j] = np.mean(ht[m])
        
        # Fit the covariange model to the empirical variogram 
        Cst,status = fit_covariance_model(Cst,hs_mean,ht_mean,gamma_bin,w=num_pairs,options=options)
        
        if status == 'converged':
            self.Cst = Cst
            
        empirical_variogram = {'gamma_bin':gamma_bin,
                               'num_pairs':num_pairs,
                               'hs_center':hs_center,
                               'hs_mean':hs_mean,
                               'ht_center':ht_center,
                               'ht_mean':ht_mean}
        
        self.empirical_variogram = empirical_variogram
                    
        return(None)
    
class SimpleKriging(Kriging):
    """
    Simple kriging class (child class of Kriging)
    """
    def __init__(self,hard_points,krig_points,response_variable,mean_value=0.0,spatial_column='geometry',temporal_column='time_val'):
        """
        param: hard_points: geodataframe of "hard" observations (i.e., realizations of space-time random field)
        param: krig_points: geodataframe of points to be interpolated via kriging
        param: response_variable: column name corresponding to quantity of interest (z)
        param: mean_value: known, constant, mean value of response variable throughout the study area
        param: spatial_column: column name corresponding to pandas geoseries of point coordinates
        param: temporal_column: column name corresponding to time values. Must be numeric (e.g., # days since 2000).  
        """
        self.m_z = mean_value
        super().__init__(hard_points,krig_points,response_variable,spatial_column,temporal_column)
        
    def krig_values(self,n_max=1000,n_min=100):
        """
        Interpolate values via simple kringing. 
        
        param: n_max: maximum number of neighbors to include in kriging weight calculation. This is to reduce the 
        computational burden of inverting the Chh covariance matrix. For each kriging point, the n_max neighbors with 
        the highest covariance values that are within the spaital / temporal cutoffs will be selected. 
        param: n_min: minimum number of neighbors to include in kriging weight calculations. Due to the relay effect, it can
        be beneficial to add additional points even if their covariance with x0 is zero. 
        """

        # Get covariance matrices (note that these are scipy sparse CSR arrays)
        cov_hh = apply_covariance_sparse(self.Cst,self.spatial_dmat_hh,self.temporal_dmat_hh,self.adjmat_hh)
        cov_kh = apply_covariance_sparse(self.Cst,self.spatial_dmat_kh,self.temporal_dmat_kh,self.adjmat_kh)
        
        self.z_k = np.nan*np.ones(self.n_k)
        self.sigma_k = np.nan*np.ones(self.n_k)
        
        # Get variance of residuals
        sigma_kk = np.sqrt(self.Cst.cov(0,0))
        
        print('Interpolating values via simple kriging:')
        
        progress_step = np.ceil(0.1*self.n_k)
        update_progress = np.arange(self.n_k) % progress_step == 0
        update_progress[0] = False
        
        t1 = time.time()
        
        for k in range(self.n_k):
            
            if cov_kh[[k]].count_nonzero() > 0:
                
                # Determine which hard points to use in kriging weight calculation for point k
                # Ideally, we want this number to be between n_min and n_max
                # Prioritize those points that exhibit strongest correlation with k
            
                extra,adj_inds = self.adjmat_kh[[k]].nonzero()
                nonzero_mask = (cov_kh[[k],adj_inds] > 0)
                adj_nonzero_cov = adj_inds[nonzero_mask]
                adj_zero_cov = adj_inds[~nonzero_mask]
                adj_nonzero_cov = adj_nonzero_cov[np.argsort(cov_kh[[k],adj_nonzero_cov])[-n_max:]]
                n_adj_nonzero_cov = len(adj_nonzero_cov)

                if (n_adj_nonzero_cov < n_min):
                    
                    # The reason we add additional points even though they have zero correlation with k
                    # is because of the relay effect. For a good explanation of this, see section 3.6.1 of 
                    # Geostatistics: Modeling Spatial Uncertainty by Jean-Paul Chiles (2012, 2nd edition)
                    
                    n_adj_zero_cov = len(adj_zero_cov)
                    n_extra = min(n_min-n_adj_nonzero_cov,n_adj_zero_cov)   
                    relay = cov_hh[adj_nonzero_cov].toarray()[:,adj_zero_cov]
                    
                    # When adding extra points to reach n_min, select those that have greatest 
                    # correlation with already-included points that are correlated with k
                    
                    extra_inds = adj_zero_cov[np.argsort(np.max(relay,axis=0))[-n_extra:]]
                    h_inds = np.concatenate((adj_nonzero_cov,extra_inds))
                    n_h_included = n_adj_nonzero_cov + n_extra
                else:
                    h_inds = adj_nonzero_cov
                    n_h_included = n_adj_nonzero_cov
                
                # Create Ckh and Chh covariance matrices
                Ckh = cov_kh[[k],h_inds].reshape(1,n_h_included)
                Chh = cov_hh[h_inds].toarray()[:,h_inds]

                # Get value of z at relevant hard points
                z_h = self.z_h[h_inds]
                z_h.shape = (n_h_included,1)

                # Take pseudoinverse of Chh covariance matrix 
                Chh_inv = np.linalg.pinv(Chh, hermitian=True)
                
                # Get kriging weights
                Wkh = Ckh @ Chh_inv
                
                # Get kriging estimate of z_k
                self.z_k[k] = self.m_z + (Wkh @ (z_h - self.m_z))[0,0]

                # Get kriging standard deviation of z_k
                self.sigma_k[k] = np.sqrt(sigma_kk**2 - (Wkh @ Ckh.T)[0,0])
                
            else:
                self.z_k[k] = 0
                self.sigma_k[k] = sigma_kk
                
            if update_progress[k]:
                
                t2 = time.time()
                elapsed_time = format_elapsed_time(t2-t1)
                progress = np.round(k/self.n_k*100,-1).astype(int)
                
                print(f'    {k} / {self.n_k} ({progress}%) complete: {elapsed_time} elapsed',flush=True)
                
        t2 = time.time()
        elapsed_time = format_elapsed_time(t2-t1)
        print(f'    {self.n_k} / {self.n_k} (100%) complete: {elapsed_time} elapsed',flush=True)
            
        return(self.z_k,self.sigma_k)
    
class OrdinaryKriging(Kriging):
    """
    Ordinary kriging class (child class of Kriging)
    """
    def __init__(self,hard_points,krig_points,response_variable,spatial_column='geometry',temporal_column='time_val'):
        """
        param: hard_points: geodataframe of "hard" observations (i.e., realizations of space-time random field)
        param: krig_points: geodataframe of points to be interpolated via kriging
        param: response_variable: column name corresponding to quantity of interest (z)
        param: spatial_column: column name corresponding to pandas geoseries of point coordinates
        param: temporal_column: column name corresponding to time values. Must be numeric (e.g., # days since 2000).  
        """
        super().__init__(hard_points,krig_points,response_variable,spatial_column,temporal_column)
        
    def krig_values(self,n_max=1000,n_min=100):
        """
        Interpolate values via ordinary kringing. 
        
        param: n_max: maximum number of neighbors to include in kriging weight calculation. This is to reduce the 
        computational burden of inverting the Chh covariance matrix. For each kriging point, the n_max neighbors with 
        the highest covariance values that are within the spaital / temporal cutoffs will be selected. 
        param: n_min: minimum number of neighbors to include in kriging weight calculations. Due to the relay effect, it can
        be beneficial to add additional points even if their covariance with x0 is zero. 
        """

        # Get covariance matrices (note that these are scipy sparse CSR arrays)
        cov_hh = apply_covariance_sparse(self.Cst,self.spatial_dmat_hh,self.temporal_dmat_hh,self.adjmat_hh)
        cov_kh = apply_covariance_sparse(self.Cst,self.spatial_dmat_kh,self.temporal_dmat_kh,self.adjmat_kh)
        
        self.z_k = np.nan*np.ones(self.n_k)
        self.sigma_k = np.nan*np.ones(self.n_k)
        
        # Get variance of residuals
        sigma_kk = np.sqrt(self.Cst.cov(0,0))
        
        print('Interpolating values via ordinary kriging:')
        
        progress_step = np.ceil(0.1*self.n_k)
        update_progress = np.arange(self.n_k) % progress_step == 0
        update_progress[0] = False
        
        t1 = time.time()
        
        for k in range(self.n_k):
            
            if cov_kh[[k]].count_nonzero() > 0:
                
                # Determine which hard points to use in kriging weight calculation for point k
                # Ideally, we want this number to be between n_min and n_max
                # Prioritize those points that exhibit strongest correlation with k
            
                extra,adj_inds = self.adjmat_kh[[k]].nonzero()
                nonzero_mask = (cov_kh[[k],adj_inds] > 0)
                adj_nonzero_cov = adj_inds[nonzero_mask]
                adj_zero_cov = adj_inds[~nonzero_mask]
                adj_nonzero_cov = adj_nonzero_cov[np.argsort(cov_kh[[k],adj_nonzero_cov])[-n_max:]]
                n_adj_nonzero_cov = len(adj_nonzero_cov)

                if (n_adj_nonzero_cov < n_min):
                    
                    # The reason we add additional points even though they have zero correlation with k
                    # is because of the relay effect. For a good explanation of this, see section 3.6.1 of 
                    # Geostatistics: Modeling Spatial Uncertainty by Jean-Paul Chiles (2012, 2nd edition)
                    
                    n_adj_zero_cov = len(adj_zero_cov)
                    n_extra = min(n_min-n_adj_nonzero_cov,n_adj_zero_cov)   
                    relay = cov_hh[adj_nonzero_cov].toarray()[:,adj_zero_cov]
                    
                    # When adding extra points to reach n_min, select those that have greatest 
                    # correlation with already-included points that are correlated with k
                    
                    extra_inds = adj_zero_cov[np.argsort(np.max(relay,axis=0))[-n_extra:]]
                    h_inds = np.concatenate((adj_nonzero_cov,extra_inds))
                    n_h_included = n_adj_nonzero_cov + n_extra
                else:
                    h_inds = adj_nonzero_cov
                    n_h_included = n_adj_nonzero_cov
                
                # Create Ckh and Chh covariance matrices
                Ckh = cov_kh[[k],h_inds].reshape(1,n_h_included)
                Chh = cov_hh[h_inds].toarray()[:,h_inds]

                # Get value of z at relevant hard points
                z_h = self.z_h[h_inds]
                z_h.shape = (n_h_included,1)

                # Take pseudoinverse of Chh covariance matrix 
                Chh_inv = np.linalg.pinv(Chh, hermitian=True)
                
                # Estimate mean
                unit = np.ones((n_h_included,1))
                term1 = (unit.T @ Chh_inv @ z_h)[0,0]
                term2 = (unit.T @ Chh_inv @ unit)[0,0]
                m_z = term1 / term2
                
                # Estimate lambda values
                term1 = 1 - unit.T @ Chh_inv @ Ckh.T
                term3 = (unit @ term1) / term2
                lam_hk = Chh_inv @ (Ckh.T + term3)
                
                # Estimate lagrange multiplier
                mu = (unit.T @ Chh_inv @ Ckh.T - 1)[0,0] / term2
                
                # Get kriging estimate of z_k
                self.z_k[k] = m_z + (Ckh @ Chh_inv @ (z_h - m_z))[0,0]

                # Get kriging standard deviation of z_k
                var_k = sigma_kk**2 - (lam_hk.T @ Ckh.T)[0,0] - mu
                self.sigma_k[k] = np.sqrt(var_k)
                
            else:
                self.z_k[k] = 0
                self.sigma_k[k] = sigma_kk
                
            if update_progress[k]:
                
                t2 = time.time()
                elapsed_time = format_elapsed_time(t2-t1)
                progress = np.round(k/self.n_k*100,-1).astype(int)
                
                print(f'    {k} / {self.n_k} ({progress}%) complete: {elapsed_time} elapsed',flush=True)
                
        t2 = time.time()
        elapsed_time = format_elapsed_time(t2-t1)
        print(f'    {self.n_k} / {self.n_k} (100%) complete: {elapsed_time} elapsed',flush=True)
            
        return(self.z_k,self.sigma_k)