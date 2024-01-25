import numpy as np
import scipy.stats as stats
import scipy.spatial as spatial
import scipy.special as special
import scipy.optimize as so
import pandas as pd
import geopandas as gpd

# SphCovFun
# ExpCovFun
# GauCovFun
# CauCovFun
# MatCovFun

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
    
class MatCovFun(CovFun):
    """
    Matern covariance function class (child class of CovFun)
    """
    def __init__(self,a=None,a_bounds=(0,10),v=None,v_bounds=(0,10)):
        """
        param: a: scale parameter determining speed at which spatial / temporal dependence decays
        param: a_bounds: bounds on a if fitting as a free parameter
        param: v: Order of modified bessel function of the second kind
        param: v_bounds: Bounds on order of modified bessel function of the second kind
        """
        self.a = a
        self.a_bounds = a_bounds
        self.v = v
        self.v_bounds = v_bounds
        self.params = ['a','v']
        self.name = 'MatCovFun'
        super().__init__()
    
    def cov(self,h):
        """
        Return value of covariance function
        param: h: value of spatial or temporal distance between points
        """
        self.verify_all_fixed()
        c = (h/self.a)**self.v*special.kv(self.v,h/self.a)/(2**(self.v-1)*special.gamma(self.v))
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
    
class ProductSumSTCovFun(STCovFun):
    """
    Product-sum space-time covariance model (child class of STCovFun). 
    Described in detail by De Iaco, Myers, and Posa (doi:10.1016/S0167-7152(00)00200-5).
    """
    
    def __init__(self,Cs,Ct,k1=None,k2=None,k3=None,k1_bounds=(0,10),k2_bounds=(0,10),k3_bounds=(0,10)):
        """
        param: Cs: spatial component of covariance function
        param: Ct: temporal component of covariance function 
        param: k1: coefficient of product term
        param: k2: coefficient of spatial term
        param: k3: coefficient of temporal term
        param: k1_bounds: bounds on k1 if fitting as a free parameter
        param: k2_bounds: bounds on k2 if fitting as a free parameter
        param: k3_bounds: bounds on k3 if fitting as a free parameter
        """
        
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        self.k1_bounds = k1_bounds
        self.k2_bounds = k2_bounds
        self.k3_bounds = k3_bounds
        self.params = ['k1','k2','k3']
        self.name = 'ProductSumSTCovFun'
        super().__init__(Cs,Ct)
    
    def cov(self,hs,ht):
        """
        Return value of covariance function
        param: hs: value of spatial distance between points
        param: ht: value of temporal distance between points
        """
        self.verify_all_fixed()
        c = self.k1*self.Cs.cov(hs)*self.Ct.cov(ht) + self.k2*self.Cs.cov(hs) + self.k3*self.Ct.cov(ht)
        return c

### *** VARIOGRAM PARAMETER ESTIMATION FUNCTIONS *** ###
    
def fit_covariance_model(Cst,hs,ht,gamma,w=None,options={}):
    """
    Fit free parameters to empirical variogram data using weighted sum of squares.

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
    
    message = 'Estimating the following parameters via weighted least squares:\n'
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
        weighted_squared_error = np.sum(w*(gamma_hat - gamma)**2)/sum_w
        
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

def empirical_variogram(z,xy,t,spatial_lags,temporal_lags):
    """
    param: z: samples drawn from a space-time random field (vector of length n)
    param: xy: x-y coordinates associated with samples (n x 2 array)
    param: t: time values associated with samples (vector of length n)
    param: spatial_lags: edges of spatial lag bins (vector of length ns). Max value is used as spatial cutoff.
    param: temporal_lags: edges of temporal lag bins (vector of length nt). Max value is used as temporal cutoff.
    """
    
    # Ensure lags are sorted and all positive. 
    spatial_lags = np.sort(spatial_lags[spatial_lags >= 0])
    temporal_lags = np.sort(temporal_lags[temporal_lags >= 0])
    
    # Define cutoff spatial / temporal distances. 
    # We won't consider any pairwise combinations that are more than this far apart in space or time.
    spatial_cutoff = np.max(spatial_lags)
    temporal_cutoff = np.max(temporal_lags)
    
    # Create KDTree that we can use for fast spatial indexing
    spatial_kd_tree = spatial.KDTree(xy)
    
    # Find spatial pairs that are within cutoff distance
    spatial_pairs = spatial_kd_tree.query_pairs(r=spatial_cutoff,output_type='ndarray')
    
    # Compute temporal distance for potential pairs
    temporal_distance = np.abs(t[spatial_pairs[:,0]] - t[spatial_pairs[:,1]])
    
    # Drop pairs that do not fall within temporal cutoff
    m = (temporal_distance < temporal_cutoff)
    pairs = spatial_pairs[m]
    
    # Compute spatial / temporal distance and squared difference in sampled value of z
    temporal_distance = temporal_distance[m]
    spatial_distance = np.sum((xy[pairs[:,0]] - xy[pairs[:,1]])**2,axis=1)**0.5
    squared_difference = (z[pairs[:,0]] - z[pairs[:,1]])**2
    
    # Get spatial and temporal bin of distances
    spatial_bin = np.digitize(spatial_distance,spatial_lags)
    temporal_bin = np.digitize(temporal_distance,temporal_lags)
    
    num_spatial_lags = len(spatial_lags)
    num_temporal_lags = len(temporal_lags)
    num_sbin = num_spatial_lags - 1
    num_tbin = num_temporal_lags - 1
    
    # Pre-allocate arrays used to store output describing semivariogram
    gamma_value = np.zeros((num_sbin,num_tbin))
    num_pairs = np.zeros((num_sbin,num_tbin))
    sbin_center = np.zeros((num_sbin,num_tbin))
    sbin_mean = np.zeros((num_sbin,num_tbin))
    tbin_center = np.zeros((num_sbin,num_tbin))
    tbin_mean = np.zeros((num_sbin,num_tbin))
    
    for i,sbin in enumerate(np.arange(1,num_spatial_lags)):
        for j,tbin in enumerate(np.arange(1,num_temporal_lags)):
            
            # Select pairs that fall within space-time bin
            m = (spatial_bin == sbin)&(temporal_bin == tbin)
            num_pairs[i,j] = np.sum(m)
            
            # Calculate value of semivariogram within bin 
            gamma_value[i,j] = np.sum(squared_difference[m])/(2*num_pairs[i,j])
            
            # Record bin centers
            sbin_center[i,j] = (spatial_lags[i] + spatial_lags[i+1])/2
            tbin_center[i,j] = (temporal_lags[j] + temporal_lags[j+1])/2
            
            # Record mean spatial and temporal distance separating pairs within bin
            sbin_mean[i,j] = np.mean(spatial_distance[m])
            tbin_mean[i,j] = np.mean(temporal_distance[m])
    
    return(gamma_value,num_pairs,sbin_center,sbin_mean,tbin_center,tbin_mean)

