import numpy as np
import scipy.stats as stats
import scipy.spatial as spatial
import pandas as pd
import geopandas as gpd

def empirical_semivariogram(z,xy,t,spatial_lags,temporal_lags):
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