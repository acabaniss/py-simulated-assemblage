"""Generate simulated assemblages and compare simulation vs. observed artifacts.

Author: Andrew Cabaniss
"""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import random as rd

def simulate_assemblages_collection(context_counts, N = 10000, p_dist = 2, center = 'observed'):
    """Calculate a theoretical object-assemblage matrix with similar marginal probabilities for an entire site.
    
    Parameters
    ----------
    context_counts :pandas.DataFrame
        Rows need to be the contexts, columns need to be integer counts of objects
    N : int
        number of theoretical assemblages to generate
    p_dist : int
        Dimensions for the Minkowski distance: 1 for Manhattan/taxi-cab, 2 for Euclidian
    center : {'observed', 'calculated'}
        If observed: the mean of all simulated distributions is used
        If calculated: the expected mean is used instead.
        
    Returns
    -------
    results : dict
        The results dict stores different data from the simulation:
        1. raw_simulation - the numpy.array of the simulated assemblages
        2. raw_observed - the formatted observed data as a numpy.array
        3. distance_simulation - the distances from the center of the simulated distribtion
        4. distance_observed - the distance of our actual assemblage from the center of the simulated distribution
        5. p-value - the proportion of simulated assemblages further from the center than the observed assemblages
        6. N - number of assemblages generated
        7. p_dist - dimensionality of the distance used
        8. center - the center chosen for distances to be calculated from
    """
    # Value checks
    if center not in ['observed', 'calculated']:
        raise ValueError("center neither 'observed' nor 'calculated")
    
    #First, the observed data are formatted and used to construct the parameters for a weighted random distribution
    observed = np.asanyarray(context_counts) #convert from DataFrame to ndarray
    num_objects = observed.sum().sum() #total number of objects across all contexts/types
    num_contexts = observed.shape[0] # number of contexts
    num_types = observed.shape[1] #number of different types of objects
    marginal_types = observed.sum(axis = 0)*1./num_objects #marginal distribution of the types of objects
    marginal_contexts = observed.sum(axis = 1)*1./num_objects #marginal distribution of the contexts 
    
    #generate simulated assemblages
    collector = np.zeros((N,num_contexts,num_types))
    for i in range(N):
        for o in range(num_objects):
            x = rd.choices(range(num_contexts),weights = marginal_contexts)
            y = rd.choices(range(num_types),weights = marginal_types)
            collector[i,x,y] += 1
            
    #Determine the center to be used for measuring all distances
    if center == 'calculated':
        center_calc = np.outer(marginal_contexts, marginal_types)*num_objects #calculate the theoretical center of the distribution
    elif center == 'observed':
        center_calc = collector.mean(axis = 0) #calculate the actual center of teh distribution
    
    #prepare the distribution, observed data, and center of teh distribution for distance calculation
    center_calc_flat = center_calc.reshape(1,-1) #reshape to be flat for distances
    collector_flat = np.reshape(collector,(N, -1)) #reshape to be flat for distances
    data_flat = np.asanyarray(observed).reshape(1, -1) #reshape to be flat for distances
    
    #calculate distances within the simulated distribution
    dists_boot = scipy.spatial.distance_matrix(center_calc_flat, collector_flat, p = p_dist).flatten()
    dist_real = scipy.spatial.distance_matrix(center_calc_flat, data_flat , p = p_dist).flatten()[0]
    
    #return results
    result = {'raw_simulation' : collector,
              'raw_observed' : observed,
              
             'distance_simulation' : dists_boot,
             'distance_observed' : dist_real,
             'p-value' : sum(dists_boot>=dist_real)*1./len(dists_boot),
             'N' : N,
             'p_dist' : p_dist,
             'center' : center}
    return result

def simulate_assemblage_single(context_counts, context_index, N = 10000, p_dist = 2, center = 'observed'):
    """Calculate a theoretical object-assemblage matrix with similar marginal probabilities for an entire site.
    
    Parameters
    ----------
    context_counts :pandas.DataFrame
        Rows need to be the contexts, columns need to be integer counts of objects
    context_index : int
        which row needs to be used
    N : int
        number of theoretical assemblages to generate
    p_dist : int
        Dimensions for the Minkowski distance: 1 for Manhattan/taxi-cab, 2 for Euclidian
    center : {'observed', 'calculated'}
        If observed: the mean of all simulated distributions is used
        If calculated: the expected mean is used instead.
        
    Returns
    -------
    results : dict
        The results dict stores different data from the simulation:
        1. raw_simulation - the numpy.array of the simulated assemblages
        2. raw_observed - the formatted observed data as a numpy.array
        3. distance_simulation - the distances from the center of the simulated distribtion
        4. distance_observed - the distance of our actual assemblage from the center of the simulated distribution
        5. p-value - the proportion of simulated assemblages further from the center than the observed assemblages
        6. N - number of assemblages generated
        7. p_dist - dimensionality of the distance used
        8. center - which center definition used
    """
    if center not in ['observed', 'calculated']:
        raise ValueError("center neither 'observed' nor 'calculated")
    
    #First, the observed data are formatted and used to construct the parameters for a weighted random distribution
    observed_sites = np.asanyarray(context_counts) #convert from DataFrame or other structure to ndarray
    observed = observed_sites[context_index]
    num_objects = observed.sum() #objects in this context
    num_types = observed_sites.shape[1] #number of different types of objects
    marginal_types = observed_sites.sum(axis = 0)*1./observed_sites.sum().sum() #marginal distribution of the types of objects
    
    #generate simulated assemblages
    collector = np.zeros((N,num_types))
    for i in range(N):
        for o in range(num_objects):
            y = rd.choices(range(num_types),weights = marginal_types)
            collector[i,y] += 1
    
    #Determine the center to be used for measuring all distances
    if center == 'calculated':
        center_calc = marginal_types*num_objects #calculate the theoretical center of the distribution
    elif center == 'observed':
        center_calc = collector.mean(axis = 0) #calculate the actual center of teh distribution
    
    #calculate distances within the simulated distribution
    dists_boot = scipy.spatial.distance_matrix(center_calc.reshape(1,-1), collector, p = p_dist).flatten()
    dist_real = scipy.spatial.distance_matrix(center_calc.reshape(1,-1), observed.reshape(1,-1) , p = p_dist).flatten()[0]
    
    #return results
    result = {'raw_simulation' : collector,
              'raw_observed' : observed,
             'distance_simulation' : dists_boot,
             'distance_observed' : dist_real,
             'p-value' : sum(dists_boot>=dist_real)*1./len(dists_boot),
             'N' : N,
             'p_dist' : p_dist,
             'center' : center}
    return result