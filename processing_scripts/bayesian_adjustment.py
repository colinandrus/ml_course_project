### To compute Bayesian-adjusted grant/appeal/appeal-grant rates (see notebook for methodology and description)

import numpy as np 
import pandas as pd 

# calibrate beta priors 

def calibrate_beta_priors(aggregate_mean): 
    """
    Takes aggregate rate and return Beta priors (alpha, beta) with prior mean approximating aggregate rate, 
    with effective sample size of 10 
    """
    
    rounded_rate = np.round(aggregate_mean, 1)
    alpha = int(rounded_rate * 10) 
    beta = 10 - alpha 
    
    return alpha, beta 


# compute posterior mean given beta priors and observed data 

def compute_posterior_mean(alpha_prior, beta_prior, num_positives, num_total): 
    """ 
    Takes Beta priors (alpha, beta) along with observed data (num_total, num_positives) 
    and returns posterior mean 
    """
    
    updated_alpha = alpha_prior + num_positives 
    updated_beta = beta_prior + num_total - num_positives 
    
    posterior_mean = float(updated_alpha) / (updated_alpha + updated_beta)
    
    return posterior_mean 

# the single function to call 

def get_beta_adj_rate(aggregate_mean, num_positives, num_total): 
    """ 
    Takes aggregate mean as a float (from 0 to 1), num_total (integer), and num_positives (integer) 
    and return the 'Beta-adjusted' rate. 
    Example: if in total 30% of Chinese nationality cases were granted, and a specific judge saw 20 cases 
    and granted 14 of them, input aggregate_mean=0.3, num_total=20, and num_positives=14 
    """
    
    if type(aggregate_mean) is not float: 
        raise ValueError("Please enter a float for aggregate mean!")
        
    if aggregate_mean < 0 or aggregate_mean > 1: 
        raise ValueError("Aggregate mean must be between 0 and 1!")
            
    alpha_prior, beta_prior = calibrate_beta_priors(aggregate_mean)
    posterior_mean = compute_posterior_mean(alpha_prior, beta_prior, num_positives, num_total)
    
    return posterior_mean