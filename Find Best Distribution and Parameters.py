#!/usr/bin/env python
# coding: utf-8

# # Find Best Distribution and Parameters

# In[ ]:


Load Modules


# In[ ]:


import warnings
import numpy as np
import pandas as pd
import scipy.stats as st
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (16.0, 12.0)
matplotlib.style.use('ggplot')


# Load Data
# 
# Data for this example is price history.

# In[ ]:


whole = pd.read_csv('C:\\Users\\SPY_IntraDayPriceFIXED.csv')

arr = whole['PRICE'].pct_change().dropna()


# Build Definition

# In[ ]:


def fit_scipy_distributions(array, bins, plot_hist = False, plot_best_fit = False, plot_all_fits = False):
    """
    Fits a range of Scipy's distributions (see scipy.stats) against an array-like input.
    Returns the sum of squared error (SSE) between the fits and the actual distribution.
    Can also choose to plot the array's histogram along with the computed fits.
    N.B. Modify the "CHANGE IF REQUIRED" comments!
    
    Input: array - array-like input
           bins - number of bins wanted for the histogram
           plot_hist - boolean, whether you want to show the histogram
           plot_best_fit - boolean, whether you want to overlay the plot of the best fitting distribution
           plot_all_fits - boolean, whether you want to overlay ALL the fits (can be messy!)
    
    Returns: results - dataframe with SSE and distribution name, in ascending order (i.e. best fit first)
             best_name - string with the name of the best fitting distribution
             best_params - list with the parameters of the best fitting distribution.
    """
    
    if plot_best_fit or plot_all_fits:
        assert plot_hist, "plot_hist must be True if setting plot_best_fit or plot_all_fits to True"
    
    # Returns un-normalised (i.e. counts) histogram
    y, x = np.histogram(np.array(array), bins=bins)
    
    # Some details about the histogram
    bin_width = x[1]-x[0]
    N = len(array)
    x_mid = (x + np.roll(x, -1))[:-1] / 2.0 # go from bin edges to bin middles
    
    # selection of available distributions
    # CHANGE THIS IF REQUIRED: Less items will be quicker.
    #DISTRIBUTIONS = [st.alpha,st.cauchy,st.cosine,st.laplace,st.levy,st.levy_l,st.norm]
    DISTRIBUTIONS = [        
        st.alpha,st.burr,st.cauchy,st.chi,st.chi2,st.cosine,
        st.dgamma,st.dweibull,st.erlang,st.expon,st.exponnorm,st.exponweib,st.exponpow,st.f,st.fatiguelife,st.fisk,
        st.foldcauchy,st.foldnorm,st.genlogistic,st.genpareto,st.gennorm,st.genexpon,
        st.genextreme,st.gausshyper,st.gamma,st.gengamma,st.genhalflogistic,st.gilbrat,st.gompertz,st.gumbel_r,
        st.gumbel_l,st.halfcauchy,st.halflogistic,st.halfnorm,st.halfgennorm,st.hypsecant,st.invgamma,st.invgauss,
        st.invweibull,st.johnsonsb,st.johnsonsu,st.ksone,st.kstwobign,st.laplace,
        st.logistic,st.loggamma,st.loglaplace,st.lognorm,st.lomax,st.maxwell,st.mielke,st.nakagami,st.ncx2,st.ncf,
        st.nct,st.norm,st.pareto,st.pearson3,st.powerlaw,st.powerlognorm,st.powernorm,st.rdist,st.reciprocal,
        st.rayleigh,st.rice,st.recipinvgauss,st.t,st.vonmises,st.vonmises_line,st.wald,st.weibull_min,st.weibull_max,st.wrapcauchy
    ]

    if plot_hist:
        fig, ax = plt.subplots()
        h = ax.hist(np.array(array), bins = bins, color = 'w')

    # loop through the distributions and store the sum of squared errors
    # so we know which one eventually will have the best fit
    sses = []
    for dist in tqdm(DISTRIBUTIONS):
        name = dist.__class__.__name__[:-4]

        params = dist.fit(np.array(array))
        arg = params[:-2]
        loc = params[-2]
        scale = params[-1]

        pdf = dist.pdf(x_mid, loc=loc, scale=scale, *arg)
        pdf_scaled = pdf * bin_width * N # to go from pdf back to counts need to un-normalise the pdf

        sse = np.sum((y - pdf_scaled)**2)
        sses.append([sse, name])

        # Not strictly necessary to plot, but pretty patterns
        if plot_all_fits:
            ax.plot(x_mid, pdf_scaled, label = name)
    
    if plot_all_fits:
        plt.legend(loc=1)

    # CHANGE THIS IF REQUIRED
    ax.set_xlabel('Percent Change')
    ax.set_ylabel('Count')

    # Things to return - df of SSE and distribution name, the best distribution and its parameters
    results = pd.DataFrame(sses, columns = ['SSE','distribution']).sort_values(by='SSE') 
    best_name = results.iloc[0]['distribution']
    best_dist = getattr(st, best_name)
    best_params = best_dist.fit(np.array(array))
    
    if plot_best_fit:
        new_x = np.linspace(x_mid[0] - (bin_width * 2), x_mid[-1] + (bin_width * 2), 1000)
        best_pdf = best_dist.pdf(new_x, *best_params[:-2], loc=best_params[-2], scale=best_params[-1])
        best_pdf_scaled = best_pdf * bin_width * N
        ax.plot(new_x, best_pdf_scaled, label = best_name)
        plt.legend(loc=1)
    
    if plot_hist:
        plt.show()

    return results, best_name, best_params


# In[ ]:


test_array = np.array(whole['PRICE'].pct_change().dropna())
sses, best_name, best_params = fit_scipy_distributions(test_array, 100, plot_hist = True, plot_best_fit=True, plot_all_fits=False)


# best_name: 'dgamma'
# best_params: (0.6607683647053637, 2.650034865769867e-06, 0.0005753356628344494)
# sses: Returns a DataFrame with two columns {SSE: Sum of Square Errors, distribution: Name of distribution}

# In[ ]:




