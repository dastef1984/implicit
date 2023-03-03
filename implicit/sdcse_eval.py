#Calc Standard Discounted Cumulative Squared Error

import numpy as np
import pandas as pd

def sdcse(relvec, predvec):
    """ Calculate STANDARD DISCOUNTED CUMULATIVE SQUARED ERROR
    as described in the paper Evaluating Top-N Recommendations
    Using Ranked Error Approach: An Empirical Analysis


    Parameters
    ----------
    relvec : relevance vector
        relevance vector for each user
    predvec : predicted scores for N items
    
    Each relvec and each predvec are within [1, 5] and [0, 1]
    intervals for explicit and implicit feedback.

    Returns
    -------
    float
        the calculated SDCSE
    """

    relvec.sort()
    predvec.sort()

    DCSE_at_N = 0
    n = len(relvec)
    SE_des = []
    WDCSE_at_N = 0

    for i in range (1,n+1):
        squared_difference = (relvec[i - 1] - predvec[i - 1])**2
        DCSE = squared_difference / np.log2(i + 1)
        DCSE_at_N += DCSE
    

    for j in range(1,n+1):
        element = (relvec[j - 1] - predvec[j - 1])**2
        SE_des.append(element)

    SE_des.sort(reverse=True) 

    for k in range(1,n+1):
        element2 = SE_des[k - 1] / np.log2(k + 1)
        WDCSE_at_N += element2
    
    SDCSE_at_N = 1 - (DCSE_at_N / WDCSE_at_N)

    return SDCSE_at_N