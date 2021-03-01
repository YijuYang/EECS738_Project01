import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv, det



def gauss_fun(x, mu, var):
    pd = (1/(np.sqrt(2*np.pi*var)))*np.exp(-1*np.square(x-mu)/(2*var))
    return pd
