import numpy as np 
import scipy, scipy.misc, scipy.integrate 
import torch 

solve_ivp = scipy.integrate.solve_ivp 


def rk4(fun, y0, t, h, *args, **kwargs):
    h2 = h/2.0
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0+ (h2 * k1), t+h2, *args, **kwargs)
    k3 = fun(y0+ (h2 * k2), t+h2, *args, **kwargs)
    k4 = fun(y0 + (h*k3), t+h2, *args, **kwargs)

    dy =  (1/6)*h*(k1+(2*(k2+k3))+k4)

    return dy 

def L2Loss(u,v):
    return (u-v).pow(2).mean()

 