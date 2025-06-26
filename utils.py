import numpy as np 
import scipy, scipy.misc, scipy.integrate 

solve_ivp = scipy.integrate.solve_ivp 


def rk4(fun, y0, t, dt, *args, **kwargs):
    dt2 = dt/2.0
    k1 = fun(y0, t, *args, **kwargs)
    k2 = fun(y0+ (dt2 * k1), t+dt2, *args, **kwargs)
    k3 = fun(y0+ (dt2 * k2), t+dt2, *args, **kwargs)
    k4 = fun(y0 + (dt*k3), t+dt2, *args, **kwargs)

    dy =  (1/6)*dt*(k1+(2*(k2+k3))+k4)

    return dy 

def L2Loss(u,v):
    return (u-v).pow(2).mean()

 