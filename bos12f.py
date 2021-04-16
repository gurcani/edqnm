#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 08:28:35 2021

@author: ogurcan
"""

#
# Solves the system described in Bos et al Physics of Fluids 24, 015108 (2012)
#

import numpy as np
from time import time
from scipy.integrate import solve_ivp,cumtrapz
from numba import njit,prange

k0=1e-2
Ndec=6
kN=k0*10**Ndec
Npd=12
N=Npd*Ndec
g=(kN/k0)**(1/N)
k=k0*g**np.arange(N)
kL=8*k0
E0k0=k**4*np.exp(-(k/kL)**2)
E0k=E0k0/np.cumsum(E0k0*k*(g-1))
lam=0.49
nu=1e-6

@njit(parallel=True, fastmath=True)
def Tf(k,Ek,eta_k,t,res):
    for i in prange(k.shape[0]):
        for j in range(k.shape[0]):
            for l in range(k.shape[0]):
                if ( ((k[l]>=k[i]) and (k[j]>=(k[l]-k[i])) and (k[j]<=(k[l]+k[i]))) or ((k[l]<=k[i]) and (k[j]>=(k[i]-k[l])) and (k[j]<=k[l]+k[i])) ):
                    x=(k[l]**2+k[j]**2-k[i]**2)/2/k[l]/k[j]
                    y=(k[l]**2+k[i]**2-k[j]**2)/2/k[l]/k[i]
                    z=(k[i]**2+k[j]**2-k[l]**2)/2/k[i]/k[j]
                    theta=(1-np.exp(-(eta_k[i]+eta_k[j]+eta_k[l])*t))/(eta_k[i]+eta_k[j]+eta_k[l])
                    res[i]+=theta*(x*y+z**3)*(k[i]**2*k[j]*Ek[j]*Ek[l]-k[j]**3*Ek[l]*Ek[i])*(g-1)**2
    return res

def fn(t,Ek):
    print(t)
    eta_k=lam*np.sqrt(np.cumsum(Ek*k**3*(g-1)))+nu*k**2
    res=np.zeros_like(k)
    return Tf(k,Ek,eta_k,t,res)-2*nu*k**2*Ek

t=np.arange(0,500,1.0)
res=solve_ivp(fn,(t[0],t[-1]),E0k,t_eval=t,dense_output=True,atol=1e-18,rtol=1e-8)
np.savez("out.npz",t=t,y=res.y,k=k)
