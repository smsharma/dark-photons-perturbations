""" Functions for computing the analytic PDF.
"""

import numpy as np

from scipy.special import gamma, factorial

def kbins(Nmax, k0, kmax):
    
    Delta = 1/(Nmax - 1) * np.log(kmax/k0)
    return k0 * np.exp(np.arange(Nmax)*Delta)

def CoeffPow1(z, log_P_m, b, Nmax, k0, kmax):
    
    Delta = 1/(Nmax - 1) * np.log(kmax/k0)
    kn = k0 * np.exp(np.arange(Nmax) * Delta)
    
    etam = b + 2j*np.pi/(Nmax * Delta) * (np.arange(Nmax+1) - Nmax/2)
    
    Pn = 10**log_P_m(z, kn)[0] * np.exp(
        -b * np.arange(Nmax) * Delta
    )
    
    cn = np.fft.fft(Pn / Pn.size)
    
    cnsym = np.zeros(Nmax + 1, dtype='complex')
    cnsym[:Nmax//2] = k0 ** (-etam[:Nmax//2]) * np.conj(cn[Nmax//2:0:-1])
    cnsym[Nmax//2:] = k0 ** (-etam[Nmax//2:]) * cn[:Nmax//2+1]
    
    result = np.stack([cnsym, etam])
    result[0,0] /= 2.
    result[0,Nmax] /= 2
    
    return result

def W(x):
    return (3/x**3)*(np.sin(x)-x*np.cos(x))

def Jsigma_over_r_fac(nu):
    return (
        9*2**(-1-nu)*(1+nu)*gamma(-1+nu)*np.sin(np.pi*nu/2)
        / (np.pi**2*(-3+nu))
    )

def Jxi_over_r_fac(nu):
    return (-1/np.pi**2)*3*2**(-2-nu)*(1+nu)*gamma(-1+nu)*np.sin(np.pi*nu/2)

def sigmaR(z, log_P_m, b, Nmax, k0, kmax):
    
    rlist2 = np.arange(-4, 40.36, 0.01)
    
    cn  = CoeffPow1(z, log_P_m, b, Nmax, k0, kmax)
    
    r_fac_list = [(10**r)**(-3-cn[1,:]) for r in rlist2]
    J_fac_list = Jsigma_over_r_fac(cn[1,:])
    
    aux = [np.real(cn[0,:] * r_fac * J_fac_list) for r_fac in r_fac_list]
    
    return np.stack(
        [10**rlist2, np.sum(aux, axis=-1)]
    )

def sigmaRfunction(x, z, log_P_m, b, Nmax, k0, kmax):

    sigmaR_res = sigmaR(z, log_P_m, b, Nmax, k0, kmax)
    
    return np.exp(np.interp(
        np.log(x), np.log(sigmaR_res[0]), np.log(sigmaR_res[1])
    ))

def xiR(z, log_P_m, b, Nmax, k0, kmax):
    
    rlist2 = np.arange(-4, 40.36, 0.01)
    
    cn = CoeffPow1(z, log_P_m, b, Nmax, k0, kmax)
    
    r_fac_list = [(10**r)**(-3-cn[1,:]) for r in rlist2]
    J_fac_list = Jxi_over_r_fac(cn[1,:])
    
    aux = [np.real(cn[0,:] * r_fac * J_fac_list) for r_fac in r_fac_list]
    
    return np.stack(
        [10**rlist2, np.sum(aux, axis=-1)]
    )

def xiRfunction(x, z, log_P_m, b, Nmax, k0, kmax):
    
    xiR_res = xiR(z, log_P_m, b, Nmax, k0, kmax)
    
    return np.exp(np.interp(
        np.log(x), np.log(xiR_res[0]), np.log(xiR_res[1])
    ))

def t10fun(z, log_P_m, b, Nmax, k0, kmax, R):
    
    tarray=np.arange(-5.,40.+0.01,0.01,dtype=float)

    return np.stack([
        10**tarray, 
        sigmaRfunction(R*(10**tarray)**(1/3), z, log_P_m, b, Nmax, k0, kmax),
           xiRfunction(R*(10**tarray)**(1/3), z, log_P_m, b, Nmax, k0, kmax)
    ])

def fun1(x, z, log_P_m, b, Nmax, k0, kmax, R):
    
    t10 = t10fun(z, log_P_m, b, Nmax, k0, kmax, R)
    
    return np.exp(np.interp(np.log(x), np.log(t10[0]), np.log(t10[1])))

def fun2(x, z, log_P_m, b, Nmax, k0, kmax, R):
    
    t10 = t10fun(z, log_P_m, b, Nmax, k0, kmax, R)
    
    return np.exp(np.interp(np.log(x), np.log(t10[0]), np.log(t10[2])))

def fup(x):
    if x > 0.005:
        return 9/2*(x-np.sin(x))**2/(1-np.cos(x))**3 
    else:
        # Taylor Expansion
        return (
            1. + 3*x**2/20 + 37/2800*x**4 
            + x**6/1120 + 2647/51744000*x**8
            + 31649/12108096000*x**10
        )

def gup(x):
    return 3/20*(6*(x-np.sin(x)))**(2/3)

def fdown(x):
    if x > 0.005:
        return 9/2*(np.sinh(x)-x)**2/(np.cosh(x)-1)**3
    else:
        # Taylor Expansion
        return (
            1. - 3*x**2/20 + 37/2800*x**4 
            - x**6/1120 + 2647/51744000*x**8
            - 31649/12108096000*x**10
        )
    
def gdown(x):
    return -3/20*(6*(np.sinh(x)-x))**(2/3)
    
def fp(t):
    return (9*(-1+np.cos(t))*(-t+np.sin(t)))/(1-np.cos(t))**3-27*np.sin(t)*(-t+np.sin(t))**2/(2*(1-np.cos(t))**4)

def gp(t):
    return 3**(2/3)*(1-np.cos(t))/(5*2**(1/3)*(t-np.sin(t))**(1/3))

def fpund(t):
    return 9*(-t+np.sinh(t))/(-1+np.cosh(t))**2 - 27*np.sinh(t)*(-t+np.sinh(t))**2/(2*(-1+np.cosh(t))**4)
    
def gpund(t):
    return - 3**(2/3)*(-1+np.cosh(t))/(5*2**(1/3)*(-t+np.sinh(t))**(1/3))

from scipy.optimize import root_scalar
#fup(x) is larger than one
def inversa(t):
    def invfup(x):
        return fup(x)-t
    if np.isclose(t,1.):
        # Catches this edge case. fup(0) = 1. 
        return 0.
    return root_scalar(invfup,bracket=[0.001,2*np.pi-1e-10]).root

#fdown is always between 0 and 1

def invDown(t):
    def invfdown(x):
        return fdown(x)-t
    if np.isclose(t,1.):
        # Catches this edge case. fdown(0) = 1. 
        return 0. 
    return root_scalar(invfdown,bracket=[0.001,100]).root

def Fup(x):
    
    try: 
        
        _ = x[0]
        return [gup(inversa(x_val)) for x_val in x]
        
    except TypeError: 
        
        return gup(inversa(x))

def Fdown(x):
    
    try: 
        
        _ = x[0]
        return [gdown(invDown(x_val)) for x_val in x]
    
    except TypeError:
        
        return gdown(invDown(x))

def Fprimeup(x):
    
    try:
        
        _ = x[0]
        res = np.array([gp(inversa(x_val))/fp(inversa(x_val)) for x_val in x])
        # Return the limit.
        if res[np.isclose(x,1)].size > 0:
            res[np.isclose(x,1)] = 1.
            
        return res
    
    except TypeError:
        
        if not np.isclose(x,1):
        
            return gp(inversa(x))/fp(inversa(x))
        
        else:
            
            return 1.

def Fprimedown(x):
    
    try:
        
        _ = x[0]
        res = np.array([gpund(invDown(x_val)) / fpund(invDown(x_val)) for x_val in x])
        # Return the limit. 
        if res[np.isclose(x,1)].size > 0:
            res[np.isclose(x,1)] = 1.
        
        return res
    
    except TypeError:
        
        if not np.isclose(x,1):
            return gpund(invDown(x))/fpund(invDown(x))
        else:
            return 1. 

def FEdS(x):
    
    try:
        
        _ = x[0]
    
        soln = np.zeros_like(x)
        if x[x > 1].size > 0:
            soln[x > 1]  = Fup(x[x > 1])
        if x[x <= 1].size > 0:
            soln[x <= 1] = Fdown(x[x <= 1])
        
        return soln
        
    except TypeError:
        
        if x > 1: 
            
            return Fup(x)
        
        else:
            return Fdown(x)
    
def Fprime(x):
    
    try:
        
        _ = x[0]
        
        soln = np.zeros_like(x)
        if x[x > 1].size > 0:
            soln[x > 1]  = Fprimeup(x[x > 1])
        if x[x <= 1].size > 0:
            soln[x <= 1] = Fprimedown(x[x <= 1])
        
        return soln
    
    except TypeError: 
        
        
        if x > 1:
            return Fprimeup(x)
        else:
            return Fprimedown(x)

def Pref(x, z, log_P_m, b, Nmax, k0, kmax, R):
    return (
        (1/np.sqrt(2*np.pi*fun1(x, z, log_P_m, b, Nmax, k0, kmax, R)))
        * (
            Fprime(x)+FEdS(x)/x
            -FEdS(x)*fun2(x, z, log_P_m, b, Nmax, k0, kmax, R)
            / fun1(x, z, log_P_m, b, Nmax, k0, kmax, R) / x
        )
    )                

def Aasp(x, z, log_P_m, b, Nmax, k0, kmax, R):

    xiR_res    = xiRfunction(R, z, log_P_m, b, Nmax, k0, kmax)
    sigmaR_res = sigmaRfunction(R, z, log_P_m, b, Nmax, k0, kmax)
    
    return (
        1. 
        + (4/21 - (xiR_res/sigmaR_res))*np.log(x) 
        + 0.04725027522494461*np.log(x)**2 
        + 0.02671239809025271*np.log(x)**3
    )

def analytic_pdf(one_plus_delt, z, log_P_m, Nmax, k0, kmax, R, b=-0.95):

    """Generates the analytic probability density function. 

    :param one_plus_delt: Array of 1 + fractional overdensity.
    :param z: Redshift. 
    :param log_P_m: Function return log(linear matter power spectrum), taking arguments in k and z. 
    :param Nmax: Number of log Fourier transform bins. 
    :param k0: Minimum k value in log Fourier transform.
    :param kmax: Maximum k value in log Fourier transform. 
    :param R: Smoothing scale. 
    :param b: Bias. 
    :return: Probability density function as a function of `one_plus_delt` at redshift `z`.  

    """

    # Unnormalized PDF.
    fun1_res =  fun1(one_plus_delt, z, log_P_m, b, Nmax, k0, kmax, R)

    unnorm_pdf = (
        Pref(one_plus_delt, z, log_P_m, b, Nmax, k0, kmax, R)
        * np.exp(-FEdS(one_plus_delt)**2/2/fun1_res)
    )
    
    # Normalization factor.
    _Aasp = 1. / np.trapz(unnorm_pdf, one_plus_delt)
    
    return unnorm_pdf * _Aasp