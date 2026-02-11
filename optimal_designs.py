from scipy.special import expit
import numpy as np
from scipy.stats import norm # type: ignore
import scipy.optimize as op # type: ignore
import pandas as pd

def load_param(model = 'logistic',  g = None, theta = None):
    # # Parameters depending on the model
    if model == 'logistic':
        W =     lambda s:       expit(s)
        dW =    lambda s:        W(s) * (1-W(s))
        nu =    lambda g = g:   np.log(g / (1-g))
        w2 =    lambda s:       W(s) * (1-W(s))
        _s, s_ = 2.3994, - 2.3994 # Constants for the value of d_1 and d_2
        
    if model == 'probit':
        eps = 1e-12 # Minimum value in order to avoid numeric problems
        W  =    lambda s:       np.clip(norm.cdf(s), eps, 1-eps)         # W 
        W_1 =   lambda s:       np.clip(1 - norm.cdf(s), eps, 1-eps)    # 1 - W
        #
        dW =    lambda s:       norm.pdf(s)       
        nu =    lambda g = g:   norm.ppf(g)
        w2 =    lambda s:       dW(s)**2 / (W(s) * W_1(s))
        _s, s_ = 1.575, -1.575# Constants for the value of d_1 and d_2
        
    # # RESTO DE PARAMETROS 
    #
    # Parameters in Z.5
    zeta =      lambda g = g:   dW(nu(g))
    w =         lambda s:       np.sqrt(w2(s))
    #
    # Parameters in X
    F =         lambda x, theta = theta:        W(theta[0] + theta[1] * x)
    dF =        lambda x,theta = theta:         dW(theta[0] + theta[1] * x) * theta[1]
    mu =        lambda g = g, theta = theta:    (nu(g) - theta[0]) / theta[1]
    eta =       lambda g = g, theta = theta:    zeta(g) * theta[1] 
    #
    # Parameters in Z_gamma
    phi =       lambda z,g = g:     z / zeta(g) + nu(g)
    phi_1 =     lambda s, g = g:    (s - nu(g)) * zeta(g)
    H =         lambda z,g = g:     W(phi(z,g))
    H_1 =       lambda t,g = g:     phi_1(nu(t),g)
    dH =        lambda z,g = g:     dW(phi(z,g)) / zeta(g)
    h2 =        lambda z,g = g:     1/ zeta(g)**2 * w2(phi(z,g))
    h =         lambda z,g = g:     np.sqrt(h2(z,g))
    #
    #
    _z = lambda g = g: phi_1(_s, g)
    z_ = lambda g = g: phi_1(s_,g)
    # FALTA DEFINIRLOS SEGUN LOS VALORES QUE SON
    d1_2 = lambda g = g:    (1 / h2(0,g) if _z(g)*z_(g) < 0 else 1 / 1**2 * (z_(g)*h(z_(g), g) + _z(g) * h(_z(g), g))**2 / (h2(z_(g), g) * h2(_z(g), g) * (_z(g) - z_(g))**2))
    d2_2 = lambda g = g:    (h(_z(g), g) + h(z_(g), g))**2 / (h2(_z(g), g) * h2(z_(g), g) * (_z(g) - z_(g))**2)
    sol_dict = {

        'W':W,  'dW': dW,   'nu': nu,    'zeta': zeta,     'w2': w2,   'w': w,           
        'F':F,  'dF': dF,   'mu': mu,   'eta': eta, 
        'H':    H, 'H_inv': H_1,'dH':   dH,'h2':   h2,'h':    h ,
        'd1_2':d1_2, 'd2_2':d2_2
    }
    return sol_dict

def A_optimal_unrestricted(model = 'logistic', g = 0.5):
    #
    # We define the values and functions we need depending on the model and gamma (g).
    params = load_param(g = g,model = model)
    h, nu, zeta, d1_2, d2_2 = [params[el] for el in ['h','nu', 'zeta', 'd1_2', 'd2_2']]
    nu, zeta, d1_2, d2_2 = nu(g), zeta(g), d1_2(g), d2_2(g)
    #
    # We define the functions depending on delta (d) for the optimization problem
    fA =     lambda d:  d1_2 + (nu* zeta - d)** 2 *d2_2 # Auxiliar function defined in Proposition 4
    phiA_d = lambda d: (np.sqrt(fA(-d)) + np.sqrt(fA(d))) / (h(-nu * zeta +d) * d) # The objective function to minimize
    #
    # In order to minimize, we first select an initial optimal value d0
    d = np.arange(1000)/ 100
    d = d[(h(-nu * zeta +d) * d) != 0]
    val = pd.Series(phiA_d(d), d)
    val = val[~np.isnan(val)]
    d0 = val.index[val.argmin()]
    #
    # We optimize it by python optimize
    d_opt = float(op.minimize(phiA_d, d0,options = {'disp': False}).x)
    # We obtain the weight asociated to the smallest dose and build the optimal design
    w_opt = np.sqrt(fA(d_opt)) / (np.sqrt(fA(d_opt)) + np.sqrt(fA(-d_opt)))
    dis_opt = pd.Series([w_opt, 1- w_opt], [- nu * zeta - d_opt, - nu * zeta + d_opt])
    return dis_opt

def A_optimal_restricted(D, model = 'logistic', g = 0.5):
    #
    # We define the values and functions we need depending on the model and gamma (g).
    params = load_param(g = g,model = model)
    h2, h, d1_2, d2_2 = [params[el] for el in [ 'h2', 'h', 'd1_2', 'd2_2']]
    d1_2, d2_2 = d1_2(g), d2_2(g)
    #
    # We define the functions depending on delta (d) for the optimization problem
    g_A = lambda z: h2(z) * (d1_2 + z**2 * d2_2)    # Auxiliar function defined in Proposition 6
    phi_A = lambda z: (np.sqrt(g_A(z)) + np.sqrt(g_A(D))) / (h(z) * np.abs(D-z))
    #
    # In order to minimize, we first select an initial optimal value z0
    z_arr = np.arange(10000)/ 1000 - 10 + D
    val = pd.Series(phi_A(z_arr), z_arr)
    val = val[~np.isnan(val)]
    z_0 = val.index[val.argmin()]
    #
    #We optimize it by python optimize
    z = float(op.minimize(phi_A, z_0, method='L-BFGS-B',bounds=[(None, D)],options={'disp': False}).x)
    if z > D: z = z_0
    # We obtain the weight asociated to the smallest dose and build the optimal design
    w = np.sqrt(g_A(D)) / (np.sqrt(g_A(z)) + np.sqrt(g_A(D)))
    dis_opt =  pd.Series([w, 1-w], [z, D])
    return dis_opt

def A_optimal(model = 'logistic', g = 0.5, D = np.inf):
    # We first compute the unrestricted optimal design
    d_opt_0 = A_optimal_unrestricted(model = model, g = g)
    if D > max(d_opt_0.index):  return d_opt_0                                      # Case (i) in section 4.3
    else:                       return A_optimal_restricted(D,model = model, g = g) # Case (ii) in section 4.3
    
def eff_mu(dis, model = 'logistic', g = 0.5):
    if len(dis) == 1 and dis.index[0] == 0: return 1
    params = load_param(g = g,model = model)
    h2, d1_2 = [params[el] for el in [ 'h2', 'd1_2']]
    d1_2 = d1_2(g)
    t = sum(h2(dis.index) * dis)
    u = sum(- h2(dis.index) * dis * dis.index)  
    v = sum(h2(dis.index) * dis * dis.index ** 2) 
    if v == 0: return 0  
    return d1_2 * (t* v - u**2) / v

def eff_eta(dis, model = 'logistic', g = 0.5):
    params = load_param(g = g,model = model)
    h2, d1_2 = [params[el] for el in [ 'h2', 'd1_2']]
    d1_2 = d1_2(g)
    t = sum(h2(dis.index) * dis)
    u = sum(- h2(dis.index) * dis * dis.index)  
    v = sum(h2(dis.index) * dis * dis.index ** 2) 
    if t == 0: return 0  
    return d1_2 * (t* v - u**2) / t
    
def cc_lambda_unrestricted(l,model = 'logistic', g = 0.5):
    #
    # We define the values and functions we need depending on the model and gamma (g).
    params = load_param(g = g,model = model)
    h, nu, zeta, d1_2, d2_2 = [params[el] for el in [ 'h', 'nu', 'zeta', 'd1_2', 'd2_2']]
    nu, zeta, d1_2, d2_2 = nu(g), zeta(g), d1_2(g), d2_2(g)
    #
    # We define the functions depending on delta (d) for the optimization problem
    fcc_l =     lambda d: (1-l) * d1_2 + (nu* zeta - d)** 2* l *d2_2 # Auxiliar function defined in Proposition 4
    phicc_l = lambda d: (np.sqrt(fcc_l(-d)) + np.sqrt(fcc_l(d))) / (h(-nu * zeta +d) * d) # The objective function to minimize
    #
    # In order to minimize, we first select an initial optimal value d0
    d = np.arange(1000)/ 100
    d = d[(h(-nu * zeta +d) * d) != 0]
    val = pd.Series(phicc_l(d), d)
    val = val[~np.isnan(val)]
    d0 = val.index[val.argmin()]
    #
    # We optimize it by python optimize
    res = op.minimize(
        phicc_l, 
        d0, 
        method='L-BFGS-B', 
        bounds=[(1e-6, None)], 
        options={'disp': False}
    )
    d_opt = float(res.x[0])
    # We obtain the weight asociated to the smallest dose and build the optimal design
    w_opt = np.sqrt(fcc_l(d_opt)) / (np.sqrt(fcc_l(d_opt)) + np.sqrt(fcc_l(-d_opt)))
    dis_opt = pd.Series([w_opt, 1- w_opt], [- nu * zeta - d_opt, - nu * zeta + d_opt])
    return dis_opt

def cc_optimal_unrestricted(model = 'logistic', g = 0.5, tol = 0.8):
    l_list = (np.arange(1,1000)/1000)[::-1]
    eff = 1; i = 0; dis = None
    while eff >= tol and i < len(l_list):
        dis0 = dis
        dis = cc_lambda_unrestricted(l_list[i],model = model, g = g)
        eff = eff_mu(dis, model = model, g = g)
        i = i + 1
    if eff_mu(dis0, model = model, g = g) < tol: print('Error: Unaable to find a design meeting the tolerance level '+ str(tol) +' for mu')
    return dis0

def cc_lambda_restricted(l, D,model = 'logistic', g = 0.5):
   #
    # We define the values and functions we need depending on the model and gamma (g).
    params = load_param(g = g,model = model)
    h2, h, d1_2, d2_2 = [params[el] for el in ['h2', 'h', 'd1_2', 'd2_2' ]]
    d1_2, d2_2 = d1_2(g), d2_2(g)
    #
    # We define the functions depending on delta (d) for the optimization problem
    gcc_l = lambda z: h2(z) * ((1-l) * d1_2 + l*  z**2 * d2_2)    # Auxiliar function defined in Proposition 6
    phicc_l = lambda z: (np.sqrt(gcc_l(z)) + np.sqrt(gcc_l(D))) / (h(z) * np.abs(D-z))
    #
    # In order to minimize, we first select an initial optimal value z0
    z_arr = np.arange(10000)/ 1000 - 10 + D
    val = pd.Series(phicc_l(z_arr), z_arr)
    val = val[~np.isnan(val)]
    z_0 = val.index[val.argmin()]
    #
    #We optimize it by python optimize
    z = float(op.minimize(phicc_l, z_0, method='L-BFGS-B',bounds=[(None, D)],options={'disp': False}).x)
    if z > D: z = z_0
    # We obtain the weight asociated to the smallest dose and build the optimal design
    w = np.sqrt(gcc_l(D)) / (np.sqrt(gcc_l(z)) + np.sqrt(gcc_l(D)))
    dis_opt =  pd.Series([w, 1-w], [z, D])
    return dis_opt


def cc_optimal(model = 'logistic', g = 0.5, D = np.inf, tol = 0.8):
    d_opt_0 = cc_optimal_unrestricted(model = model, g = g, tol = tol)
    if D > max(d_opt_0.index): return d_opt_0
    else:
        l_list = (np.arange(1000)/1000)[::-1]
        eff = 1; i = 0; dis = None
        while eff >= tol and i < len(l_list):
            dis0 = dis
            dis = cc_lambda_restricted(l_list[i],D,model = model, g = g)
            eff = eff_mu(dis, model = model, g = g)
            i = i + 1
        if type(dis0) == type(None): print('Error: Unable to find a design meeting the tolerance level '+ str(tol) +' for mu, the maximum efficienci you can get with that restriction is ' + str(round(eff ,3)))
        return dis0