import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import expit
from typing import Dict, Callable, Any, Optional, Tuple, Union

def load_param(
    model: str = 'logistic', 
    g: Optional[float] = None, 
    theta: Optional[Tuple[float, float]] = None, 
    m: float = 0.5
) -> Dict[str, Any]:
    """
    Loads mathematical parameters and functions for a specific statistical model.

    Args:
        model (str): The name of the statistical model (e.g., 'logistic', 'probit').
        g (float, optional): Base parameter for generating model functions.
        theta (Tuple[float, float], optional): Coefficients [intercept, slope] for F and dF.
        m (float): Skewness parameter, primarily used in the 'skewed logit' model.

    Returns:
        Dict[str, Any]: A dictionary containing the standard functions and bounds 
        (_s, s_) for the selected model.
        
    Raises:
        ValueError: If the provided model string is not supported.
    """
    
    # Initialization of bounds to avoid UnboundLocalError
    _s, s_ = 0.0, 0.0

    if model == 'logistic':
        W = lambda s: expit(s)
        dW = lambda s: W(s) * (1 - W(s))
        nu = lambda g=g: np.log(g / (1 - g))
        w2 = lambda s: W(s) * (1 - W(s))
        _s, s_ = 2.3994, -2.3994
        
    elif model == 'probit':
        eps = 1e-12 
        W = lambda s: np.clip(norm.cdf(s), eps, 1 - eps)
        W_1 = lambda s: np.clip(1 - norm.cdf(s), eps, 1 - eps)
        dW = lambda s: norm.pdf(s)       
        nu = lambda g=g: norm.ppf(g)
        w2 = lambda s: dW(s)**2 / (W(s) * W_1(s))
        _s, s_ = 1.575, -1.575

    elif model == 'gumbel':
        W = lambda s: np.exp(-np.exp(-s))
        dW = lambda s: np.exp(-s) * W(s)
        nu = lambda g=g: -np.log(-np.log(g))
        w2 = lambda s: np.where(
            (s < -6.5) | (s > 350), 
            0.0, 
            1.0 / (np.exp(2 * np.clip(s, -6.5, 350)) * np.expm1(np.exp(-np.clip(s, -6.5, 350))))
        )
        _s, s_ = 2.0731078099, -1.2686195509
        
    elif model == 'double exponential':
        def W_func(s: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
            if isinstance(s, pd.Series):
                return s.apply(lambda x: 0.5 * np.exp(x) if x <= 0 else 1 - 0.5 * np.exp(-x))
            elif isinstance(s, (np.ndarray, list)):
                return np.array([0.5 * np.exp(x) if x <= 0 else 1 - 0.5 * np.exp(-x) for x in s])
            return 0.5 * np.exp(s) if s <= 0 else 1 - 0.5 * np.exp(-s)
            
        W = W_func
        dW = lambda s: 0.5 * np.exp(-np.abs(s))
        nu = lambda g=g: np.log(2 * g) if g <= 0.5 else -np.log(2 * (1 - g))
        w2 = lambda s: 1 / (2 * np.exp(np.abs(s)) - 1)
        _s, s_ = 1.8414, -1.8414
        
    elif model == 'double reciprocal':
        def W_func(s: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
            if isinstance(s, pd.Series):
                return s.apply(lambda x: 1 / (2 * (1 - x)) if x <= 0 else 1 - 1 / (2 * (1 + x)))
            elif isinstance(s, (np.ndarray, list)):
                return np.array([1 / (2 * (1 - x)) if x <= 0 else 1 - 1 / (2 * (1 + x)) for x in s])
            return 1 / (2 * (1 - s)) if s <= 0 else 1 - 1 / (2 * (1 + s))
            
        W = W_func
        dW = lambda s: 1 / (2 * (1 + np.abs(s))**2)
        nu = lambda g=g: 1 - 1 / (2 * g) if g <= 0.5 else 1 / (2 * (1 - g)) - 1
        w2 = lambda s: 1 / ((1 + np.abs(s))**2 * (2 * (1 + np.abs(s)) - 1))
        _s, s_ = 1.6180, -1.6180
        
    elif model == 'skewed logit':
        logit = lambda s: 1 / (1 + np.exp(-s))
        W = lambda s: logit(s)**m
        dW = lambda s: m * W(s) * logit(-s)
        w2 = lambda s: m**2 * W(s) / logit(-s)**(m - 2)
        nu = lambda g=g: -np.log(g**(-1 / m) - 1)
        # Asegúrate de que compute_s esté definida en el mismo archivo antes de llamar a esto
        _s, s_ = compute_s(lambda s: np.sqrt(w2(s)), 2)
            
    elif model == 'complementary log-log':
        W = lambda s: 1 - np.exp(-np.exp(s))
        dW = lambda s: np.exp(s) * (1 - W(s))
        nu = lambda g=g: np.log(-np.log(1 - g))
        w2 = lambda s: np.where(
            s > 6.5, 
            0.0, 
            np.exp(2 * np.minimum(s, 6.5)) / np.expm1(np.exp(np.minimum(s, 6.5)))
        )
        _s, s_ = 1.2686, -2.0731     
        
    else:
        raise ValueError(f"Unknown model: '{model}'. Please select a valid model.")

    # --- Derived Parameters ---
    
    # Parameters in Z.5
    zeta = lambda g=g: dW(nu(g))
    w = lambda s: np.sqrt(w2(s))
    
    # Parameters in X
    F = lambda x, theta=theta: W(theta[0] + theta[1] * x)
    dF = lambda x, theta=theta: dW(theta[0] + theta[1] * x) * theta[1]
    mu = lambda g=g, theta=theta: (nu(g) - theta[0]) / theta[1]
    eta = lambda g=g, theta=theta: zeta(g) * theta[1] 
    
    # Parameters in Z_gamma
    phi = lambda z, g=g: z / zeta(g) + nu(g)
    phi_1 = lambda s, g=g: (s - nu(g)) * zeta(g)
    H = lambda z, g=g: W(phi(z, g))
    H_1 = lambda t, g=g: phi_1(nu(t), g)
    dH = lambda z, g=g: dW(phi(z, g)) / zeta(g)
    h2 = lambda z, g=g: 1 / zeta(g)**2 * w2(phi(z, g))
    h = lambda z, g=g: np.sqrt(h2(z, g))
    
    # Boundary evaluations
    _z = lambda g=g: phi_1(_s, g)
    z_ = lambda g=g: phi_1(s_, g)
    
    # Complex efficiency functions
    r1_2 = lambda g=g: (
        1 / h2(0, g) if _z(g) * z_(g) < 0 else 
        1 / 1**2 * (z_(g) * h(z_(g), g) + _z(g) * h(_z(g), g))**2 / 
        (h2(z_(g), g) * h2(_z(g), g) * (_z(g) - z_(g))**2)
    )
    r2_2 = lambda g=g: (h(_z(g), g) + h(z_(g), g))**2 / (h2(_z(g), g) * h2(z_(g), g) * (_z(g) - z_(g))**2)
    
    return {
        'W': W, 'dW': dW, 'nu': nu, 'zeta': zeta, 'w2': w2, 'w': w, 's_': s_, '_s': _s,   
        'F': F, 'dF': dF, 'mu': mu, 'eta': eta, 
        'H': H, 'H_inv': H_1, 'dH': dH, 'h2': h2, 'h': h,
        'r1_2': r1_2, 'r2_2': r2_2
    }




def compute_s_aux(w: Callable[[np.ndarray], np.ndarray], s: np.ndarray) -> Tuple[float, float]:
    """
    Computes the auxiliary boundary points based on the weight function w.
    """
    x = w(s)
    y_vals = s * x
    
    # Máscara booleana para filtrar en lugar de indexación múltiple
    valid_mask = (x > 0.001) & (x < np.inf)
    vx = x[valid_mask]
    vy = y_vals[valid_mask]
    
    # Vectorización pura: evita list() a toda costa en cálculos numéricos
    cx = np.concatenate([x, -x])
    cy = np.concatenate([y_vals, -y_vals])
    
    # zip() hace que iterar sobre dos arrays simultáneamente sea más limpio
    for xz, yz in zip(vx[:-1], vy[:-1]):
        for xz1, yz1 in zip(-vx, -vy):
            
            # Cálculo de la interpolación / cota
            t = (cx - xz1) / (xz - xz1)
            y_bound = t * yz + (1 - t) * yz1
            
            # np.any() es considerablemente más rápido que sum() == 0
            if not np.any(cy > y_bound):
                return yz / xz, yz1 / xz1
                
    # Buena práctica: Siempre levanta un error si una función se queda sin caminos de retorno
    raise ValueError("No valid support points found in compute_s_aux")


def compute_s(w: Callable, r: int) -> Tuple[float, float]:
    """
    Iteratively refines the boundary points _s and s_ up to r decimal places.
    """
    initial_s = np.arange(100.0) - 50.0
    _s, s_ = compute_s_aux(w, initial_s)
    
    for i in range(1, r + 1):
        aux = (np.arange(11.0) - 5.0) / (10 ** i)
        s_refined = np.concatenate([_s + aux, s_ + aux])
        _s, s_ = compute_s_aux(w, s_refined)
        
    return float(np.round(_s, r)), float(np.round(s_, r))


def eff_mu(dis: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    """
    Calculates the efficiency for mu.
    """
    if len(dis) == 1 and dis.index[0] == 0: 
        return 1.0
        
    params = load_param(g=g, model=model, m=m)
    h2_func = params['h2']
    r1_2 = params['r1_2']()
    
    # Extraemos arrays de numpy para acelerar las operaciones matemáticas
    z = dis.index.to_numpy()
    weights = dis.to_numpy()
    
    # CACHÉ DE LLAMADA: Computamos h2(z) una única vez en lugar de tres
    h2_vals = h2_func(z)
    
    t = np.sum(h2_vals * weights)
    u = np.sum(-h2_vals * weights * z)  
    v = np.sum(h2_vals * weights * (z ** 2)) 
    
    # np.isclose protege contra errores de coma flotante (ej. 0.0000000000000001)
    if np.isclose(v, 0.0): 
        return 0.0  
        
    return r1_2 * (t * v - u**2) / v


def eff_eta(dis: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    """
    Calculates the efficiency for eta.
    """
    params = load_param(g=g, model=model, m=m)
    h2_func = params['h2']
    
    # Pasamos g en caso de que load_param retorne una lambda dependiente de g
    r2_2 = params['r2_2'](g)
    
    z = dis.index.to_numpy()
    weights = dis.to_numpy()
    
    h2_vals = h2_func(z)
    
    t = np.sum(h2_vals * weights)
    u = np.sum(-h2_vals * weights * z)  
    v = np.sum(h2_vals * weights * (z ** 2)) 
    
    if np.isclose(t, 0.0): 
        return 0.0  
        
    return r2_2 * (t * v - u**2) / t