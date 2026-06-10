import numpy as np
import pandas as pd
import scipy.optimize as op
import os
from typing import Optional, Tuple, Union, Any

# Assuming these functions are correctly exposed in reparameterization.py
from reparameterization import load_param, eff_mu, eff_eta

# =============================================================================
# 1. UNRESTRICTED DESIGNS
# =============================================================================

def D_optimal_unrestricted_case_I(model: str = 'logistic', g: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model)
    h_func = params['h']
    nu_val = params['nu'](g)
    zeta_val = params['zeta'](g)
    
    def phiD_delta(delta_val: float) -> float:
        return - (h_func(-nu_val * zeta_val + delta_val)**2 * delta_val)
    
    delta_grid = np.arange(1.0, 1000.0) / 100.0
    val = pd.Series([phiD_delta(d) for d in delta_grid], index=delta_grid)
    delta0 = val.idxmin()
    
    result = op.minimize(phiD_delta, delta0, options={'disp': False})
    opt_delta = np.abs(float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x)
    
    opt_design = pd.Series([0.5, 0.5], index=[-nu_val * zeta_val - opt_delta, -nu_val * zeta_val + opt_delta])
    return opt_design.sort_index()

def D_optimal_unrestricted_case_II(model: str = 'skewed logit', m: float = 0.5, g: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model, m=m)
    h_func = params['h']
    
    def phiD_z(z_input: np.ndarray) -> float:
        z1, z2 = z_input
        return -np.abs(z1 - z2) * h_func(z1) * h_func(z2)

    z_list = np.arange(100.0) / 10.0 - 5.0
    tuple_list = [(z1, z2) for z1 in z_list for z2 in z_list if z1 != z2]
    z0 = tuple_list[np.argmin([phiD_z(t) for t in tuple_list])]
    
    result = op.minimize(phiD_z, z0, options={'disp': False})
    opt_z1, opt_z2 = result.x
    
    opt_design = pd.Series([0.5, 0.5], index=[opt_z1, opt_z2])
    return opt_design.sort_index()

def D_optimal_unrestricted(model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    if model in ['logistic', 'probit']: 
        return D_optimal_unrestricted_case_I(model=model, g=g)
    elif model in ['skewed logit', 'complementary log-log', 'gumbel']: 
        return D_optimal_unrestricted_case_II(model=model, g=g, m=m)
    raise ValueError(f"Error: Model '{model}' not supported.")

def A_optimal_unrestricted_case_I(model: str = 'logistic', g: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model)
    h_func = params['h']
    nu_val, zeta_val = params['nu'](g), params['zeta'](g)
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)
    
    def fA_func(delta_val: float) -> float:
        return r1_sq + (nu_val * zeta_val - delta_val)**2 * r2_sq
        
    def phiA_delta(delta_val: float) -> float:
        numerator = np.sqrt(fA_func(-delta_val)) + np.sqrt(fA_func(delta_val))
        denominator = h_func(-nu_val * zeta_val + delta_val) * delta_val
        return numerator / denominator if denominator != 0.0 else np.inf
    
    delta_grid = np.arange(0.0, 10.0, 0.01)
    delta_grid = delta_grid[(h_func(-nu_val * zeta_val + delta_grid) * delta_grid) != 0.0]
    val = pd.Series([phiA_delta(d) for d in delta_grid], index=delta_grid).dropna()
    delta0 = val.idxmin()
    
    result = op.minimize(phiA_delta, delta0, options={'disp': False})
    opt_delta = float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x
    
    opt_weight = np.sqrt(fA_func(opt_delta)) / (np.sqrt(fA_func(-opt_delta)) + np.sqrt(fA_func(opt_delta)))
    opt_design = pd.Series([opt_weight, 1.0 - opt_weight], index=[-nu_val * zeta_val - opt_delta, -nu_val * zeta_val + opt_delta])
    return opt_design.sort_index()

def A_optimal_unrestricted_case_II(model: str = 'skewed logit', m: float = 0.5, g: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model, m=m)
    h2_func, h_func = params['h2'], params['h']
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)
    
    def fA_func(z_val: float) -> float:
        return h2_func(z_val) * (r1_sq + z_val**2 * r2_sq)
    
    def phiA_z(z_input: np.ndarray) -> float:
        z1, z2 = z_input
        if z1 == z2: return 1e12 
        
        fA1, fA2 = max(fA_func(z1), 0.0), max(fA_func(z2), 0.0)
        denominator = np.abs(z1 - z2) * h_func(z1) * h_func(z2)
        if denominator < 1e-12: return 1e12
        return (np.sqrt(fA1) + np.sqrt(fA2)) / denominator
    
    z_list = np.arange(100.0) / 10.0 - 5.0
    tuple_list = [(z1, z2) for z1 in z_list for z2 in z_list if z1 != z2]
    z0 = tuple_list[np.argmin([phiA_z(t) for t in tuple_list])]
    
    result = op.minimize(phiA_z, z0, bounds=[(-10.0, 10.0), (-10.0, 10.0)], options={'disp': False})
    opt_z1, opt_z2 = result.x
    
    opt_weight = np.sqrt(max(fA_func(opt_z2), 0.0)) / (np.sqrt(max(fA_func(opt_z1), 0.0)) + np.sqrt(max(fA_func(opt_z2), 0.0)))        
    opt_design = pd.Series([opt_weight, 1.0 - opt_weight], index=[opt_z1, opt_z2])
    return opt_design.sort_index()

def A_optimal_unrestricted(model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    if model in ['logistic', 'probit']: 
        return A_optimal_unrestricted_case_I(model=model, g=g)
    elif model in ['skewed logit', 'complementary log-log', 'gumbel']: 
        return A_optimal_unrestricted_case_II(model=model, g=g, m=m)
    raise ValueError(f"Error: Model '{model}' not supported.")

def cc_lambda_unrestricted(lambda_val: float, model: str = 'logistic', g: float = 0.5, m: float = 0.5, x0: Optional[Any] = None, return_x0: bool = False) -> Any:
    params = load_param(g=g, model=model, m=m)
    H_inv, h2_func, h_func = params['H_inv'], params['h2'], params['h']
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)

    if np.isnan(r1_sq) or np.isnan(r2_sq) or np.isinf(r1_sq) or np.isinf(r2_sq):
        return (None, None) if return_x0 else None

    def fcc_l_func(z_val: float) -> float:
        return ((1.0 - lambda_val) * r1_sq + lambda_val * r2_sq * z_val**2) * h2_func(z_val)

    if model in ['logistic', 'probit']:
        nu_val, zeta_val = params['nu'](g), params['zeta'](g)
        delta0 = x0 if x0 is not None else 1.0

        def phi_delta(delta_input: Union[float, np.ndarray]) -> float:
            delta_val = float(delta_input[0]) if isinstance(delta_input, np.ndarray) else float(delta_input)
            z1, z2 = -nu_val * zeta_val - delta_val, -nu_val * zeta_val + delta_val
            denominator = h2_func(z2) * delta_val
            if denominator == 0.0: return np.inf
            return (np.sqrt(fcc_l_func(z1)) + np.sqrt(fcc_l_func(z2))) / denominator

        result = op.minimize(phi_delta, delta0, bounds=[(0.01, None)], options={'disp': False})
        opt_delta = float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x
        opt_z1, opt_z2 = -nu_val * zeta_val - opt_delta, -nu_val * zeta_val + opt_delta
        
        weight_denominator = np.sqrt(fcc_l_func(opt_z1)) + np.sqrt(fcc_l_func(opt_z2))
        opt_weight = np.sqrt(fcc_l_func(opt_z2)) / weight_denominator if weight_denominator != 0.0 else 0.5

        opt_design = pd.Series([opt_weight, 1.0 - opt_weight], index=[opt_z1, opt_z2]).sort_index()
        return (opt_design, opt_delta) if return_x0 else opt_design

    elif model in ['skewed logit', 'complementary log-log', 'gumbel']:
        z0 = x0 if x0 is not None else np.array([H_inv(0.25), H_inv(0.75)])

        def phi_opt(z_input: np.ndarray) -> float:
            z1, z2 = z_input
            if z1 == z2: return 1e12 
            
            f1, f2 = max(fcc_l_func(z1), 0.0), max(fcc_l_func(z2), 0.0)
            denominator = np.abs(z1 - z2) * h_func(z1) * h_func(z2)
            if denominator < 1e-12: return 1e12
            return (np.sqrt(f1) + np.sqrt(f2)) / denominator

        result = op.minimize(phi_opt, z0, bounds=[(-10.0, 10.0), (-10.0, 10.0)], options={'disp': False})
        opt_z1, opt_z2 = result.x

        weight_denominator = np.sqrt(max(fcc_l_func(opt_z1), 0.0)) + np.sqrt(max(fcc_l_func(opt_z2), 0.0))
        opt_weight = np.sqrt(max(fcc_l_func(opt_z2), 0.0)) / weight_denominator if weight_denominator != 0.0 else 0.5

        opt_design = pd.Series([opt_weight, 1.0 - opt_weight], index=[opt_z1, opt_z2]).sort_index()
        return (opt_design, result.x) if return_x0 else opt_design
        
    raise ValueError(f"Error: Model '{model}' not supported.")

def E_optimal_unrestricted_case_I(model: str = 'logistic', g: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model)
    h_func = params['h']
    nu_val, zeta_val = params['nu'](g), params['zeta'](g)
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)
    
    def phiE_1_delta(delta_val: float) -> float:
        return - (h_func(-nu_val * zeta_val + delta_val) * delta_val)
    
    delta_grid = np.arange(1.0, 100.0) / 10.0
    val1 = pd.Series([phiE_1_delta(d) for d in delta_grid], index=delta_grid)
    delta0_1 = val1.idxmin()
    
    result = op.minimize(phiE_1_delta, delta0_1, bounds=[(1e-4, None)], options={'disp': False})
    opt_delta_1 = float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x
    opt_weight_1 = 0.5 * (1.0 - (nu_val * zeta_val * r2_sq * opt_delta_1) / (r1_sq + r2_sq * nu_val**2 * zeta_val**2))
    
    opt_delta_2 = np.sqrt(r1_sq / r2_sq + nu_val**2 * zeta_val**2)
    opt_weight_2 = 0.5 - (nu_val * zeta_val) / (2.0 * opt_delta_2)
    
    def T_func(d: float, w: float) -> float:
        return r1_sq + r2_sq * (nu_val**2 * zeta_val**2 + d**2 + 2.0 * nu_val * zeta_val * d * (2.0 * w - 1.0))
        
    def D_func(d: float, w: float) -> float:
        return 4.0 * r1_sq * r2_sq * d**2 * w * (1.0 - w)
    
    def lambda_min(d: float, w: float) -> float:
        return (T_func(d, w) - np.sqrt(max(T_func(d, w)**2 - 4.0 * D_func(d, w), 0.0))) / 2.0
    
    if np.round(lambda_min(opt_delta_1, opt_weight_1), 5) > np.round(lambda_min(opt_delta_2, opt_weight_2), 5) and (0.0 <= opt_weight_1 <= 1.0):
        best_delta, best_weight = opt_delta_1, opt_weight_1
    else:
        best_delta, best_weight = opt_delta_2, opt_weight_2

    opt_design = pd.Series([best_weight, 1.0 - best_weight], index=[-nu_val * zeta_val - best_delta, -nu_val * zeta_val + best_delta])
    return opt_design.sort_index()

def E_optimal_unrestricted_case_II(model: str = 'skewed logit', m: float = 0.5, g: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model, m=m)
    h2_func, h_func = params['h2'], params['h']
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)
    
    def f_func(z_val: float) -> float: return (r1_sq + r2_sq * z_val**2) * h2_func(z_val)
    def g_val_func(z1: float, z2: float) -> float: return (r1_sq + r2_sq * z1 * z2) * h_func(z1) * h_func(z2)

    def phiE_1_z(z_input: np.ndarray) -> float:
        z1, z2 = z_input
        if z1 == z2: return np.inf
        numerator = h2_func(z1) * h2_func(z2) * r1_sq * r2_sq * (z1 - z2)**2
        denominator = f_func(z1) + f_func(z2) + 2.0 * np.abs(g_val_func(z1, z2))
        return -numerator / denominator if denominator != 0.0 else np.inf

    z_grid = np.linspace(-5.0, 5.0, 50)
    Z1, Z2 = np.meshgrid(z_grid, z_grid)
    mask = Z1 != Z2
    Z1_flat, Z2_flat = Z1[mask], Z2[mask]
    
    phi_vals = np.array([phiE_1_z((z1, z2)) for z1, z2 in zip(Z1_flat, Z2_flat)])
    best_index = np.argmin(phi_vals)
    z0_1 = np.array([Z1_flat[best_index], Z2_flat[best_index]])
    
    res1 = op.minimize(phiE_1_z, z0_1, options={'disp': False})
    opt_z1_1, opt_z2_1 = res1.x
    
    if f_func(opt_z1_1) > f_func(opt_z2_1):
        opt_z1_1, opt_z2_1 = opt_z2_1, opt_z1_1
        
    g_abs = np.abs(g_val_func(opt_z1_1, opt_z2_1))
    weight_denominator_1 = f_func(opt_z1_1) + f_func(opt_z2_1) + 2.0 * g_abs
    opt_weight_1 = (f_func(opt_z2_1) + g_abs) / weight_denominator_1 if weight_denominator_1 != 0.0 else 0.5
    lambda_1 = -res1.fun

    def phiE_2_z1(z1_input: Union[float, np.ndarray]) -> float:
        z1 = float(z1_input[0]) if isinstance(z1_input, np.ndarray) else float(z1_input)
        if z1 == 0.0: return np.inf
        z2 = -r1_sq / (r2_sq * z1)
        denominator = f_func(z1) + f_func(z2)
        return -(f_func(z1) * f_func(z2)) / denominator if denominator != 0.0 else np.inf

    z_grid_2 = np.linspace(-5.0, 5.0, 100)
    z_grid_2 = z_grid_2[z_grid_2 != 0.0]
    z0_2 = z_grid_2[np.argmin([phiE_2_z1(z) for z in z_grid_2])]
    
    res2 = op.minimize(phiE_2_z1, z0_2, options={'disp': False})
    opt_z1_2 = float(res2.x[0]) if isinstance(res2.x, np.ndarray) else res2.x
    opt_z2_2 = -r1_sq / (r2_sq * opt_z1_2)
    
    weight_denominator_2 = opt_z2_2 * h2_func(opt_z2_2) - opt_z1_2 * h2_func(opt_z1_2)
    opt_weight_2 = (opt_z2_2 * h2_func(opt_z2_2)) / weight_denominator_2 if weight_denominator_2 != 0.0 else 0.5
    lambda_2 = -res2.fun

    if lambda_1 >= lambda_2 and (0.0 <= opt_weight_1 <= 1.0):
        best_z1, best_z2, best_weight = opt_z1_1, opt_z2_1, opt_weight_1
    else:
        best_z1, best_z2, best_weight = opt_z1_2, opt_z2_2, opt_weight_2

    opt_design = pd.Series([best_weight, 1.0 - best_weight], index=[best_z1, best_z2])
    return opt_design.sort_index()

def E_optimal_unrestricted(model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    if model in ['logistic', 'probit']: 
        return E_optimal_unrestricted_case_I(model=model, g=g)
    elif model in ['skewed logit', 'complementary log-log', 'gumbel']: 
        return E_optimal_unrestricted_case_II(model=model, g=g, m=m)
    raise ValueError(f"Error: Model '{model}' not supported.")


# =============================================================================
# 2. RESTRICTED DESIGNS (z_2 = delta)
# =============================================================================

def D_optimal_delta(delta: float, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model, m=m)
    h_func = params['h']
    
    def phiD_z(z_input: Union[float, np.ndarray]) -> float:
        z_val = float(z_input[0]) if isinstance(z_input, np.ndarray) else float(z_input)
        return - (h_func(z_val)**2 * (z_val - delta)**2)
    
    z_grid = np.arange(-50.0, int(delta * 10)) / 10.0
    z_grid = z_grid[z_grid < delta]
    if len(z_grid) == 0: z_grid = np.array([delta - 1.0])
    
    z0 = z_grid[np.argmin([phiD_z(z) for z in z_grid])]
    
    result = op.minimize(phiD_z, z0, bounds=[(None, delta - 1e-4)], options={'disp': False})
    opt_z = float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x
    
    opt_design = pd.Series([0.5, 0.5], index=[opt_z, delta])
    return opt_design.sort_index()

def A_optimal_delta(delta: float, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model, m=m)
    h_func = params['h']
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)
    
    def fA_func(z_val: float) -> float:
        return h_func(z_val)**2 * (r1_sq + r2_sq * z_val**2)
    
    def phiA_z(z_input: Union[float, np.ndarray]) -> float:
        z_val = float(z_input[0]) if isinstance(z_input, np.ndarray) else float(z_input)
        denominator = np.abs(z_val - delta) * h_func(z_val) * h_func(delta)
        if denominator < 1e-12: return 1e12
        return (np.sqrt(max(fA_func(z_val), 0.0)) + np.sqrt(max(fA_func(delta), 0.0))) / denominator
        
    z_grid = np.linspace(-10.0, delta - 0.1, 100)
    z0 = z_grid[np.argmin([phiA_z(z) for z in z_grid])]
    
    result = op.minimize(phiA_z, z0, bounds=[(None, delta - 1e-4)], options={'disp': False})
    opt_z = float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x
    
    fA_delta_safe, fA_zopt_safe = max(fA_func(delta), 0.0), max(fA_func(opt_z), 0.0)
    opt_weight = np.sqrt(fA_delta_safe) / (np.sqrt(fA_zopt_safe) + np.sqrt(fA_delta_safe))
        
    opt_design = pd.Series([opt_weight, 1.0 - opt_weight], index=[opt_z, delta])
    return opt_design.sort_index()

def cc_lambda_delta(lambda_val: float, delta: float, model: str = 'logistic', g: float = 0.5, m: float = 0.5, x0: Optional[float] = None, return_x0: bool = False) -> Any: 
    params = load_param(g=g, model=model, m=m)
    H_func, H_inv_func, h_func = params['H'], params['H_inv'], params['h']
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)

    if np.isnan(r1_sq) or np.isnan(r2_sq) or np.isinf(r1_sq) or np.isinf(r2_sq):
        return (None, None) if return_x0 else None

    def fcc_l_func(z_val: float) -> float:
        return h_func(z_val)**2 * ((1 - lambda_val) * r1_sq + lambda_val * r2_sq * z_val**2)

    def phi_z(z_input: Union[float, np.ndarray]) -> float:
        z_val = float(z_input[0]) if isinstance(z_input, np.ndarray) else float(z_input)
        if z_val == delta: return np.inf
        denominator = np.abs(z_val - delta) * h_func(z_val) * h_func(delta)
        return (np.sqrt(fcc_l_func(z_val)) + np.sqrt(fcc_l_func(delta))) / denominator if denominator != 0.0 else np.inf

    z0 = x0 if x0 is not None else H_inv_func(H_func(delta) / 2.0)
    result = op.minimize(phi_z, z0, bounds=[(None, delta - 1e-4)], options={'disp': False})
    opt_z = float(result.x[0]) if isinstance(result.x, np.ndarray) else result.x

    opt_weight = np.sqrt(fcc_l_func(delta)) / (np.sqrt(fcc_l_func(opt_z)) + np.sqrt(fcc_l_func(delta)))
    opt_design = pd.Series([opt_weight, 1.0 - opt_weight], index=[opt_z, delta]).sort_index()
    return (opt_design, opt_z) if return_x0 else opt_design

def E_optimal_delta(delta: float, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    params = load_param(g=g, model=model, m=m)
    h_func = params['h']
    r1_sq, r2_sq = params['r1_2'](g), params['r2_2'](g)
    
    def f_safe(z_val: float) -> float:
        val = (r1_sq + r2_sq * z_val**2) * h_func(z_val)**2
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)

    def g_safe(z_val: float) -> float:
        val = (r1_sq + r2_sq * z_val * delta) * h_func(z_val) * h_func(delta)
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)
        
    def term_zh2_safe(z_val: float) -> float:
        val = z_val * h_func(z_val)**2
        return 0.0 if np.isnan(val) or np.isinf(val) else float(val)
    
    def phiE_1(z_input: Union[float, np.ndarray]) -> float:
        z_val = float(z_input[0]) if isinstance(z_input, np.ndarray) else float(z_input)
        denominator = f_safe(z_val) + f_safe(delta) + 2.0 * np.abs(g_safe(z_val))
        if denominator < 1e-12: return 1e12
        numerator = h_func(z_val)**2 * h_func(delta)**2 * r1_sq * r2_sq * (z_val - delta)**2
        return -(0.0 if np.isnan(numerator) else float(numerator)) / denominator
        
    z_grid = np.linspace(-10.0, delta - 0.1, 100)
    z0_1 = z_grid[np.argmin([phiE_1(z) for z in z_grid])]
    res1 = op.minimize(phiE_1, z0_1, bounds=[(None, delta - 1e-4)], options={'disp': False})
    opt_z_1 = float(res1.x[0]) if isinstance(res1.x, np.ndarray) else res1.x
    
    weight_denominator_1 = f_safe(opt_z_1) + f_safe(delta) + 2.0 * np.abs(g_safe(opt_z_1))
    opt_weight_1 = (f_safe(delta) + np.abs(g_safe(opt_z_1))) / weight_denominator_1 if weight_denominator_1 != 0.0 else 0.5
    lambda_1 = -res1.fun if res1.success else -np.inf
    
    z_denominator = r2_sq * delta
    if z_denominator == 0.0:
        opt_z_2, opt_weight_2, lambda_2 = None, None, -np.inf
    else:
        opt_z_2 = - r1_sq / z_denominator
        if opt_z_2 <= delta: 
            weight_denominator_2 = term_zh2_safe(delta) - term_zh2_safe(opt_z_2)
            opt_weight_2 = term_zh2_safe(delta) / weight_denominator_2 if weight_denominator_2 != 0.0 else 0.5
            lambda_denominator_2 = f_safe(opt_z_2) + f_safe(delta)
            lambda_2 = (f_safe(opt_z_2) * f_safe(delta)) / lambda_denominator_2 if lambda_denominator_2 != 0.0 else 0.0
        else:
            opt_z_2, opt_weight_2, lambda_2 = None, None, -np.inf
            
    if lambda_1 >= lambda_2 and (0.0 <= opt_weight_1 <= 1.0):
        best_z, best_weight = opt_z_1, opt_weight_1
    else:
        best_z, best_weight = opt_z_2, opt_weight_2
        
    opt_design = pd.Series([best_weight, 1.0 - best_weight], index=[best_z, delta])
    return opt_design.sort_index()


# =============================================================================
# 3. MASTER CONTROLLERS (Check constraints and branch accordingly)
# =============================================================================

def D_optimal(delta: float = float('inf'), model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    """Master controller for D-optimal design."""
    unrestricted_design = D_optimal_unrestricted(model=model, g=g, m=m)
    if unrestricted_design.index.max() <= delta:
        return unrestricted_design
    return D_optimal_delta(delta, model=model, g=g, m=m)

def A_optimal(delta: float = float('inf'), model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    """Master controller for A-optimal design."""
    unrestricted_design = A_optimal_unrestricted(model=model, g=g, m=m)
    if unrestricted_design.index.max() <= delta:
        return unrestricted_design
    return A_optimal_delta(delta, model=model, g=g, m=m)

def E_optimal(delta: float = float('inf'), model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> pd.Series:
    """Master controller for E-optimal design."""
    unrestricted_design = E_optimal_unrestricted(model=model, g=g, m=m)
    if unrestricted_design.index.max() <= delta:
        return unrestricted_design
    return E_optimal_delta(delta, model=model, g=g, m=m)

def cc_optimal(
    delta: float = float('inf'), 
    model: str = 'logistic', 
    g: float = 0.5, 
    m: float = 0.5, 
    tolerance: Optional[float] = 0.8, 
    approx: bool = True,
    warning: bool = True
) -> Union[Tuple[pd.Series, float], pd.DataFrame]:
    """Master controller for cc-optimal design with caching system."""
    is_approximated = False
    
    if model in ['logistic', 'probit'] and m == 0.5:
        if approx and (np.round(g, 2) != g):
            g = float(np.round(g, 2))
            if warning:
                print(f"Warning: Gamma rounded to {g} to speed up algorithm via cache.")
                
        if np.round(g, 2) == g:
            try:
                current_dir = os.path.dirname(os.path.abspath(__file__))
                data_dir = os.path.join(current_dir, '..', 'data')
                
                path_D = os.path.join(data_dir, f'D_{model}.csv')
                path_mu = os.path.join(data_dir, f'Eff_mu_{model}.csv')
                path_eta = os.path.join(data_dir, f'Eff_eta_{model}.csv')
                
                D_s = pd.read_csv(path_D, index_col=0)[str(g)]
                eff_mu_s = pd.read_csv(path_mu, index_col=0)[str(g)]
                eff_eta_s = pd.read_csv(path_eta, index_col=0)[str(g)]
                
                l_change = D_s[D_s > delta].index
                x0 = None
                for l in l_change:
                    dis, x0 = cc_lambda_delta(l, delta, model=model, g=g, m=m, x0=x0, return_x0=True)
                    eff_mu_s[l] = eff_mu(dis, model=model, g=g, m=m)
                    eff_eta_s[l] = eff_eta(dis, model=model, g=g, m=m)
                
                is_approximated = True
            except FileNotFoundError:
                if warning:
                    print(f"Warning: Cache files for {model} not found. Calculating from scratch.")
                is_approximated = False

    if not is_approximated:
        lambda_list = np.arange(1, 101) / 100.0
        eff_mu_s = pd.Series(index=lambda_list, dtype=float)
        eff_eta_s = pd.Series(index=lambda_list, dtype=float)
        
        x0 = None
        for l in lambda_list:
            dis, x0 = cc_lambda_unrestricted(l, model=model, g=g, m=m, x0=x0, return_x0=True)
            if dis.index.max() > delta: 
                dis, x0 = cc_lambda_delta(l, delta, model=model, g=g, m=m, x0=x0, return_x0=True)
            eff_mu_s[l] = eff_mu(dis, model=model, g=g, m=m)
            eff_eta_s[l] = eff_eta(dis, model=model, g=g, m=m)
            
    df_lambda = pd.DataFrame({'mu': eff_mu_s, 'eta': eff_eta_s})
    
    if tolerance is not None:
        valid_designs = df_lambda[df_lambda['mu'] >= tolerance]
        if valid_designs.empty:
            if warning: print(f"Warning: No design reaches the tolerance of {tolerance} for mu.")
            return valid_designs 
            
        opt_lambda = valid_designs['eta'].idxmax()
        dis, _ = cc_lambda_unrestricted(opt_lambda, model=model, g=g, m=m, x0=None, return_x0=True)
        if dis.index.max() > delta: 
            dis, _ = cc_lambda_delta(opt_lambda, delta, model=model, g=g, m=m, x0=None, return_x0=True)
            
        return dis, opt_lambda
    else:
        return df_lambda