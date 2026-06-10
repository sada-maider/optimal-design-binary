import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Tuple, Union, List

# Importing from your custom modules
from optimal_designs import D_optimal, A_optimal, E_optimal, cc_optimal
from reparameterization import load_param, eff_mu, eff_eta

def setup_plot_style() -> None:
    """
    Configures the default Matplotlib and Seaborn plotting styles for the thesis.
    Call this function in your main script before generating plots.
    """
    plt.rcParams['text.usetex'] = True
    sns.set_theme(
        style="whitegrid", 
        rc={
            "axes.grid": True,
            "grid.color": "#e0e0e0", 
            "grid.linewidth": 0.5,
            "axes.facecolor": "white" 
        }
    )

# Dictionary of standard colors for the project
COLORS = {
    'r': '#bf775f', 'r2': '#d5572c', 'o': "#e38800", 'y': '#e6d26b',
    'g': "#648B00", 'c': '#98cacd', 'b': '#709acd', 'p': "#645697",  
    'gr': '#444444', 'b2': '#4c72b0'
}

# =============================================================================
# 1. BASE MATHEMATICAL FUNCTIONS
# =============================================================================

def std_FIM(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> Tuple[float, float, float]:
    """Calculates the standard Fisher Information Matrix components (t, u, v)."""
    param = load_param(model=model, g=g, m=m)
    r1_sq = param['r1_2']()
    r2_sq = param['r2_2']()
    h2_func = param['h2']
    
    z = design.index.to_numpy()
    weights = design.to_numpy()
    h2_vals = h2_func(z)
    
    t_val = np.sum(r1_sq * h2_vals * weights)
    u_val = -np.sum(np.sqrt(r1_sq * r2_sq) * h2_vals * weights * z)
    v_val = np.sum(r2_sq * h2_vals * weights * z**2)
    
    return float(t_val), float(u_val), float(v_val)


def cost_rel(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    """Calculates the relative cost of a given design."""
    param = load_param(model=model, g=g, m=m)
    H_func = param['H']
    
    z = design.index.to_numpy()
    weights = design.to_numpy()
    
    return float(np.sum(H_func(z) * weights))


# =============================================================================
# 2. D-OPTIMALITY CRITERIA
# =============================================================================

def D_criterium(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    t_val, u_val, v_val = std_FIM(design, model=model, g=g, m=m)
    determinant = t_val * v_val - u_val**2
    
    return np.inf if np.isclose(determinant, 0.0) else 1.0 / determinant
    

def Eff_D(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    design_opt = D_optimal(model=model, g=g, m=m)
    d_crit_design = D_criterium(design, model=model, g=g, m=m)
    
    if np.isinf(d_crit_design): 
        return 0.0
        
    d_crit_opt = D_criterium(design_opt, model=model, g=g, m=m)
    return np.sqrt(d_crit_opt / d_crit_design)


def cost_ratio_D(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    design_opt = D_optimal(model=model, g=g, m=m)
    
    cost_current = cost_rel(design, model=model, g=g, m=m)
    cost_opt = cost_rel(design_opt, model=model, g=g, m=m)
    
    d_crit_current = D_criterium(design, model=model, g=g, m=m)
    d_crit_opt = D_criterium(design_opt, model=model, g=g, m=m)
    
    return (cost_current / cost_opt) * np.sqrt(d_crit_current / d_crit_opt)


# =============================================================================
# 3. A-OPTIMALITY CRITERIA
# =============================================================================

def std_A_criterium(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    t_val, u_val, v_val = std_FIM(design, model=model, g=g, m=m)
    determinant = t_val * v_val - u_val**2
    
    return np.inf if np.isclose(determinant, 0.0) else (t_val + v_val) / determinant


def Eff_A(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    design_opt = A_optimal(model=model, g=g, m=m)
    return std_A_criterium(design_opt, model=model, g=g, m=m) / std_A_criterium(design, model=model, g=g, m=m)


def cost_ratio_A(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    design_opt = A_optimal(model=model, g=g, m=m)
    
    cost_current = cost_rel(design, model=model, g=g, m=m)
    cost_opt = cost_rel(design_opt, model=model, g=g, m=m)
    
    a_crit_current = std_A_criterium(design, model=model, g=g, m=m)
    a_crit_opt = std_A_criterium(design_opt, model=model, g=g, m=m)
    
    return (cost_current / cost_opt) * (a_crit_current / a_crit_opt)


# =============================================================================
# 4. E-OPTIMALITY CRITERIA
# =============================================================================

def std_E_criterium(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    t_val, u_val, v_val = std_FIM(design, model=model, g=g, m=m)
    determinant = t_val * v_val - u_val**2
    
    if np.isclose(determinant, 0.0):   
        return np.inf
        
    trace = t_val + v_val
    discriminant = round(trace**2 - 4 * determinant, 5)
    
    if discriminant < 0.0:
        return np.inf
        
    return 2.0 / (trace - np.sqrt(discriminant))
    

def Eff_E(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    design_opt = E_optimal(model=model, g=g, m=m)
    return std_E_criterium(design_opt, model=model, g=g, m=m) / std_E_criterium(design, model=model, g=g, m=m)


def cost_ratio_E(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5) -> float:
    design_opt = E_optimal(model=model, g=g, m=m)
    
    cost_current = cost_rel(design, model=model, g=g, m=m)
    cost_opt = cost_rel(design_opt, model=model, g=g, m=m)
    
    e_crit_current = std_E_criterium(design, model=model, g=g, m=m)
    e_crit_opt = std_E_criterium(design_opt, model=model, g=g, m=m)
    
    return (cost_current / cost_opt) * (e_crit_current / e_crit_opt)


# =============================================================================
# 5. cc-OPTIMALITY CRITERIA
# =============================================================================

def cost_ratio_cc(design: pd.Series, model: str = 'logistic', g: float = 0.5, m: float = 0.5, tol: float = 0.8) -> Optional[float]:
    if eff_mu(design, model=model, g=g, m=m) < tol:
        print(f'Warning: Eff_mu(design) is not enough (minimum required: {tol})')
        return None
        
    design_opt = cc_optimal(model=model, g=g, m=m, tolerance=tol)
    
    if design_opt is None:
        return None
        
    cost_current = cost_rel(design, model=model, g=g, m=m)
    cost_opt = cost_rel(design_opt, model=model, g=g, m=m)
    
    eta_current = eff_eta(design, model=model, g=g, m=m)
    eta_opt = eff_eta(design_opt, model=model, g=g, m=m)
    
    # Protecting against division by zero
    if np.isclose(eta_current, 0.0):
        return np.inf
        
    return (cost_current / cost_opt) * (eta_opt / eta_current)


# =============================================================================
# 6. MASTER DESIGN SELECTOR
# =============================================================================

def min_cost_design(
    g: float = 0.5, 
    model: str = 'logistic', 
    m: float = 0.5, 
    criterion: str = 'A', 
    return_cost: bool = False, 
    g0_list: Optional[np.ndarray] = None, 
    tolerance: float = 0.8
) -> Union[pd.Series, Tuple[pd.Series, pd.Series]]:
    """
    Finds the optimal design that minimizes the cost ratio across a grid of g0 values.
    """
    if g0_list is None:
        g0_list = np.arange(1.0, 100.0) / 100.0
        
    cost_ratios = []
    param = load_param(g=g, model=model, m=m)
    H_inv = param['H_inv']
    
    for g0 in g0_list:
        if criterion == 'A':
            design = A_optimal(H_inv(g0), g=g, model=model, m=m)
            cost_ratios.append(cost_ratio_A(design, g=g, model=model, m=m))
            
        elif criterion == 'cc':
            design = cc_optimal(H_inv(g0), g=g, model=model, m=m, warning=False, tolerance=tolerance)
            if design is not None:
                cost_ratios.append(cost_ratio_cc(design, g=g, model=model, m=m, tol=tolerance))
            else:
                cost_ratios.append(np.inf) # Use inf instead of None for easier argmin calculation
                
        elif criterion == 'E':
            design = E_optimal(H_inv(g0), g=g, model=model, m=m)
            cost_ratios.append(cost_ratio_E(design, g=g, model=model, m=m))
            
        else:
            raise ValueError(f"Unknown criterion: '{criterion}'. Supported criteria are 'A', 'E', and 'cc'.")
            
    # Convert list to Series for easy indexing
    cost_series = pd.Series(cost_ratios, index=g0_list)
    
    # Drop infinite values (if cc_optimal returned None)
    cost_series = cost_series[cost_series != np.inf]
    
    if cost_series.empty:
        raise ValueError(f"No valid designs found for criterion '{criterion}' across the provided g0 grid.")
        
    best_g0 = cost_series.idxmin()
    
    # Recalculate the best design
    if criterion == 'A':    
        best_design = A_optimal(H_inv(best_g0), g=g, model=model, m=m)
    elif criterion == 'cc': 
        best_design = cc_optimal(H_inv(best_g0), g=g, model=model, m=m, warning=False, tolerance=tolerance)
    elif criterion == 'E':  
        best_design = E_optimal(H_inv(best_g0), g=g, model=model, m=m)
        
    return (best_design, cost_series) if return_cost else best_design