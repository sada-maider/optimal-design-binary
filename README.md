# optimal-design-binary
This repository contains the Python implementation of the algorithms and optimization procedures described in the paper:

> **Joint Estimation of Target Dose and Slope in Binary Response Models: A Geometric Approach**
> 
> *Authors:* Maider Sada¹, José A. Moler¹, and Nancy Flournoy²
> 
> *¹Public University of Navarre (UPNA), ²University of Missouri*

## Overview

In Phase I clinical trials, estimating both the target dose $\mu_\Gamma$ (associated with a specific toxicity rate $\Gamma$) and the slope $\eta_\Gamma$ of the dose-response curve is critical for safety and efficiency.

This work generalizes the geometric framework of Sitter and Fainaru (1997) to arbitrary quantiles $\Gamma \in (0,1)$. We demonstrate that for a wide class of optimality criteria, the optimal design is a two-point design symmetric about the median. This characterization significantly reduces the computational complexity from a five-parameter optimization problem to a **single-parameter** problem.

## Key Features

* **Generalized Geometric Approach:** Computes optimal designs for any target toxicity rate $\Gamma \in (0,1)$.
* **Efficiency:** Reduces the optimization search space to a single parameter ($\delta$), enabling fast and precise calculations.
* **Supported Models:** Logistic and Probit models.
* **Optimality Criteria:**
    * **Standardized A-optimality:** Minimizes the sum of variances of the parameter estimates.
    * **cc-optimality:** Minimizes the variance of the slope estimate given a constraint on the variance of the dose estimate.
* **Constrained Designs:** Algorithms to compute optimal designs under restricted dose spaces (e.g., dose $\le \Delta$).

## Repository Structure

* **`optimal_designs.py`**: Core library containing the model definitions and optimization algorithms.
* **`Tutorial.ipynb`**: A Jupyter Notebook demonstrating how to use the library. It reproduces **Table 2** (Standardized A-optimal) and **Table 3** (cc-optimal) from the manuscript.
* **`requirements.txt`**: List of dependencies.

## Installation

To run the code, you need **Python 3** and the following scientific libraries:

```bash
pip install -r requirements.txt
