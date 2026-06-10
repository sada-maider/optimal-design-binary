# Optimal Experimental Design for Binary Response Data

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the official Python implementation and auxiliary code for the Ph.D. thesis: **"Contributions to Experimental Designs for Binary Response Data"** by Maider Sada Allo.

> ⚠️ **Note:** > This repository contains the continuously evolving code for the complete Ph.D. thesis. If you are looking for the exact code used to generate the results for the manuscript **"Joint Estimation of Target Dose and Slope in Binary Response Models: A Geometric Approach"**, please refer to the official archived version via this DOI: **[10.5281/zenodo.18611357](https://doi.org/10.5281/zenodo.18611357)**.

## 📂 Repository Structure

The project has been organized following professional software engineering standards to ensure full reproducibility:

* **`src/`**: The core mathematical engine. Contains the pure Python modules for calculating Fisher Information Matrices (`reparameterization.py`), evaluating optimal designs (`optimal_designs.py`), and computing penalized cost ratios (`minimum_cost.py`).
* **`tutorials/`**: Interactive Jupyter Notebooks demonstrating how to use the modules. Includes step-by-step examples for unrestricted designs and cost minimization analysis.
* **`data/`**: Contains pre-calculated `.csv` files (e.g., `D_logistic.csv`, `Eff_mu_logistic.csv`) used to speed up the execution of the restricted algorithmic functions.

## 🚀 Installation

To ensure full reproducibility, you can install all the exact dependencies required to run the code by executing the following command in your terminal:

```bash
pip install -r requirements.txt