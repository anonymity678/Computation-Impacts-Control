# Impact of Computation in Integral Reinforcement Learning for Continuous-Time Control

This repository explores the impact of computation in integral reinforcement learning for continuous-time control systems. It contains various Python scripts that demonstrate different methods and approaches.

## Environment Requirements

- **Python**
- **Libraries**:
  - `numpy`
  - `matplotlib`
  - `scipy`
  - `probnum`
  - `sklearn`
  - `argparse`
  - `os`

## Examples
1. **Simulations for trapezoidal rule for Example 1**
   - **File**: `example_tos_trapz.py`

2. **Simulations for Bayesian quadrature with Matérn kernel for Example 1**
   - **File**: `example_tos_Matern.py`

3. **Simulations for trapezoidal rule rule for Example 2**
   - **File**: `example_sin_trapz.py`

4. **Simulations for Bayesian quadrature with Matérn kernel for Example 2**
   - **File**: `example_sin_Matern.py`

## Subdirectories

- **ThirdOrderSystem**: Contains simulation results related to Example 1.
- **SinSystem**: Contains simulation results related to Example 2.
- **Illustration_of_BQ**: Contains scripts like `BQ_matern.py` and `BQ_wiener.py` that illustrate Bayesian quadrature methods.
- **Motivation_Sinsystem**: Motivated example for scenerios when the internal dynamics is unknown.
- **Motivation_Cartpole**: Motivated example for scenerios when the internal dynamics is known.