import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import trapz, quad
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, ConstantKernel as C

warnings.filterwarnings("ignore")

class WienerKernel(Kernel):
    def __init__(self, length_scale=1.0, length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
        K = np.minimum(X, Y.T)
        if eval_gradient:
            if self.hyperparameter_length_scale.fixed:
                return K, np.empty((X.shape[0], X.shape[0], 0))
            else:
                return K, (K / self.length_scale**2)[:, :, np.newaxis]
        else:
            return K

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return True

def func(x):
    return 1 * np.sin(3 * 2 * np.pi / 10 * x) * x / 10 + 2

a, b = 2, 10
x = np.linspace(a, b, 10000)
y_trapz = func(x)

integral_trapz = trapz(y_trapz, x)

num_points_list = [4, 6, 8, 10]
results = {}

def gp_predict_1d(x):
    x_array = np.atleast_1d(x)[:, np.newaxis]
    return gp.predict(x_array).squeeze()

for num_points in num_points_list:
    x_obs = np.linspace(a, b, num_points)
    y_obs = func(x_obs)

    kernel = C(1.0, (1e-3, 1e3)) * WienerKernel()
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(x_obs[:, np.newaxis], y_obs)
    x_gp = np.linspace(a, b, 500)
    y_mean, y_std = gp.predict(x_gp[:, np.newaxis], return_std=True)

    integral_bq, _ = quad(gp_predict_1d, a, b)
    results[num_points] = integral_bq

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5)
    ax.plot(x, func(x), color='k', lw=2)
    ax.fill_between(x, func(x), alpha=0.08, color='blue', label='Actual Integral')
    ax.plot(x_gp, y_mean, 'k--', color='purple', lw=2, label="GP Regression")
    ax.fill_between(x_gp, y_mean - y_std, y_mean + y_std, alpha=0.2, color='purple',
                    label='GP Covariance')
    ax.scatter(x_obs, y_obs, color='r', marker='o', s=30, label='Sample Points')
    ax.tick_params(axis='x', labelsize=30)
    ax.tick_params(axis='y', labelsize=30)
    ax.set_xlabel('$t$', fontsize=30)
    ax.set_ylabel('$l(x(t), u(x(t)))$', fontsize=30)
    plt.ylim(0, 4)
    ax.legend(fontsize=30, loc='upper right', ncol=2)
    plt.tight_layout()
    plt.savefig(f"BQ_with_Wiener_{num_points}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Actual Integral: {integral_trapz:.4f}")
    print(f"Bayesian Quadrature Integral with {num_points} points: {integral_bq:.4f}")

print("Results for all runs:", results)
