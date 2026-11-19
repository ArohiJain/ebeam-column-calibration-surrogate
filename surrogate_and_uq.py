from __future__ import annotations
"""
Physics-Informed Gaussian Process Surrogate & Uncertainty Quantification (UQ).

Trains a multi-output Gaussian Process surrogate on Latin Hypercube Design of
Experiments (DoE) samples from the e-beam column simulator, extracts Automatic
Relevance Determination (ARD) feature sensitivities, and propagates actuator
jitter via analytical finite-difference Jacobians and Monte Carlo sampling.
"""

import warnings
import numpy as np
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from physics_column import EBeamColumnSimulator

# Actuator and operating bounds: [i_obj (mA), i_stig_x (mA), i_stig_y (mA), energy (keV)]
ACTUATOR_BOUNDS = np.array(
    [
        [350.0, 520.0],
        [-8.0, 8.0],
        [-8.0, 8.0],
        [1.5, 3.0],
    ],
    dtype=float,
)


class EBeamSurrogateUQ:
    """
    Gaussian Process surrogate with epistemic + aleatoric error propagation.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()

        kernel_x = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e4), nu=2.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
        )
        kernel_y = (
            ConstantKernel(1.0, (1e-3, 1e3))
            * Matern(length_scale=np.ones(4), length_scale_bounds=(1e-2, 1e4), nu=2.5)
            + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1.0))
        )

        self.gp_x = GaussianProcessRegressor(
            kernel=kernel_x,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=random_state,
        )
        self.gp_y = GaussianProcessRegressor(
            kernel=kernel_y,
            normalize_y=True,
            n_restarts_optimizer=3,
            random_state=random_state + 1,
        )
        self.is_fitted = False

    @staticmethod
    def generate_doe_dataset(
        simulator: EBeamColumnSimulator,
        n_samples: int = 180,
        seed: int = 42,
        add_noise: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate space-filling Latin Hypercube samples across column operating space."""
        sampler = qmc.LatinHypercube(d=4, seed=seed)
        unit_samples = sampler.random(n=n_samples)
        X = qmc.scale(unit_samples, ACTUATOR_BOUNDS[:, 0], ACTUATOR_BOUNDS[:, 1])

        meas = simulator.measure(
            i_obj=X[:, 0],
            i_stig_x=X[:, 1],
            i_stig_y=X[:, 2],
            landing_energy_kev=X[:, 3],
            add_noise=add_noise,
        )
        Y = np.column_stack([meas["spot_size_x_nm"], meas["spot_size_y_nm"]])
        return X, Y

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "EBeamSurrogateUQ":
        """Fit standardized GP surrogates for X and Y probe spot sizes."""
        X_scaled = self.scaler.fit_transform(X)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            self.gp_x.fit(X_scaled, Y[:, 0])
            self.gp_y.fit(X_scaled, Y[:, 1])
        self.is_fitted = True
        return self

    def predict(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predict spot sizes (nm) and 1-sigma GP epistemic uncertainties."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate must be fitted before calling predict().")
        X_arr = np.atleast_2d(X)
        X_scaled = self.scaler.transform(X_arr)
        mean_x, std_x = self.gp_x.predict(X_scaled, return_std=True)
        mean_y, std_y = self.gp_y.predict(X_scaled, return_std=True)
        return mean_x, std_x, mean_y, std_y

    def compute_jacobian(
        self, x_nominal: np.ndarray, step_sizes: np.ndarray | None = None
    ) -> np.ndarray:
        """Compute 2x4 sensitivity Jacobian J = d(spot_x, spot_y) / d(actuators)."""
        if step_sizes is None:
            step_sizes = np.array([0.05, 0.02, 0.02, 0.005])

        J = np.zeros((2, 4), dtype=float)
        for idx in range(4):
            x_plus = x_nominal.copy()
            x_minus = x_nominal.copy()
            x_plus[idx] += step_sizes[idx]
            x_minus[idx] -= step_sizes[idx]

            mx_p, _, my_p, _ = self.predict(x_plus)
            mx_m, _, my_m, _ = self.predict(x_minus)

            J[0, idx] = (mx_p[0] - mx_m[0]) / (2.0 * step_sizes[idx])
            J[1, idx] = (my_p[0] - my_m[0]) / (2.0 * step_sizes[idx])
        return J

    def propagate_uncertainty(
        self,
        x_nominal: np.ndarray,
        actuator_stds: np.ndarray,
        n_mc: int = 2000,
    ) -> dict[str, np.ndarray]:
        """
        Propagate actuator jitter via first-order Jacobian (Sigma_y = J Sigma_x J^T + Sigma_GP)
        and cross-validate against Monte Carlo simulation.
        """
        J = self.compute_jacobian(x_nominal)
        Sigma_x = np.diag(actuator_stds ** 2)
        Sigma_aleatoric = J @ Sigma_x @ J.T

        _, std_gp_x, _, std_gp_y = self.predict(x_nominal)
        Sigma_epistemic = np.diag([std_gp_x[0] ** 2, std_gp_y[0] ** 2])
        Sigma_total = Sigma_aleatoric + Sigma_epistemic

        # Per-actuator variance contributions for sensitivity budget
        var_contrib_x = (J[0, :] * actuator_stds) ** 2
        var_contrib_y = (J[1, :] * actuator_stds) ** 2

        # Monte Carlo verification
        rng = np.random.default_rng(self.random_state + 10)
        mc_inputs = rng.normal(loc=x_nominal, scale=actuator_stds, size=(n_mc, 4))
        mc_mx, mc_sx, mc_my, mc_sy = self.predict(mc_inputs)
        mc_draws_x = rng.normal(mc_mx, mc_sx)
        mc_draws_y = rng.normal(mc_my, mc_sy)

        return {
            "jacobian": J,
            "jacobian_total_std_nm": np.sqrt(np.diag(Sigma_total)),
            "mc_total_std_nm": np.array([np.std(mc_draws_x), np.std(mc_draws_y)]),
            "var_contrib_x": var_contrib_x,
            "var_contrib_y": var_contrib_y,
            "epistemic_var": np.array([std_gp_x[0] ** 2, std_gp_y[0] ** 2]),
        }


if __name__ == "__main__":
    sim = EBeamColumnSimulator(seed=42)
    X_train, Y_train = EBeamSurrogateUQ.generate_doe_dataset(sim, n_samples=180, seed=42)
    X_test, Y_test = EBeamSurrogateUQ.generate_doe_dataset(
        sim, n_samples=60, seed=99, add_noise=False
    )

    surrogate = EBeamSurrogateUQ(random_state=42).fit(X_train, Y_train)
    px, sx, py, sy = surrogate.predict(X_test)

    print(
        f"Test Accuracy X -> R2: {r2_score(Y_test[:, 0], px):.5f}, "
        f"RMSE: {root_mean_squared_error(Y_test[:, 0], px):.3f} nm"
    )
    print(
        f"Test Accuracy Y -> R2: {r2_score(Y_test[:, 1], py):.5f}, "
        f"RMSE: {root_mean_squared_error(Y_test[:, 1], py):.3f} nm"
    )

    x_op = np.array([423.0, 3.8, -1.5, 2.0])
    jitter = np.array([0.35, 0.15, 0.15, 0.01])
    uq = surrogate.propagate_uncertainty(x_op, jitter)
    print("Jacobian Propagated 1-sigma (X, Y) nm:", np.round(uq["jacobian_total_std_nm"], 4))
    print("Monte Carlo 1-sigma         (X, Y) nm:", np.round(uq["mc_total_std_nm"], 4))