from __future__ import annotations
"""
Uncertainty-Aware E-Beam Calibration & Autonomous EWMA Drift Monitoring.

1. Calibrates objective lens and X/Y stigmators via multi-start L-BFGS-B over
   the GP surrogate using an Upper Confidence Bound (UCB) risk penalty.
2. Calibrates 2D beam deflector gain, rotation, and cubic pincushion distortion.
3. Simulates 48 hours of thermal & HV drift, using an Exponentially Weighted
   Moving Average (EWMA) control chart to trigger autonomous recalibration.
4. Saves summary diagnostic figures to assets/.
"""

import os
import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from physics_column import EBeamColumnSimulator
from surrogate_and_uq import ACTUATOR_BOUNDS, EBeamSurrogateUQ


def calibrate_focus_and_stigmation(
    surrogate: EBeamSurrogateUQ,
    landing_energy_kev: float = 2.0,
    ucb_kappa: float = 0.5,
) -> np.ndarray:
    """
    Find optimal [i_obj, i_stig_x, i_stig_y] by minimizing effective probe spot
    diameter plus an epistemic uncertainty penalty (UCB) to avoid under-sampled regions.
    """
    bounds_3d = [
        tuple(ACTUATOR_BOUNDS[0]),
        tuple(ACTUATOR_BOUNDS[1]),
        tuple(ACTUATOR_BOUNDS[2]),
    ]

    def objective(u: np.ndarray) -> float:
        x_query = np.array([[u[0], u[1], u[2], landing_energy_kev]])
        mx, sx, my, sy = surrogate.predict(x_query)
        d_eff = np.sqrt(0.5 * (mx[0] ** 2 + my[0] ** 2))
        unc_penalty = 0.5 * (sx[0] + sy[0])
        return float(d_eff + ucb_kappa * unc_penalty)

    initial_guesses = [
        np.array([400.0, 0.0, 0.0]),
        np.array([440.0, 2.0, -2.0]),
        np.array([420.0, -2.0, 2.0]),
    ]

    best_res = None
    for x0 in initial_guesses:
        res = minimize(objective, x0=x0, bounds=bounds_3d, method="L-BFGS-B")
        if best_res is None or res.fun < best_res.fun:
            best_res = res

    return best_res.x


def calibrate_deflector_grid(simulator: EBeamColumnSimulator) -> dict[str, float]:
    """
    Fit 2D deflector gain, rotation angle, and cubic distortion coefficients
    from a 5x5 wafer fiducial calibration grid via linear least squares.
    """
    grid_1d = np.linspace(-20.0, 20.0, 5)
    vx, vy = np.meshgrid(grid_1d, grid_1d)
    vx_flat, vy_flat = vx.ravel(), vy.ravel()

    meas = simulator.measure(
        i_obj=420.0,
        i_stig_x=3.2,
        i_stig_y=-2.1,
        v_def_x=vx_flat,
        v_def_y=vy_flat,
        add_noise=True,
    )
    px, py = meas["beam_pos_x_um"], meas["beam_pos_y_um"]

    # Design matrix: [vx, vy, vx^3, vy^3]
    Phi = np.column_stack([vx_flat, vy_flat, vx_flat ** 3, vy_flat ** 3])
    coef_x, _, _, _ = np.linalg.lstsq(Phi, px, rcond=None)
    coef_y, _, _, _ = np.linalg.lstsq(Phi, py, rcond=None)

    est_theta = float(np.arctan2(coef_y[0], coef_x[0]))
    est_gx = float(np.hypot(coef_x[0], coef_y[0]))
    est_gy = float(np.hypot(coef_x[1], coef_y[1]))

    pred_px = Phi @ coef_x
    pred_py = Phi @ coef_y
    residual_nm = float(
        np.mean(np.hypot(px - pred_px, py - pred_py)) * 1000.0
    )

    return {
        "est_gain_x": est_gx,
        "est_gain_y": est_gy,
        "est_rotation_mrad": est_theta * 1000.0,
        "mean_placement_residual_nm": residual_nm,
    }


def run_48hr_drift_monitor(
    u_initial: np.ndarray,
    baseline_spot_nm: float = 2.10,
    noise_std_nm: float = 0.15,
    ewma_lambda: float = 0.25,
    control_sigma_mult: float = 3.0,
) -> dict[str, np.ndarray | list[float] | float]:
    """
    Simulate 48 hours of wafer-lot operation comparing open-loop drift against
    closed-loop EWMA drift detection and rapid online recalibration.
    """
    hours = np.arange(0.0, 48.5, 0.5)
    sim_open = EBeamColumnSimulator(noise_std_nm=noise_std_nm, seed=101)
    sim_closed = EBeamColumnSimulator(noise_std_nm=noise_std_nm, seed=101)

    u_open = u_initial.copy()
    u_closed = u_initial.copy()

    ucl = baseline_spot_nm + control_sigma_mult * noise_std_nm * np.sqrt(
        ewma_lambda / (2.0 - ewma_lambda)
    )
    z_ewma = baseline_spot_nm

    spot_open_hist = []
    spot_closed_hist = []
    ewma_hist = []
    recal_hours = []

    for t in hours:
        sim_open.apply_time_drift(t)
        sim_closed.apply_time_drift(t)

        m_open = sim_open.measure(*u_open, add_noise=True)
        d_open = float(
            np.sqrt(0.5 * (m_open["spot_size_x_nm"] ** 2 + m_open["spot_size_y_nm"] ** 2))
        )
        spot_open_hist.append(d_open)

        m_closed = sim_closed.measure(*u_closed, add_noise=True)
        d_closed = float(
            np.sqrt(
                0.5 * (m_closed["spot_size_x_nm"] ** 2 + m_closed["spot_size_y_nm"] ** 2)
            )
        )

        z_ewma = ewma_lambda * d_closed + (1.0 - ewma_lambda) * z_ewma
        spot_closed_hist.append(d_closed)
        ewma_hist.append(z_ewma)

        if z_ewma > ucl:
            recal_hours.append(float(t))
            # Rapid local Powell recalibration on live column diagnostics
            res = minimize(
                lambda u: float(
                    np.sqrt(
                        0.5
                        * (
                            sim_closed.measure(u[0], u[1], u[2], add_noise=True)[
                                "spot_size_x_nm"
                            ]
                            ** 2
                            + sim_closed.measure(u[0], u[1], u[2], add_noise=True)[
                                "spot_size_y_nm"
                            ]
                            ** 2
                        )
                    )
                ),
                x0=u_closed,
                method="Powell",
                options={"maxfev": 35, "xtol": 0.05, "ftol": 0.02},
            )
            u_closed = res.x
            z_ewma = baseline_spot_nm

    return {
        "hours": hours,
        "spot_open": np.array(spot_open_hist),
        "spot_closed": np.array(spot_closed_hist),
        "ewma": np.array(ewma_hist),
        "ucl": ucl,
        "recal_hours": recal_hours,
    }


def generate_summary_plots(
    Y_test: np.ndarray,
    pred_x: np.ndarray,
    std_x: np.ndarray,
    uq_summary: dict[str, np.ndarray],
    drift_results: dict[str, np.ndarray | list[float] | float],
    output_dir: str = "assets",
) -> None:
    """Save diagnostic plots for GitHub README documentation."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    # --- Figure 1: Surrogate Parity & Uncertainty Budget ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8), dpi=160)

    r2 = r2_score(Y_test[:, 0], pred_x)
    rmse = root_mean_squared_error(Y_test[:, 0], pred_x)
    ax1.errorbar(
        Y_test[:, 0],
        pred_x,
        yerr=2.0 * std_x,
        fmt="o",
        color="#1f77b4",
        ecolor="#aec7e8",
        alpha=0.8,
        markersize=5,
        label=r"GP Prediction ($\pm 2\sigma$ UQ)",
    )
    lims = [min(Y_test[:, 0].min(), pred_x.min()) - 2, max(Y_test[:, 0].max(), pred_x.max()) + 2]
    ax1.plot(lims, lims, "k--", lw=1.2, label="Ideal 1:1 Parity")
    ax1.set_xlim(lims)
    ax1.set_ylim(lims)
    ax1.set_xlabel("True Physics Spot Size X (nm)", fontsize=10)
    ax1.set_ylabel("GP Surrogate Prediction X (nm)", fontsize=10)
    ax1.set_title(f"GP Surrogate Validation ($R^2={r2:.4f}$, RMSE={rmse:.2f} nm)", fontsize=11)
    ax1.legend(loc="upper left", frameon=True)

    labels = [
        r"$I_{\mathrm{obj}}$ Jitter",
        r"$I_{\mathrm{stig},x}$ Jitter",
        r"$I_{\mathrm{stig},y}$ Jitter",
        r"$E_0$ HV Ripple",
        "GP Epistemic",
    ]
    var_x = list(uq_summary["var_contrib_x"]) + [uq_summary["epistemic_var"][0]]
    var_y = list(uq_summary["var_contrib_y"]) + [uq_summary["epistemic_var"][1]]
    x_idx = np.arange(len(labels))
    width = 0.36

    ax2.bar(x_idx - width / 2, var_x, width, label="Spot Size X Variance", color="#2b5c8f")
    ax2.bar(x_idx + width / 2, var_y, width, label="Spot Size Y Variance", color="#d95f02")
    ax2.set_xticks(x_idx)
    ax2.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax2.set_ylabel("Variance Contribution ($\mathrm{nm}^2$)", fontsize=10)
    ax2.set_title(
        f"Jacobian UQ Budget (Total 1$\sigma$: {uq_summary['jacobian_total_std_nm'][0]:.2f} nm vs MC {uq_summary['mc_total_std_nm'][0]:.2f} nm)",
        fontsize=10.5,
    )
    ax2.legend(frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "surrogate_uq_validation.png"))
    plt.close(fig)

    # --- Figure 2: 48-Hour Drift & Autonomous Recalibration ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 6.2), sharex=True, dpi=160, gridspec_kw={"height_ratios": [1.3, 1]}
    )

    hours = drift_results["hours"]
    ax_top.plot(
        hours,
        drift_results["spot_open"],
        color="#d62728",
        lw=1.8,
        alpha=0.85,
        label="Open-Loop (No Recalibration)",
    )
    ax_top.plot(
        hours,
        drift_results["spot_closed"],
        color="#1f77b4",
        lw=1.8,
        label="Closed-Loop (EWMA Self-Calibration)",
    )
    for idx, t_rec in enumerate(drift_results["recal_hours"]):
        ax_top.axvline(
            t_rec,
            color="#2ca02c",
            ls="--",
            lw=1.0,
            alpha=0.75,
            label="Autonomous Recalibration" if idx == 0 else None,
        )
    ax_top.set_ylabel("Probe Spot Diameter (nm)", fontsize=10)
    ax_top.set_title("48-Hour E-Beam Column Drift Compensation & Self-Recovery", fontsize=11.5)
    ax_top.legend(loc="upper left", frameon=True)

    ax_bot.plot(hours, drift_results["ewma"], color="#9467bd", lw=1.8, label=r"EWMA Statistic ($Z_t$)")
    ax_bot.axhline(
        drift_results["ucl"],
        color="#d62728",
        ls="--",
        lw=1.4,
        label=f"Upper Control Limit (UCL = {drift_results['ucl']:.2f} nm)",
    )
    for t_rec in drift_results["recal_hours"]:
        ax_bot.axvline(t_rec, color="#2ca02c", ls="--", lw=1.0, alpha=0.75)
    ax_bot.set_xlabel("Operating Time (Hours)", fontsize=10)
    ax_bot.set_ylabel("EWMA Monitor (nm)", fontsize=10)
    ax_bot.legend(loc="upper left", frameon=True)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, "drift_and_self_calibration.png"))
    plt.close(fig)


if __name__ == "__main__":
    sim = EBeamColumnSimulator(seed=42)

    # 1. Train GP Surrogate
    X_train, Y_train = EBeamSurrogateUQ.generate_doe_dataset(sim, n_samples=180, seed=42)
    X_test, Y_test = EBeamSurrogateUQ.generate_doe_dataset(
        sim, n_samples=60, seed=99, add_noise=False
    )
    surrogate = EBeamSurrogateUQ(random_state=42).fit(X_train, Y_train)
    px, sx, _, _ = surrogate.predict(X_test)

    # 2. Optimize Focus & Stigmation
    u_opt = calibrate_focus_and_stigmation(surrogate, landing_energy_kev=2.0, ucb_kappa=0.5)
    val = sim.measure(u_opt[0], u_opt[1], u_opt[2], landing_energy_kev=2.0, add_noise=False)
    print(f"Calibrated Actuators (I_obj, Stig_X, Stig_Y): {np.round(u_opt, 3)}")
    print(
        f"Validated Spot Size (X, Y): "
        f"{val['spot_size_x_nm']:.3f} nm, {val['spot_size_y_nm']:.3f} nm"
    )

    # 3. Calibrate 2D Deflector Field
    def_cal = calibrate_deflector_grid(sim)
    print(
        f"Deflector Calibration -> Gain: ({def_cal['est_gain_x']:.4f}, {def_cal['est_gain_y']:.4f}), "
        f"Rotation: {def_cal['est_rotation_mrad']:.2f} mrad, "
        f"Residual: {def_cal['mean_placement_residual_nm']:.2f} nm"
    )

    # 4. Run UQ Sensitivity & 48-Hour Drift Monitor
    uq_summary = surrogate.propagate_uncertainty(
        x_nominal=np.array([423.0, 3.8, -1.5, 2.0]),
        actuator_stds=np.array([0.35, 0.15, 0.15, 0.01]),
    )
    drift_results = run_48hr_drift_monitor(u_initial=u_opt)
    print(f"Autonomous Recalibrations Triggered: {len(drift_results['recal_hours'])} events")
    print(
        f"48h Spot Size -> Open-Loop: {drift_results['spot_open'][-1]:.2f} nm | "
        f"Closed-Loop: {drift_results['spot_closed'][-1]:.2f} nm"
    )

    generate_summary_plots(Y_test, px, sx, uq_summary, drift_results)
    print("Saved plots to assets/surrogate_uq_validation.png and assets/drift_and_self_calibration.png")