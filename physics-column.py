"""
Virtual Electron-Optical Column Forward Model.

Simulates paraxial electron optics, primary lens aberrations (spherical,
chromatic, astigmatism), beam deflection distortion, and time-dependent
hardware drift (thermal focus shift and high-voltage instability).
"""

from dataclasses import dataclass
import numpy as np


@dataclass
class ColumnTrueState:
    """Hidden physical state of the e-beam column that calibration must infer."""
    optimal_obj_current_ma: float = 420.0   # True focal current at nominal energy (mA)
    parasitic_stig_x_ma: float = 3.2        # Intrinsic column astigmatism X (mA)
    parasitic_stig_y_ma: float = -2.1       # Intrinsic column astigmatism Y (mA)
    deflector_gain_x: float = 1.015         # X-deflector scale error (um/V)
    deflector_gain_y: float = 0.988         # Y-deflector scale error (um/V)
    deflector_rotation_rad: float = 0.018   # Column-to-stage rotational misalignment (rad)
    hv_instability_ppm: float = 25.0        # High-voltage ripple / energy spread (ppm)


class EBeamColumnSimulator:
    """
    Analytical physics forward model of a scanning electron-optical column.

    Control Inputs (Actuators):
        - i_obj: Objective lens excitation current (mA)
        - i_stig_x, i_stig_y: Stigmator coil currents (mA)
        - v_def_x, v_def_y: Electrostatic deflector voltages (V)
        - landing_energy_kev: Primary beam landing energy (keV)

    Measured Outputs (Diagnostics):
        - spot_size_x_nm, spot_size_y_nm: Beam probe diameter (nm, 1-sigma)
        - beam_pos_x_um, beam_pos_y_um: Wafer landing coordinates (um)
    """

    def __init__(
        self,
        true_state: ColumnTrueState | None = None,
        noise_std_nm: float = 0.15,
        pos_noise_std_um: float = 0.005,
        seed: int = 42,
    ):
        self.state = true_state if true_state is not None else ColumnTrueState()
        self.noise_std_nm = noise_std_nm
        self.pos_noise_std_um = pos_noise_std_um
        self.rng = np.random.default_rng(seed)

        # Fixed column optical constants
        self.source_size_nm = 1.8          # Virtual source demagnified size (nm)
        self.semi_angle_mrad = 6.5         # Beam convergence semi-angle alpha (mrad)
        self.cs_mm = 1.4                   # Spherical aberration coefficient (mm)
        self.cc_mm = 1.1                   # Chromatic aberration coefficient (mm)
        self.defocus_sens_nm_per_ma = 0.85 # Focus sensitivity to objective current
        self.stig_sens_nm_per_ma = 1.10    # Astigmatism blur sensitivity

    def apply_time_drift(self, elapsed_hours: float) -> None:
        """
        Simulate gradual thermal lens drift and vacuum/HV aging over operation time.
        """
        # Thermal expansion shifts optimal objective current exponentially toward saturation
        thermal_shift_ma = 4.5 * (1.0 - np.exp(-elapsed_hours / 3.0))
        self.state.optimal_obj_current_ma = 420.0 + thermal_shift_ma

        # Slow linear drift in parasitic astigmatism and HV ripple
        self.state.parasitic_stig_x_ma = 3.2 + 0.18 * elapsed_hours
        self.state.parasitic_stig_y_ma = -2.1 - 0.12 * elapsed_hours
        self.state.hv_instability_ppm = 25.0 + 2.5 * elapsed_hours

    def measure(
        self,
        i_obj: np.ndarray | float,
        i_stig_x: np.ndarray | float,
        i_stig_y: np.ndarray | float,
        v_def_x: np.ndarray | float = 0.0,
        v_def_y: np.ndarray | float = 0.0,
        landing_energy_kev: np.ndarray | float = 2.0,
        add_noise: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Evaluate beam spot size and landing position for single or batched actuator inputs.
        """
        i_obj = np.asarray(i_obj, dtype=float)
        i_stig_x = np.asarray(i_stig_x, dtype=float)
        i_stig_y = np.asarray(i_stig_y, dtype=float)
        v_def_x = np.asarray(v_def_x, dtype=float)
        v_def_y = np.asarray(v_def_y, dtype=float)
        energy = np.asarray(landing_energy_kev, dtype=float)

        alpha = self.semi_angle_mrad * 1e-3  # Convert mrad to rad

        # 1. Defocus blur (scales with energy-adjusted optimal current)
        energy_scale = np.sqrt(energy / 2.0)
        target_i_obj = self.state.optimal_obj_current_ma * energy_scale
        d_defocus = self.defocus_sens_nm_per_ma * (i_obj - target_i_obj)

        # 2. Spherical & chromatic aberration disks (converted from mm to nm)
        d_spherical = 0.5 * (self.cs_mm * 1e6) * (alpha ** 3)
        delta_e_over_e = self.state.hv_instability_ppm * 1e-6
        d_chromatic = (self.cc_mm * 1e6) * delta_e_over_e * alpha

        # 3. Residual astigmatism in X and Y
        d_ast_x = self.stig_sens_nm_per_ma * (i_stig_x - self.state.parasitic_stig_x_ma)
        d_ast_y = self.stig_sens_nm_per_ma * (i_stig_y - self.state.parasitic_stig_y_ma)

        # Quadrature sum for total probe spot size (nm)
        spot_x = np.sqrt(
            self.source_size_nm ** 2
            + d_defocus ** 2
            + d_spherical ** 2
            + d_chromatic ** 2
            + d_ast_x ** 2
        )
        spot_y = np.sqrt(
            self.source_size_nm ** 2
            + d_defocus ** 2
            + d_spherical ** 2
            + d_chromatic ** 2
            + d_ast_y ** 2
        )

        # 4. Deflector coordinate mapping with rotation and cubic pincushion distortion
        theta = self.state.deflector_rotation_rad
        gx = self.state.deflector_gain_x * v_def_x
        gy = self.state.deflector_gain_y * v_def_y

        pos_x = gx * np.cos(theta) - gy * np.sin(theta) + 1.2e-5 * (gx ** 3)
        pos_y = gx * np.sin(theta) + gy * np.cos(theta) + 1.2e-5 * (gy ** 3)

        if add_noise:
            spot_x = spot_x + self.rng.normal(0.0, self.noise_std_nm, size=spot_x.shape)
            spot_y = spot_y + self.rng.normal(0.0, self.noise_std_nm, size=spot_y.shape)
            pos_x = pos_x + self.rng.normal(0.0, self.pos_noise_std_um, size=pos_x.shape)
            pos_y = pos_y + self.rng.normal(0.0, self.pos_noise_std_um, size=pos_y.shape)

        return {
            "spot_size_x_nm": spot_x,
            "spot_size_y_nm": spot_y,
            "beam_pos_x_um": pos_x,
            "beam_pos_y_um": pos_y,
        }


if __name__ == "__main__":
    # Quick sanity check: compare uncalibrated vs. well-tuned column
    sim = EBeamColumnSimulator()
    uncalibrated = sim.measure(i_obj=412.0, i_stig_x=0.0, i_stig_y=0.0, add_noise=False)
    calibrated = sim.measure(i_obj=420.0, i_stig_x=3.2, i_stig_y=-2.1, add_noise=False)

    print(f"Uncalibrated spot size (X, Y): {uncalibrated['spot_size_x_nm']:.2f} nm, {uncalibrated['spot_size_y_nm']:.2f} nm")
    print(f"Calibrated spot size   (X, Y): {calibrated['spot_size_x_nm']:.2f} nm, {calibrated['spot_size_y_nm']:.2f} nm")