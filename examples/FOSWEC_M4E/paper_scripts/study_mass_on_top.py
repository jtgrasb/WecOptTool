from collections import OrderedDict
import numpy as np
import xarray as xr
import logging
import os
import matplotlib.pyplot as plt
import itertools

import FOSWEC_M4E_inputs
from foswec_m4e_base import FOSWECM4EStudyBase

logging.getLogger().setLevel(logging.INFO)


nfreq = 10
depth = 2.0
overwrite_bem = False
store_full_results = False

run_data_dir = "run_data"
os.makedirs(run_data_dir, exist_ok=True)

bem_data_dir = "bem_data"

common_study_kwargs = dict(
    m4e_inputs_module=FOSWEC_M4E_inputs,

    nfreq=nfreq,
    depth=depth,
    phase=0.0,
    wavedir=0.0,
    wave_seed=0,

    bem_data_dir=bem_data_dir,

    # Mesh
    mesh_size_factor=0.3,
    flap_mesh_size=0.1,

    # Geometry
    flap_thickness_bottom=0.04,
    flap_thickness_top=0.1,
    flap_center_distance_apart=1.44,
    flap_width=0.76,
    flap_height=0.58,
    flap_draft=0.59,
    cg_height_above_hinge=0.17,

    platform_frame_length=1.44,
    platform_frame_thickness=0.05,
    platform_frame_width=1.06,
    platform_frame_top_depth=0.53 + 0.1,
    platform_cg=(0.0, 0.0, -0.8),

    full_width=1.63,
    columns_radius=0.1,
    columns_draft=0.53 + 0.1 + 0.4,

    daq_length=0.72,
    daq_width=0.6,
    daq_height=0.2,
    daq_top_depth=0.53 + 0.1 + 0.05,

    # Rigid-body properties
    m1=189.8,
    J1=30.0,
    m2=23.1,
    J2=1.19,

    # PTO/drivetrain values from your current class setup
    gear_ratio=3.75,
    torque_constant=1.021,
    winding_resistance=1.028,
    winding_inductance=0.0,
    drivetrain_inertia=0.05,
    drivetrain_friction_aft=3.27 / 3.75**2,
    drivetrain_friction_bow=2.7 / 3.75**2,
    drivetrain_stiffness=0.0,

    # Flap pitch constraints
    max_flap_pitch_deg=30.0,
    nsubsteps_constraint=4,
    use_flap_pitch_constraints=True,

    # Stabilization
    mooring_stiffness_full=np.diag([8e3, 2e5, 2e5, 0, 0, 0, 0, 0, 0]),
    mooring_damping_full=np.diag([8e2, 1e4, 1e4, 0, 0, 0, 0, 0, 0]),
    flap_linear_damping_full=np.diag([0, 0, 0, 0, 0, 10, 0, 0, 10]),

    # Optimization
    scale_x_wec=1e1,
    scale_x_opt=1e0,
    scale_obj=1e1,
    optim_maxiter=300,
    nsubsteps_post=10,
    pto_gains_init=-1e2,

    log_level=logging.INFO,
)

# ---------------------------------------------------------------------
# User inputs
# ---------------------------------------------------------------------
wavefreq_vec = np.array([1/6, 1/5, 1/4, 1/3, 1/2, 1/1], dtype=float)
amplitude_vec = np.array([0.05,0.1, 0.15, 0.2, 0.25], dtype=float)
mass_vec = np.array([0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=float)

stiffness_init_vals = [
    10.0,
    60.0,
    100.0,
    150.0,
    200.0,
    500.0,
]

controller_kwargs = dict(
    P=False,
    I=False,
    coupled=False,
    unstructured=True,
)

overwrite_bem = False
wave_seed = 0

# If you want f1 tied to peak wave frequency like before:
f1_factor = 1 / 1

mass_sweep_results = xr.Dataset(
    data_vars={
        "avg_power": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "objective": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "generated_power": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "success": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), False, dtype=bool),
        ),
        "flap1_total_mass": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "flap1_cg_height_above_hinge": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "flap1_inertia_cg": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "flap2_total_mass": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "flap2_cg_height_above_hinge": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
        "flap2_inertia_cg": (
            ("wavefreq", "amplitude", "mass_on_top"),
            np.full((len(wavefreq_vec), len(amplitude_vec), len(mass_vec)), np.nan),
        ),
    },
    coords={
        "wavefreq": wavefreq_vec,
        "amplitude": amplitude_vec,
        "mass_on_top": mass_vec,
    },
    attrs={
        "description": "Full-system FOSWEC added-top-mass sweep",
        "controller": "unstructured",
        "f1_factor": f1_factor,
    },
)

full_results = OrderedDict()

record_vars = [
    "opt_power",
    "elec_power_W",
    "mean_elec_power",
    "opt_stiffness_flap1",
    "opt_stiffness_flap2",

    "initial_stiffness_flap1_used",
    "initial_stiffness_flap2_used",
    "n_stiffness_attempts",
    "final_objective_positive",

    "flap1_max_abs_pos",
    "flap2_max_abs_pos",
    "flap1_peak_rotation_utilization",
    "flap2_peak_rotation_utilization",

    "flap1_peak_torque_utilization",
    "flap2_peak_torque_utilization",
    "flap1_rms_torque_utilization",
    "flap2_rms_torque_utilization",
]

for var in record_vars:
    mass_sweep_results[var] = (
        ("wavefreq", "amplitude", "mass_on_top"),
        np.full(
            (len(wavefreq_vec), len(amplitude_vec), len(mass_vec)),
            np.nan,
        ),
    )
    
def make_x_opt_0_for_stiffness_pair(
    study,
    controller_kwargs,
    stiffness_init_flap1,
    stiffness_init_flap2,
):
    nstate_opt, _ = study.build_controller_bounds(**controller_kwargs)

    x_opt_0 = np.zeros(nstate_opt)

    # Last two entries are stiffness for flap 1 and flap 2
    x_opt_0[-2] = float(stiffness_init_flap1)
    x_opt_0[-1] = float(stiffness_init_flap2)

    return x_opt_0


def solve_with_stiffness_pair_sweep(
    study,
    controller_kwargs,
    stiffness_init_values,
):
    best_case = None
    best_objective = np.inf
    best_k1_init = None
    best_k2_init = None

    n_attempts = 0
    n_successes = 0
    last_exception = None

    for k1_init, k2_init in itertools.product(
        stiffness_init_values,
        stiffness_init_values,
    ):
        n_attempts += 1

        print(f"    Trying initial stiffnesses: k1={k1_init:g}, k2={k2_init:g}")

        try:
            x_opt_0 = make_x_opt_0_for_stiffness_pair(
                study=study,
                controller_kwargs=controller_kwargs,
                stiffness_init_flap1=k1_init,
                stiffness_init_flap2=k2_init,
            )

            case = study.solve_controller_case(
                **controller_kwargs,
                x_opt_0_override=x_opt_0,
            )

            objective = float(case["summary"]["objective"])
            n_successes += 1

            print(
                f"      objective = {objective:.6g}, "
                f"elec_power = {-objective:.6g}, "
                f"opt_k1 = {case['summary']['opt_stiffness_flap1']:.6g}, "
                f"opt_k2 = {case['summary']['opt_stiffness_flap2']:.6g}"
            )

            if np.isfinite(objective) and objective < best_objective:
                best_objective = objective
                best_case = case
                best_k1_init = k1_init
                best_k2_init = k2_init

        except Exception as exc:
            last_exception = exc
            logging.exception(
                f"    Solve failed for initial stiffnesses: "
                f"k1={k1_init:g}, k2={k2_init:g}"
            )

    if best_case is None:
        raise RuntimeError(
            "All stiffness-initialization combinations failed."
        ) from last_exception

    best_case["summary"]["initial_stiffness_flap1_used"] = float(best_k1_init)
    best_case["summary"]["initial_stiffness_flap2_used"] = float(best_k2_init)
    best_case["summary"]["n_stiffness_attempts"] = int(n_attempts)
    best_case["summary"]["n_stiffness_successes"] = int(n_successes)
    best_case["summary"]["final_objective_positive"] = bool(best_objective > 0.0)

    print(
        f"    Best initial stiffnesses: "
        f"k1={best_k1_init:g}, k2={best_k2_init:g}, "
        f"best objective={best_objective:.6g}, "
        f"best elec_power={-best_objective:.6g}"
    )

    return best_case

for wavefreq in wavefreq_vec:
    wave_period = 1.0 / wavefreq

    for mass_on_top in mass_vec:
        print(
            f"\nRunning setup: wavefreq={wavefreq:.4f} Hz, "
            f"T={wave_period:.3f} s, mass_on_top={mass_on_top:g} kg"
        )

        try:
            study = FOSWECM4EStudyBase(
                wavefreq=float(wavefreq),
                f1=float(wavefreq * f1_factor),
                wave_type="regular",
                amplitude=float(amplitude_vec[0]),
                flap1_mass_on_top=float(mass_on_top),
                flap2_mass_on_top=float(mass_on_top),
                **common_study_kwargs,
            )

            study.load_or_run_bem(overwrite=overwrite_bem)
            study.setup_m4e_reduction()

        except Exception:
            logging.exception(
                f"Failed setup for wavefreq={wavefreq:g}, mass={mass_on_top:g}"
            )
            continue

        for amplitude in amplitude_vec:
            loc = dict(
                wavefreq=wavefreq,
                amplitude=amplitude,
                mass_on_top=mass_on_top,
            )

            print(
            	f"Solving f={wavefreq:.3f} Hz, "
                f"A={amplitude:.3f} m, "
                f"mass={mass_on_top:g} kg"
            )

            try:
                study.set_wave_amplitude(
                    amplitude=float(amplitude),
                    seed=wave_seed,
                )

                case = solve_with_stiffness_pair_sweep(
		    study=study,
		    controller_kwargs=controller_kwargs,
		    stiffness_init_values=stiffness_init_vals,
		)

                summary = case["summary"]
                cmet = case["constraint_metrics"]

                
                mass_sweep_results["opt_power"].loc[loc] = summary["opt_power"]
                mass_sweep_results["elec_power_W"].loc[loc] = -summary["opt_power"]
                mass_sweep_results["mean_elec_power"].loc[loc] = summary["mean_elec_power"]
                mass_sweep_results["opt_stiffness_flap1"].loc[loc] = summary["opt_stiffness_flap1"]
                mass_sweep_results["opt_stiffness_flap2"].loc[loc] = summary["opt_stiffness_flap2"]
                
                mass_sweep_results["initial_stiffness_flap1_used"].loc[loc] = summary["initial_stiffness_flap1_used"]
                mass_sweep_results["initial_stiffness_flap2_used"].loc[loc] = summary["initial_stiffness_flap2_used"]
                mass_sweep_results["n_stiffness_attempts"].loc[loc] = summary["n_stiffness_attempts"]
                mass_sweep_results["final_objective_positive"].loc[loc] = float(summary["final_objective_positive"])

                for var in [
                    "flap1_max_abs_pos",
                    "flap2_max_abs_pos",
                    "flap1_peak_rotation_utilization",
                    "flap2_peak_rotation_utilization",
                    "flap1_peak_torque_utilization",
                    "flap2_peak_torque_utilization",
                    "flap1_rms_torque_utilization",
                    "flap2_rms_torque_utilization",
                ]:
                    mass_sweep_results[var].loc[loc] = cmet[var]

                avg_power = summary["avg_power"]
                objective = summary["objective"]

                mass_sweep_results["avg_power"].loc[loc] = avg_power
                mass_sweep_results["objective"].loc[loc] = objective

                # If avg_power already has your desired sign, use this.
                mass_sweep_results["generated_power"].loc[loc] = avg_power

                mass_sweep_results["success"].loc[loc] = True

                mass_sweep_results["flap1_total_mass"].loc[loc] = summary["flap1_total_mass"]
                mass_sweep_results["flap1_cg_height_above_hinge"].loc[loc] = summary["flap1_cg_height_above_hinge"]
                mass_sweep_results["flap1_inertia_cg"].loc[loc] = summary["flap1_inertia_cg"]

                mass_sweep_results["flap2_total_mass"].loc[loc] = summary["flap2_total_mass"]
                mass_sweep_results["flap2_cg_height_above_hinge"].loc[loc] = summary["flap2_cg_height_above_hinge"]
                mass_sweep_results["flap2_inertia_cg"].loc[loc] = summary["flap2_inertia_cg"]

                key = (
                    float(wavefreq),
                    float(amplitude),
                    float(mass_on_top),
                )
                full_results[key] = case

                print(
		    f"    avg_power = {avg_power:.6g}, "
		    f"objective = {objective:.6g}, "
		    f"initial_k1={summary['initial_stiffness_flap1_used']:g}, "
		    f"initial_k2={summary['initial_stiffness_flap2_used']:g}, "
		    f"opt_stiffness_1 = {summary['opt_stiffness_flap1']:.6g}"
		    f"opt_stiffness_2 = {summary['opt_stiffness_flap2']:.6g}"
		)

            except Exception:
                logging.exception(
                    f"Failed solve: wavefreq={wavefreq:g}, "
                    f"amplitude={amplitude:g}, mass={mass_on_top:g}"
                )
                mass_sweep_results["success"].loc[loc] = False

mass_sweep_results.to_netcdf("run_data/foswec_mass_on_top_sweep_results.nc")
