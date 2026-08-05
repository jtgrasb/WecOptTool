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
amplitude_vec = np.array([0.05, 0.1, 0.15, 0.2, 0.25], dtype=float)

added_thickness_vec = np.array([0.0, 0.01, 0.02, 0.04, 0.06], dtype=float)

stiffness_init_vals = [
    10.0,
    60.0,
    100.0,
    150.0,
    200.0,
    500.0,
]

base_flap_thickness_bottom = 0.04
base_flap_thickness_top = 0.10

overwrite_bem = False
wave_seed = 0
f1_factor = 1 / 1

run_data_dir = "run_data"
os.makedirs(run_data_dir, exist_ok=True)

controller_kwargs = dict(
    P=False,
    I=False,
    coupled=False,
    unstructured=True,
)

thickness_sweep_results = xr.Dataset(
    data_vars={
        "avg_power": (
            ("wavefreq", "amplitude", "added_thickness"),
            np.full(
                (len(wavefreq_vec), len(amplitude_vec), len(added_thickness_vec)),
                np.nan,
            ),
        ),
        "objective": (
            ("wavefreq", "amplitude", "added_thickness"),
            np.full(
                (len(wavefreq_vec), len(amplitude_vec), len(added_thickness_vec)),
                np.nan,
            ),
        ),
        "generated_power": (
            ("wavefreq", "amplitude", "added_thickness"),
            np.full(
                (len(wavefreq_vec), len(amplitude_vec), len(added_thickness_vec)),
                np.nan,
            ),
        ),
        "success": (
            ("wavefreq", "amplitude", "added_thickness"),
            np.full(
                (len(wavefreq_vec), len(amplitude_vec), len(added_thickness_vec)),
                False,
                dtype=bool,
            ),
        ),
        "flap_thickness_bottom": (
            ("added_thickness",),
            base_flap_thickness_bottom + added_thickness_vec,
        ),
        "flap_thickness_top": (
            ("added_thickness",),
            base_flap_thickness_top + added_thickness_vec,
        ),
    },
    coords={
        "wavefreq": wavefreq_vec,
        "amplitude": amplitude_vec,
        "added_thickness": added_thickness_vec,
    },
    attrs={
        "description": "Full-system FOSWEC flap-thickness sweep",
        "controller": "unstructured",
        "optimize_stiffness": 1,
        "fixed_stiffness": 0.0,
        "f1_factor": f1_factor,
    },
)

record_vars = [
    "opt_power",
    "elec_power_W",
    "mean_elec_power",
    
    "opt_stiffness_flap1",
    "opt_stiffness_flap2",
    "initial_stiffness_flap1_used",
    "initial_stiffness_flap2_used",
    "n_stiffness_attempts",
    "n_stiffness_successes",
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
    thickness_sweep_results[var] = (
        ("wavefreq", "amplitude", "added_thickness"),
        np.full(
            (len(wavefreq_vec), len(amplitude_vec), len(added_thickness_vec)),
            np.nan,
        ),
    )

full_results = OrderedDict()

def num_to_str(x):
    return f"{float(x):.6g}".replace("-", "m").replace(".", "p")


def thickness_bem_filename(wavefreq, f1, nfreq, depth, thickness_bottom, thickness_top):
    return (
        f"foswec_thickness"
        f"_wf_{num_to_str(wavefreq)}"
        f"_f1_{num_to_str(f1)}"
        f"_nf_{nfreq}"
        f"_depth_{num_to_str(depth)}"
        f"_tb_{num_to_str(thickness_bottom)}"
        f"_tt_{num_to_str(thickness_top)}"
        f".nc"
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
    f1 = wavefreq * f1_factor

    for added_thickness in added_thickness_vec:
        flap_thickness_bottom = base_flap_thickness_bottom + added_thickness
        flap_thickness_top = base_flap_thickness_top + added_thickness

        print(
            f"\nRunning setup: wavefreq={wavefreq:.4f} Hz, "
            f"T={wave_period:.3f} s, "
            f"added_thickness={added_thickness:g} m"
        )

        try:
            study_kwargs = dict(common_study_kwargs)

            study_kwargs.update(
            dict(flap_thickness_bottom=float(flap_thickness_bottom),
            flap_thickness_top=float(flap_thickness_top),
            
            # Enable stiffness optimization
            optimize_stiffness=True,fixed_stiffness=0.0, 
            ))

            study = FOSWECM4EStudyBase(
                wavefreq=float(wavefreq),
                f1=float(f1),
                amplitude=float(amplitude_vec[0]),
                flap1_mass_on_top=0.0,
                flap2_mass_on_top=0.0,
                **study_kwargs,
            )

            bem_file = thickness_bem_filename(
                wavefreq=wavefreq,
                f1=f1,
                nfreq=nfreq,
                depth=depth,
                thickness_bottom=flap_thickness_bottom,
                thickness_top=flap_thickness_top,
            )

            study.load_or_run_bem(
                bem_data_file=bem_file,
                overwrite=overwrite_bem,
            )
            study.setup_m4e_reduction()

        except Exception:
            logging.exception(
                f"Failed setup for wavefreq={wavefreq:g}, "
                f"added_thickness={added_thickness:g}"
            )
            continue

        for amplitude in amplitude_vec:
            loc = dict(
                wavefreq=wavefreq,
                amplitude=amplitude,
                added_thickness=added_thickness,
            )

            print(
                f"  Solving f={wavefreq:.3f} Hz, "
                f"A={amplitude:.3f} m, "
                f"added_thickness={added_thickness:g} m"
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

                print("summary keys:", list(summary.keys()))
                print("constraint metric keys:", list(cmet.keys()))

                avg_power = summary["avg_power"]
                objective = summary["objective"]

                thickness_sweep_results["avg_power"].loc[loc] = avg_power
                thickness_sweep_results["objective"].loc[loc] = objective
                thickness_sweep_results["generated_power"].loc[loc] = avg_power

                thickness_sweep_results["opt_power"].loc[loc] = summary["opt_power"]
                thickness_sweep_results["elec_power_W"].loc[loc] = -summary["opt_power"]
                thickness_sweep_results["mean_elec_power"].loc[loc] = summary["mean_elec_power"]

                thickness_sweep_results["opt_stiffness_flap1"].loc[loc] = summary[
                    "opt_stiffness_flap1"
                ]
                thickness_sweep_results["opt_stiffness_flap2"].loc[loc] = summary[
                    "opt_stiffness_flap2"
                ]

                thickness_sweep_results["initial_stiffness_flap1_used"].loc[loc] = summary[
		    "initial_stiffness_flap1_used"
		]
                thickness_sweep_results["initial_stiffness_flap2_used"].loc[loc] = summary[
		    "initial_stiffness_flap2_used"
		]
                thickness_sweep_results["n_stiffness_attempts"].loc[loc] = summary[
                    "n_stiffness_attempts"
                ]
                thickness_sweep_results["n_stiffness_successes"].loc[loc] = summary[
                    "n_stiffness_successes"
                ]
                thickness_sweep_results["final_objective_positive"].loc[loc] = float(
                    summary["final_objective_positive"]
                )
                
		
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
                    print(f"Recording cmet[{var}]")
                    thickness_sweep_results[var].loc[loc] = cmet[var]

                key = (
                    float(wavefreq),
                    float(amplitude),
                    float(added_thickness),
                )

                if store_full_results:
                    full_results[key] = case

                thickness_sweep_results["success"].loc[loc] = True

                print(
                    f"    SUCCESS: avg_power={avg_power:.6g}, "
                    f"objective={objective:.6g}, "
                    f"initial_k1={summary['initial_stiffness_flap1_used']:g}, "
                    f"initial_k2={summary['initial_stiffness_flap2_used']:g}, "
                    f"opt_stiffness_1={summary['opt_stiffness_flap1']:.6g}, "
                    f"opt_stiffness_2={summary['opt_stiffness_flap2']:.6g}"
                )

            except Exception:
                logging.exception(
                    f"Failed solve/record: wavefreq={wavefreq:g}, "
                    f"amplitude={amplitude:g}, "
                    f"added_thickness={added_thickness:g}"
                )
                thickness_sweep_results["success"].loc[loc] = False
                
def clean_attrs_for_netcdf(ds):
    ds = ds.copy()

    for key, value in list(ds.attrs.items()):
        if isinstance(value, (bool, np.bool_)):
            ds.attrs[key] = int(value)
        elif value is None:
            ds.attrs[key] = "None"
        elif isinstance(value, (list, tuple, dict)):
            ds.attrs[key] = str(value)

    for var in ds.data_vars:
        for key, value in list(ds[var].attrs.items()):
            if isinstance(value, (bool, np.bool_)):
                ds[var].attrs[key] = int(value)
            elif value is None:
                ds[var].attrs[key] = "None"
            elif isinstance(value, (list, tuple, dict)):
                ds[var].attrs[key] = str(value)

    return ds
  
print("Success count before save:", thickness_sweep_results["success"].sum().values)
print("Finite elec_power_W count before save:", np.isfinite(thickness_sweep_results["elec_power_W"]).sum().values)

thickness_sweep_results_clean = clean_attrs_for_netcdf(thickness_sweep_results)

out_file = "run_data/foswec_added_thickness_sweep_results.nc"

# Optional: remove old file first to avoid confusion
if os.path.exists(out_file):
    os.remove(out_file)

thickness_sweep_results_clean.to_netcdf(out_file)

print(f"Saved results to: {out_file}")
