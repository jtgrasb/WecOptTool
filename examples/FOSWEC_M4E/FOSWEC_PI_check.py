import os
import gc
import logging
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import FOSWEC_M4E_inputs

# Replace this import with your actual class-script filename.
# For example, if your class is saved in "foswec_m4e_study.py":
from foswec_m4e_base import FOSWECM4EStudyBase

logging.getLogger().setLevel(logging.INFO)

wave_periods = np.array([1, 3, 6], dtype=float)
amplitudes = np.array([0.5, 0.2], dtype=float)

controller_names = np.array(
    ["P", "P_coupled", "PI", "PI_coupled", "unstructured"],
    dtype=str,
)

nfreq = 60
depth = 2.0
overwrite_bem = False
store_full_results = False

run_data_dir = "run_data"
os.makedirs(run_data_dir, exist_ok=True)

results_base_name = "coupled_controller_res"
results_nc_file = os.path.join(run_data_dir, f"{results_base_name}.nc")
x_solution_file = os.path.join(run_data_dir, f"{results_base_name}_x_solutions.npz")

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
    scale_x_opt=1e-2,
    scale_obj=1.0,
    optim_maxiter=300,
    nsubsteps_post=40,
    pto_gains_init=1e2,

    log_level=logging.INFO,
)

wave_period = wave_periods[0]

wavefreq = 1.0 / wave_period

logging.info("=" * 80)
logging.info(f"Starting wave period T = {wave_period:.3f} s")
logging.info(f"Peak frequency fp = {wavefreq:.6f} Hz")
logging.info("=" * 80)

study = FOSWECM4EStudyBase(
    wavefreq=wavefreq,
    amplitude=float(amplitudes[0]),
    **common_study_kwargs,
)

study.load_or_run_bem(overwrite=overwrite_bem)
study.setup_m4e_reduction()

print("\nPeriod setup complete:")
print(f"  T = {wave_period:.3f} s")
print(f"  wavefreq = {wavefreq:.6f} Hz")
print(f"  f1 = {study.f1:.6f} Hz")
print(f"  nfreq = {study.nfreq}")
print(f"  len(study.waves.freq) = {len(study.waves.freq)}")
print(f"  len(study.bem_data.omega) = {len(study.bem_data.omega)}")
print(f"  impedance_reduced shape = {study.impedance_reduced.shape}")


amplitude = amplitudes[0]

logging.info("-" * 80)
logging.info(
    f"Wave condition: T = {wave_period:.3f} s, A = {amplitude:.3f} m"
)
logging.info("-" * 80)

study.set_wave_amplitude(
    amplitude=float(amplitude),
    seed=0,
)

# First solve PI controller
pi_case = study.solve_controller_case(
    P=True,
    I=True,
    coupled=False,
    unstructured=False,
)

# Extract WEC state from PI solution
# Assumes scipy result vector begins with WEC states.
x_pi_full = np.asarray(pi_case["results"][0].x)
x_wec_pi = x_pi_full[: pi_case["wec"].nstate_wec]

# Now solve unstructured controller using PI WEC state as initial condition
unstruct_case = study.solve_controller_case(
    P=False,
    I=False,
    coupled=False,
    unstructured=True,
    x_wec_0_override=x_wec_pi,
)

print("PI avg power:", pi_case["summary"]["avg_power"])
print("Unstructured avg power:", unstruct_case["summary"]["avg_power"])
