# -*- coding: utf-8 -*-
"""
Created on 8/19/2026

@author: jtgrasb
"""

# Python libraries
import numpy as np
import matplotlib.pyplot as plt
import os
from functools import reduce
from time import time

# Custom modules
from multibody import MbdSystem            # new public class
import multibody as mbd
import wecopttool as wot

# Linearization
from multibody.linearization.linearization_main import LinearizationManager
from multibody.linearization.hydro_linear_mckf import HydroLinearMCKF

############ Example to import ############
import WaveBot_M4E_inputs as ex
from WaveBot_hydro_inputs import waves, body_inputs, m0, J0

############### Beginning of the multibody simulation ###############
# 1- Initialize the Multibody system
MBDsys = MbdSystem.from_example(ex)

# 2 - Define initial numerical values
# Define the equilibrium position
q0 = (MBDsys.ic - ex.ic)[:len(MBDsys.Q)] # This may need revision but avoids user thinking too much
# Define initial conditions w.r.t equilibrium
mainNumVars = ex.ic.copy()

print('Finished initialization')

# 3 - Linearization manager
LinManager = LinearizationManager(MBDsys, q0, mainNumVars, m0, J0, print_sym_matrices=False)

# Instanciate and register hydrodynamics adapter
hydroAdapter = HydroLinearMCKF(
        MBDsys,
        (mainNumVars, m0, J0),
        is_2D=False,
        omega_r=waves.omega.values,
        body_inputs=body_inputs,
        wave_amplitude=waves.attrs['Amplitude (m)'],
        equilibrium_pos=q0,
        )

LinManager.register(hydroAdapter)

# --- Compile and integrate ---
# NOTE: if you want to integrate, first provide an operating point i.e. operating omega and ramp time (Default ramp is 0.0). 
omega0 = waves.omega.values[0]
LinManager.compile_operating_point(omega0, eq_tol=1e-6, eq_mode="warn", ramp_T=100.0)

result = LinManager.integrate_linear_system(
    tspan=ex.tspan,
    dt=ex.TimeStep,
    method="RK45",
    solver_opts={'rtol': 1e-6, 'atol': 1e-9},
    T_ramp=100.0,
)

com_positions, com_velocities, angle_positions, _ = mbd.evaluate_trajectories(MBDsys, result, mainNumVars.copy())

# Create figure with 3 subplots for x, z, and pitch DOF
fig, axes = plt.subplots(3, 1, figsize=(10, 8))

# Plot x position for all bodies
for body in range(len(com_positions[1])):
        axes[0].plot(result.t, com_positions[:, body, 0], label=f'Body {body+1}')
        axes[1].plot(result.t, com_positions[:, body, 1], label=f'Body {body+1}')
        axes[2].plot(result.t, angle_positions[:, body]*180/np.pi, label=f'Body {body+1}')

plt.show()