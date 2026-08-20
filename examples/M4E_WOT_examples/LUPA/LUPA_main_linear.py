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
# from Examples import spiderfloat as ex
import LUPA_M4E_inputs as ex
# from single_flap import M4E_inputs as ex
from LUPA_hydro_inputs import waves, body_inputs, m0, J0

class CustomMooring():
        name = "Mooring"

        def __init__(self):
                pass

        def frequency_domain_MCKF(self, omega):
                # Mooring stiffness/damping (spar only, at spar CG = [0,0,-1.3]).
                M = np.zeros((6,6))
                K = np.diag([2e3, 5e2, 5e2, 0, 0, 0])
                C = np.zeros((6,6))

                # Constant pretension forces
                rho       = 1025.0
                g         = 9.81
                V_float   = 0.2395912807595307
                m_float   = m0[1]  # m_torus
                V_spar    = 0.20265689154076563
                m_spar    = m0[0]  # m_spar
                floatCG_x = 0.01
                floatCB_x = -3.883345171911409e-06

                # assume mooring pretension is sufficient to keep the spar and float in equilibrium
                dc = np.zeros(6)
                dc[1] = -g * (rho * V_spar - m_spar)                      # spar heave pretension
                dc[4] = -g * (rho * V_float - m_float)                    # float heave pretension
                dc[5] = -g * rho * V_float * (floatCG_x - floatCB_x)      # float pitch pretension
                F = {'dc': dc, 'phasor': np.zeros(6)}
                
                return M, C, K, F

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
mooringAdapter = CustomMooring()

LinManager.register(hydroAdapter)
LinManager.register(mooringAdapter)

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