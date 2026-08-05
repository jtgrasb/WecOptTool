import os
import copy
import logging
import numpy as np
import xarray as xr
import jax.numpy as jnp

import pygmsh
import capytaine as cpt
import wecopttool as wot


class FOSWECFlapStudyBase:
    """
    Base class for running added-top-mass studies on a single FOSWEC flap.

    This class encapsulates:
      - geometry creation
      - Capytaine floating body setup
      - BEM execution
      - PTO definition
      - optimization constraints
      - solving and post-processing

    Designed to be imported into a notebook for rapid parameter sweeps.
    """

    def __init__(
        self,
        wavefreq=1 / 8,
        nfreq=10,
        amplitude=0.5,
        phase=0.0,
        wavedir=0.0,
        depth=2.0,
        flap_thickness_bottom=0.04,
        flap_thickness_top=0.1,
        flap_center_distance_apart=1.44,
        flap_width=0.76,
        flap_height=0.58,
        flap_draft=0.59,
        cg_height_above_hinge=0.17,
        flap_mass=23.1,
        flap_inertia_cg=1.19,
        gear_ratio=3.75,
        torque_constant=1.021,
        winding_resistance=1.028,
        winding_inductance=0.0,
        drivetrain_inertia=0.05,
        drivetrain_friction=None,
        flap_side="aft",
        #drivetrain_friction_aft=3.27 / 3.75**2,
        #drivetrain_friction_bow=2.7 / 3.75**2,
        drivetrain_stiffness=0.0,
        max_torque_drive=40.0,
        max_rms_torque_drive=20.0,
        max_pos= jnp.pi / 6,
        nsubsteps_constraint=4,
        nsubsteps_post=5,
        optim_maxiter=200,
        scale_x_wec=1.0,
        scale_x_opt=1e-1,
        scale_obj=1.0,
        optimize_stiffness=True,
        stiffness_init_factor=6.0,
        fixed_stiffness=0.0,
        log_level=logging.WARNING,
    ):
        logging.getLogger().setLevel(log_level)

        # Wave / frequency
        self.wavefreq = wavefreq
        self.f1 = wavefreq
        self.nfreq = nfreq
        self.freq = wot.frequency(self.f1, self.nfreq, False)
        self.amplitude = amplitude
        self.phase = phase
        self.wavedir = wavedir
        self.waves = wot.waves.regular_wave(
            self.f1, self.nfreq, self.wavefreq, self.amplitude, self.phase, self.wavedir
        )
        self.depth = depth

        self.bem_data = None

        # Geometry / body
        self.flap_thickness_bottom = flap_thickness_bottom
        self.flap_thickness_top = flap_thickness_top
        self.flap_center_distance_apart = flap_center_distance_apart
        self.flap_width = flap_width
        self.flap_height = flap_height
        self.flap_draft = flap_draft
        self.cg_height_above_hinge = cg_height_above_hinge

        # Baseline rigid body properties
        self.flap_mass = flap_mass
        self.flap_inertia_cg = flap_inertia_cg

        # PTO / drivetrain
        self.gear_ratio = gear_ratio
        self.torque_constant = torque_constant
        self.winding_resistance = winding_resistance
        self.winding_inductance = winding_inductance
        self.drivetrain_inertia = drivetrain_inertia
        self.drivetrain_friction = drivetrain_friction
        self.drivetrain_stiffness = drivetrain_stiffness

        self.flap_side = flap_side

        if drivetrain_friction is None:
            if flap_side == "aft":
                drivetrain_friction = 3.27 / gear_ratio**2
            elif flap_side == "bow":
                drivetrain_friction = 2.7 / gear_ratio**2
            else:
                raise ValueError("flap_side must be 'aft' or 'bow'.")

        self.drivetrain_friction = drivetrain_friction

        # Constraints
        self.max_torque = max_torque_drive * torque_constant
        self.max_rms_torque = max_rms_torque_drive * torque_constant
        self.max_pos = max_pos
        self.nsubsteps_constraint = nsubsteps_constraint
        self.nsubsteps_post = nsubsteps_post

        # Optimization
        self.obj_fun = None
        self.optim_options = {"maxiter": optim_maxiter, "disp": False}
        self.scale_x_wec = scale_x_wec
        self.scale_x_opt = scale_x_opt
        self.scale_obj = scale_obj
        self.stiffness_init_factor = stiffness_init_factor
        self.optimize_stiffness = optimize_stiffness
        self.fixed_stiffness = fixed_stiffness

        # Build reusable pieces
        self.base_mesh = self._build_mesh()
        self.base_body = self._build_body(
            total_mass=self.flap_mass,
            cg_height_above_hinge=self.cg_height_above_hinge,
            inertia_about_hinge=self._baseline_inertia_hinge(),
        )

        # PTO depends on omega; use baseline BEM once or rebuild per case
        self.pto = None
        self.f_add = None
        self.constraints = None

    # ------------------------------------------------------------------
    # Geometry / body construction
    # ------------------------------------------------------------------
    def _build_mesh(self):
        with pygmsh.geo.Geometry() as geom:
            flap_poly = geom.add_polygon(
                [
                    [-self.flap_thickness_bottom / 2, -self.flap_width / 2, -self.flap_draft],
                    [ self.flap_thickness_bottom / 2, -self.flap_width / 2, -self.flap_draft],
                    [ self.flap_thickness_top / 2,    -self.flap_width / 2, self.flap_height - self.flap_draft],
                    [-self.flap_thickness_top / 2,    -self.flap_width / 2, self.flap_height - self.flap_draft],
                ],
                mesh_size=0.1,
            )
            geom.extrude(flap_poly, [0, self.flap_width, 0])
            mesh = geom.generate_mesh()
        return mesh

    def _baseline_inertia_hinge(self):
        return self.flap_inertia_cg + self.flap_mass * self.cg_height_above_hinge**2

    def _build_body(self, total_mass, cg_height_above_hinge, inertia_about_hinge):
        center_of_mass = [0, 0, -self.flap_draft + cg_height_above_hinge]

        body = cpt.FloatingBody(
            mesh=self.base_mesh,
            name="flap1",
            center_of_mass=center_of_mass,
        )
        body.rotation_center = (0, 0, -self.flap_draft)
        body.add_rotation_dof(name="Pitch")

        body.mass = total_mass

        inertia_matrix = xr.DataArray(
            data=np.asarray(np.diag([inertia_about_hinge])),
            dims=["influenced_dof", "radiating_dof"],
            coords={
                "influenced_dof": list(body.dofs),
                "radiating_dof": list(body.dofs),
            },
            name="inertia_matrix",
        )
        body.inertia_matrix = inertia_matrix
        body.hydrostatic_stiffness = body.immersed_part().compute_hydrostatic_stiffness()
        return body

    # ------------------------------------------------------------------
    # Mass-property updates
    # ------------------------------------------------------------------
    def compute_mass_properties(self, mass_on_top):
        """
        Compute updated rigid-body properties when a point mass is added at the top of the flap.

        Assumes the added mass is located at flap_height above the hinge.
        """
        total_mass = self.flap_mass + mass_on_top

        cg_new = (
            self.cg_height_above_hinge * self.flap_mass
            + mass_on_top * self.flap_height
        ) / total_mass

        # Baseline hinge inertia
        I_hinge_base = self.flap_inertia_cg + self.flap_mass * self.cg_height_above_hinge**2

        # Added top mass contribution about hinge
        I_hinge_added = mass_on_top * self.flap_height**2

        I_hinge_new = I_hinge_base + I_hinge_added

        # Convert combined hinge inertia to combined CG inertia
        I_cg_new = I_hinge_new - total_mass * cg_new**2

        return {
            "mass_on_top": mass_on_top,
            "cg_height": cg_new,
            "total_mass": total_mass,
            "inertia_cg": I_cg_new,
            "inertia_hinge": I_hinge_new,
        }

    def build_body_for_mass(self, mass_on_top):
        props = self.compute_mass_properties(mass_on_top)
        body = self._build_body(
            total_mass=props["total_mass"],
            cg_height_above_hinge=props["cg_height"],
            inertia_about_hinge=props["inertia_hinge"],
        )
        return body, props

    # ------------------------------------------------------------------
    # BEM / PTO / constraints
    # ------------------------------------------------------------------
    def load_or_run_bem(self, bem_data_file="foswec_1_flap_WOT.nc", overwrite=False):
        """
        Load a cached BEM file if present, otherwise run BEM on the baseline body and save it.
        """
        if (not overwrite) and os.path.exists(bem_data_file):
            bem_data = wot.read_netcdf(bem_data_file)
        else:
            bem_data = wot.run_bem(self.base_body, self.freq, depth=self.depth)
            bem_data["radiating_dof"] = bem_data["radiating_dof"].astype(str)
            bem_data["influenced_dof"] = bem_data["influenced_dof"].astype(str)
            wot.write_netcdf(bem_data_file, bem_data)

        self.bem_data = bem_data
        return bem_data

    def set_bem_data(self, bem_data):
        """
        Manually provide an already-loaded BEM dataset.
        """
        self.bem_data = bem_data

    def build_pto(self, omega):
        drivetrain_impedance = (
            1j * omega * self.drivetrain_inertia
            + self.drivetrain_friction
            + 1 / (1j * omega) * self.drivetrain_stiffness
        )

        winding_impedance = self.winding_resistance + 1j * omega * self.winding_inductance

        pto_impedance_11 = -1 * self.gear_ratio**2 * drivetrain_impedance
        off_diag = np.sqrt(3.0 / 2.0) * self.torque_constant * self.gear_ratio
        pto_impedance_12 = -1 * (off_diag + 0j) * np.ones(omega.shape)
        pto_impedance_21 = -1 * (off_diag + 0j) * np.ones(omega.shape)
        pto_impedance_22 = winding_impedance

        pto_impedance = np.array(
            [
                [pto_impedance_11, pto_impedance_12],
                [pto_impedance_21, pto_impedance_22],
            ]
        )

        ndof = 1
        name = ["flap_pitch"]
        kinematics = np.eye(ndof)
        controller = wot.controllers.unstructured_controller()
        loss = None

        self.pto = wot.pto.PTO(ndof, kinematics, controller, pto_impedance, loss, name)
        self.obj_fun = self.pto.average_power
        return self.pto

    def build_forcing_functions(self):
        def drivetrain_stiffness_torque(wec, x_wec, x_opt, wave, nsubsteps=1):
            pos = wec.vec_to_dofmat(x_wec)
            time_matrix = wec.time_mat_nsubsteps(nsubsteps)

            if self.optimize_stiffness:
                k_dt = x_opt[-1]
            else:
                k_dt = self.fixed_stiffness

            drivetrain_spring = -k_dt * pos
            spring_torque = jnp.dot(time_matrix, drivetrain_spring)
            return spring_torque

        self.f_add = {
            "PTO": self.pto.force_on_wec,
            "drivetrain_stiffness": drivetrain_stiffness_torque,
        }
        return self.f_add

    def build_constraints(self):
        pto = self.pto
        gear_ratio = self.gear_ratio
        max_torque = self.max_torque
        max_rms_torque = self.max_rms_torque
        max_pos = self.max_pos
        nsubsteps = self.nsubsteps_constraint

        def const_motor_torque(wec, x_wec, x_opt, wave):
            pto_torque = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
            motor_torque = pto_torque / gear_ratio
            return max_torque - jnp.abs(motor_torque.flatten())

        def const_motor_rms_torque(wec, x_wec, x_opt, wave):
            pto_torque = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
            motor_torque = pto_torque / gear_ratio
            return max_rms_torque - jnp.sqrt(jnp.mean(motor_torque.flatten() ** 2) + 1e-12)

        def const_flap_rotation(wec, x_wec, x_opt, wave):
            pos = wec.vec_to_dofmat(x_wec)
            time_matrix = wec.time_mat_nsubsteps(nsubsteps)
            pos_td = jnp.dot(time_matrix, pos)
            return max_pos - jnp.abs(pos_td.flatten())

        self.constraints = [
            {"type": "ineq", "fun": const_motor_torque},
            {"type": "ineq", "fun": const_flap_rotation},
            {"type": "ineq", "fun": const_motor_rms_torque},
        ]
        return self.constraints

    # ------------------------------------------------------------------
    # Initial guesses / solve
    # ------------------------------------------------------------------
    def build_initial_conditions(self, mass_on_top):
        if self.optimize_stiffness:
            nstate_opt = 2 * self.nfreq + 1
            bounds_opt = tuple([(-1e10, 1e10)] * (nstate_opt - 1) + [(0, 1e10)])

            x_opt_0 = np.zeros((nstate_opt,))
            x_opt_0[-1] = self.stiffness_init_factor * mass_on_top
        else:
            nstate_opt = 2 * self.nfreq
            bounds_opt = tuple([(-1e10, 1e10)] * nstate_opt)

            x_opt_0 = np.zeros((nstate_opt,))

        x_wec_0 = np.repeat(np.squeeze(self.waves), 2)

        return {
            "nstate_opt": nstate_opt,
            "bounds_opt": bounds_opt,
            "x_wec_0": x_wec_0,
            "x_opt_0": x_opt_0,
        }

    def solve_case(self, mass_on_top, bem_data_file=None, use_bem_cache=False):
        body, props = self.build_body_for_mass(mass_on_top)
        if self.bem_data is None:
            raise ValueError(
                "No BEM data available. Run load_or_run_bem(...) or set_bem_data(...) first."
            )

        bem_data = self.bem_data.copy(deep=True)
                # Overwrite rigid-body properties for this mass case, if present in dataset
        if "inertia_matrix" in bem_data:
            bem_data["inertia_matrix"] = body.inertia_matrix

        if "hydrostatic_stiffness" in bem_data:
            bem_data["hydrostatic_stiffness"] = body.hydrostatic_stiffness

        if "mass" in bem_data:
            bem_data["mass"] = body.mass

        omega = bem_data.omega.values
        self.build_pto(omega)
        self.build_forcing_functions()
        self.build_constraints()

        wec = wot.WEC.from_bem(
            bem_data,
            constraints=self.constraints,
            friction=None,
            f_add=self.f_add,
        )

        init = self.build_initial_conditions(mass_on_top)

        results = wec.solve(
            self.waves,
            self.obj_fun,
            init["nstate_opt"],
            optim_options=self.optim_options,
            scale_x_wec=self.scale_x_wec,
            scale_x_opt=self.scale_x_opt,
            scale_obj=self.scale_obj,
            x_wec_0=init["x_wec_0"],
            x_opt_0=init["x_opt_0"],
            bounds_opt=init["bounds_opt"],
        )

        pto_fdom, pto_tdom = self.pto.post_process(
            wec, results, self.waves, nsubsteps=self.nsubsteps_post
        )
        wec_fdom, wec_tdom = wec.post_process(
            wec, results, self.waves, nsubsteps=self.nsubsteps_post
        )

        opt_average_power = results[0].fun
        if self.optimize_stiffness:
            opt_stiffness = float(results[0].x[-1])
        else:
            opt_stiffness = float(self.fixed_stiffness)
        summary = {
            **props,
            "hydrostatic_stiffness": body.hydrostatic_stiffness,
            "opt_stiffness": opt_stiffness,
            "fixed_stiffness": self.fixed_stiffness,
            "opt_power": opt_average_power,
            "max_pos": float(
                np.max(wec_tdom.sel(realization=0)["pos"].isel(influenced_dof=0).values)
            ),
            "max_abs_pos": float(
                np.max(np.abs(wec_tdom.sel(realization=0)["pos"].isel(influenced_dof=0).values))
            ),
            "max_torque": float(
                np.max(pto_tdom.sel(realization=0)["force"].isel(dof=0).values / self.gear_ratio)   #Thus torque from generator
            ),
            "max_abs_torque": float(
                np.max(np.abs(pto_tdom.sel(realization=0)["force"].isel(dof=0).values / self.gear_ratio))
            ),
            "mean_elec_power": float(
                np.mean(
                    pto_tdom["power"].sel(type="elec").isel(realization=0, dof=0).values
                )
            ),
            "mean_mech_power": float(
                np.mean(
                    pto_tdom["power"].sel(type="mech").isel(realization=0, dof=0).values
                )
            ),
        }
        constraint_metrics = self.evaluate_constraint_margins(
            {
                "wec_tdom": wec_tdom,
                "pto_tdom": pto_tdom,
            }
        )
        return {
            "mass_on_top": mass_on_top,
            "body": body,
            "bem_data": bem_data,
            "wec": wec,
            "pto": self.pto,
            "results": results,          # full optimization results
            "wec_fdom": wec_fdom,
            "wec_tdom": wec_tdom,
            "pto_fdom": pto_fdom,
            "pto_tdom": pto_tdom,
            "summary": summary,
            "constraint_metrics": constraint_metrics,
        }
    def evaluate_constraint_margins(self, case_data, realization=0, dof=0, influenced_dof=0):
        """
        Evaluate postprocessed constraint margins from time-domain results.

        Returns positive margins when constraints are satisfied,
        zero when active, and negative when violated.
        """
        wec_tdom = case_data["wec_tdom"]
        pto_tdom = case_data["pto_tdom"]

        pos = wec_tdom["pos"].isel(realization=realization, influenced_dof=influenced_dof).values
        force = pto_tdom["force"].isel(realization=realization, dof=dof).values
        motor_torque = force / self.gear_ratio

        max_abs_pos = float(np.max(np.abs(pos)))
        max_abs_motor_torque = float(np.max(np.abs(motor_torque)))
        rms_motor_torque = float(np.sqrt(np.mean(motor_torque**2)))

        peak_rotation_margin = float(self.max_pos - max_abs_pos)
        peak_torque_margin = float(self.max_torque - max_abs_motor_torque)
        rms_torque_margin = float(self.max_rms_torque - rms_motor_torque)

        peak_rotation_utilization = float(max_abs_pos / self.max_pos) if self.max_pos != 0 else np.nan
        peak_torque_utilization = float(max_abs_motor_torque / self.max_torque) if self.max_torque != 0 else np.nan
        rms_torque_utilization = float(rms_motor_torque / self.max_rms_torque) if self.max_rms_torque != 0 else np.nan

        return {
            "max_abs_pos": max_abs_pos,
            "max_abs_motor_torque": max_abs_motor_torque,
            "rms_motor_torque": rms_motor_torque,
            "peak_rotation_margin": peak_rotation_margin,
            "peak_torque_margin": peak_torque_margin,
            "rms_torque_margin": rms_torque_margin,
            "peak_rotation_utilization": peak_rotation_utilization,
            "peak_torque_utilization": peak_torque_utilization,
            "rms_torque_utilization": rms_torque_utilization,
            "all_constraints_satisfied": bool(
                (peak_rotation_margin >= 0)
                and (peak_torque_margin >= 0)
                and (rms_torque_margin >= 0)
            ),
            "most_active_constraint_margin": float(
                min(peak_rotation_margin, peak_torque_margin, rms_torque_margin)
            ),
        }
    def run_mass_sweep(self, mass_values, bem_cache_prefix=None, use_bem_cache=False):
        all_results = {}
        for mass in mass_values:
            cache_file = None
            if bem_cache_prefix is not None:
                cache_file = f"{bem_cache_prefix}_mass_{mass:g}.nc"

            logging.info(f"Running case for mass_on_top = {mass}")
            all_results[mass] = self.solve_case(
                mass_on_top=mass,
                bem_data_file=cache_file,
                use_bem_cache=use_bem_cache,
            )
        return all_results

    def results_table(self, results_dict):
        rows = []
        for mass, data in results_dict.items():
            row = copy.deepcopy(data["summary"])
            row["mass_on_top"] = mass
            rows.append(row)
        try:
            import pandas as pd
            return pd.DataFrame(rows).sort_values("mass_on_top").reset_index(drop=True)
        except ImportError:
            return rows