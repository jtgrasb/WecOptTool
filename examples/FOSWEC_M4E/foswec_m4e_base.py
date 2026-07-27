import os
import copy
import logging
import numpy as np
import xarray as xr
import jax.numpy as jnp

import pygmsh
import gmsh
import capytaine as cpt
import wecopttool as wot


class _ControllerBase:
    pass


class CoupledDampingController(_ControllerBase):
    """
    PTO controller supporting:
      - P only
      - PI
      - diagonal or fully coupled gains
    """

    def __init__(self, ndof_pto: int, P=True, I=False, coupled=False):
        self._ndof = ndof_pto
        self.P = P
        self.I = I
        self.coupled = coupled

    @property
    def ndof(self) -> int:
        return self._ndof

    @property
    def ngains(self) -> int:
        if self.coupled:
            ngains = self.ndof**2
        else:
            ngains = self.ndof

        if self.I:
            ngains = 2 * ngains

        return ngains

    @property
    def nstate(self) -> int:
        return self.ngains

    def force(self, pto, wec, x_wec, x_opt, wave=None, nsubsteps=1):
        pos_td = pto.position(wec, x_wec, x_opt, wave, nsubsteps)
        vel_td = pto.velocity(wec, x_wec, x_opt, wave, nsubsteps)

        if self.I and self.coupled:
            P = jnp.reshape(x_opt[: self.ndof**2], (self.ndof, self.ndof))
            I = jnp.reshape(x_opt[self.ndof**2 :], (self.ndof, self.ndof))
            force_td = jnp.dot(vel_td, P.T) + jnp.dot(pos_td, I.T)
        elif self.I:
            force_td = vel_td * x_opt[: self.ndof] + pos_td * x_opt[self.ndof :]
        elif self.coupled:
            P = jnp.reshape(x_opt, (self.ndof, self.ndof))
            force_td = jnp.dot(vel_td, P.T)
        else:
            force_td = vel_td * x_opt

        return force_td


class FOSWECM4EStudyBase:
    """
    Base class for reduced-coordinate FOSWEC M4E optimization studies.

    Encapsulates:
      - geometry creation
      - Capytaine body setup
      - BEM execution/load
      - M4E reduction
      - PTO/controller creation
      - additional forcing definitions
      - optimization solve and post-processing

    Intended for reuse in notebooks or scripts for controller and parameter sweeps.
    """

    def __init__(
        self,
        m4e_inputs_module,
        wavefreq=1 / 4,
        nfreq=10,
        amplitude=0.25,
        phase=0.0,
        wavedir=0.0,
        depth=2.0,
        mesh_size_factor=0.3,
        flap_mesh_size=0.1,
        # flap geometry
        flap_thickness_bottom=0.04,
        flap_thickness_top=0.1,
        flap_center_distance_apart=1.44,
        flap_width=0.76,
        flap_height=0.58,
        flap_draft=0.59,
        cg_height_above_hinge=0.17,
        # platform geometry
        platform_frame_length=1.44,
        platform_frame_thickness=0.05,
        platform_frame_width=1.06,
        platform_frame_top_depth=0.53 + 0.1,
        platform_cg=(0.0, 0.0, -0.8),
        # columns
        full_width=1.63,
        columns_radius=0.1,
        columns_draft=0.53 + 0.1 + 0.4,
        # DAQ box
        daq_length=0.72,
        daq_width=0.6,
        daq_height=0.2,
        daq_top_depth=0.53 + 0.1 + 0.05,
        # rigid-body properties
        m1=189.8,
        J1=30.0,
        m2=23.1,
        J2=1.19,
        # drivetrain / PTO
        gear_ratio=3.75,
        torque_constant=1.021,
        winding_resistance=1.028,
        winding_inductance=0.0,
        drivetrain_inertia=0.05,
        drivetrain_friction_aft = 3.27 / 3.75**2,
        drivetrain_friction_bow = 2.7 / 3.75**2,
        drivetrain_stiffness=0.0,
        # stabilization
        mooring_stiffness_full=None,
        mooring_damping_full=None,
        flap_linear_damping_full=None,
        # optimization
        scale_x_wec=1e1,
        scale_x_opt=1e-2,
        scale_obj=1.0,
        optim_maxiter=300,
        nsubsteps_post=10f,
        pto_gains_init=1e2,
        log_level=logging.INFO,
    ):
        logging.getLogger().setLevel(log_level)

        self.m4e_inputs_module = m4e_inputs_module

        # Wave/frequency
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

        # Mesh settings
        self.mesh_size_factor = mesh_size_factor
        self.flap_mesh_size = flap_mesh_size

        # Geometry
        self.flap_thickness_bottom = flap_thickness_bottom
        self.flap_thickness_top = flap_thickness_top
        self.flap_center_distance_apart = flap_center_distance_apart
        self.flap_width = flap_width
        self.flap_height = flap_height
        self.flap_draft = flap_draft
        self.cg_height_above_hinge = cg_height_above_hinge

        self.platform_frame_length = platform_frame_length
        self.platform_frame_thickness = platform_frame_thickness
        self.platform_frame_width = platform_frame_width
        self.platform_frame_top_depth = platform_frame_top_depth
        self.platform_cg = list(platform_cg)

        self.full_width = full_width
        self.columns_radius = columns_radius
        self.columns_draft = columns_draft

        self.daq_length = daq_length
        self.daq_width = daq_width
        self.daq_height = daq_height
        self.daq_top_depth = daq_top_depth

        # Derived geometry
        self.cutout_width = self.platform_frame_width - 2 * self.platform_frame_thickness
        self.cutout_length = self.platform_frame_length - 2 * self.platform_frame_thickness
        self.columns_xy = [
            self.flap_center_distance_apart / 2,
            self.full_width / 2 - self.columns_radius,
        ]

        self.flap1_cg = [
            -self.flap_center_distance_apart / 2,
            0.0,
            -self.flap_draft + self.cg_height_above_hinge,
        ]
        self.flap2_cg = [
            self.flap_center_distance_apart / 2,
            0.0,
            -self.flap_draft + self.cg_height_above_hinge,
        ]

        # Rigid-body properties
        self.m1 = m1
        self.J1 = J1
        self.m2 = m2
        self.J2 = J2

        # PTO parameters
        self.gear_ratio = gear_ratio
        self.torque_constant = torque_constant
        self.winding_resistance = winding_resistance
        self.winding_inductance = winding_inductance
        self.drivetrain_inertia = drivetrain_inertia
        self.drivetrain_friction_aft = drivetrain_friction_aft
        self.drivetrain_friction_bow = drivetrain_friction_bow
        self.drivetrain_stiffness = drivetrain_stiffness

        # Stabilization matrices
        if mooring_stiffness_full is None:
            mooring_stiffness_full = np.diag([1e5, 1e5, 1e5, 0, 0, 0, 0, 0, 0])
        if mooring_damping_full is None:
            mooring_damping_full = np.diag([1e4, 1e4, 1e4, 0, 0, 0, 0, 0, 0])
        if flap_linear_damping_full is None:
            flap_linear_damping_full = np.diag([0, 0, 0, 0, 0, 100, 0, 0, 100])

        self.mooring_stiffness_full = mooring_stiffness_full
        self.mooring_damping_full = mooring_damping_full
        self.flap_linear_damping_full = flap_linear_damping_full

        # Optimization settings
        self.scale_x_wec = scale_x_wec
        self.scale_x_opt = scale_x_opt
        self.scale_obj = scale_obj
        self.optim_options = {"maxiter": optim_maxiter}
        self.nsubsteps_post = nsubsteps_post
        self.pto_gains_init = pto_gains_init

        # Objects populated later
        self.platform_mesh = None
        self.flap1_mesh = None
        self.flap2_mesh = None

        self.platform = None
        self.flap1 = None
        self.flap2 = None
        self.allbodies = None

        self.bem_data = None
        self.impedance_reduced = None
        self.excitation_reduced = None
        self.hydrostatic_stiffness_reduced = None
        self.R0 = None
        self.M4E_ndof = None

        self.pto = None
        self.wec = None

        self._build_geometry_and_bodies()

    # ------------------------------------------------------------------
    # Geometry and body construction
    # ------------------------------------------------------------------
    def _build_geometry_and_bodies(self):
        self.platform_mesh = self._build_platform_mesh()
        self.flap1_mesh = self._build_flap_mesh(side="left")
        self.flap2_mesh = self._build_flap_mesh(side="right")

        self.platform = self._build_platform_body()
        self.flap1 = self._build_flap_body(
            name="flap1",
            mesh=self.flap1_mesh,
            center_of_mass=self.flap1_cg,
        )
        self.flap2 = self._build_flap_body(
            name="flap2",
            mesh=self.flap2_mesh,
            center_of_mass=self.flap2_cg,
        )

        self.allbodies = self.platform + self.flap1 + self.flap2

    def _build_platform_mesh(self):
        with pygmsh.occ.Geometry() as geom:
            gmsh.option.setNumber("Mesh.MeshSizeFactor", self.mesh_size_factor)

            platform = geom.add_box(
                [
                    -self.platform_frame_length / 2,
                    -self.platform_frame_width / 2,
                    -self.platform_frame_top_depth - self.platform_frame_thickness,
                ],
                [
                    self.platform_frame_length,
                    self.platform_frame_width,
                    self.platform_frame_thickness,
                ],
            )

            cutout = geom.add_box(
                [
                    -self.cutout_length / 2,
                    -self.cutout_width / 2,
                    -self.platform_frame_top_depth - self.platform_frame_thickness,
                ],
                [
                    self.cutout_length,
                    self.cutout_width,
                    self.platform_frame_thickness,
                ],
            )

            cyl1 = geom.add_cylinder(
                [-self.columns_xy[0], -self.columns_xy[1], -self.columns_draft],
                [0, 0, self.columns_draft + 0.01],
                self.columns_radius,
            )
            cyl2 = geom.add_cylinder(
                [-self.columns_xy[0], self.columns_xy[1], -self.columns_draft],
                [0, 0, self.columns_draft + 0.01],
                self.columns_radius,
            )
            cyl3 = geom.add_cylinder(
                [self.columns_xy[0], -self.columns_xy[1], -self.columns_draft],
                [0, 0, self.columns_draft + 0.01],
                self.columns_radius,
            )
            cyl4 = geom.add_cylinder(
                [self.columns_xy[0], self.columns_xy[1], -self.columns_draft],
                [0, 0, self.columns_draft + 0.01],
                self.columns_radius,
            )

            daq = geom.add_box(
                [
                    -self.daq_length / 2,
                    -self.daq_width / 2,
                    -self.daq_top_depth - self.daq_height,
                ],
                [self.daq_length, self.daq_width, self.daq_height],
            )

            platform_frame = geom.boolean_difference(platform, cutout)
            geom.boolean_union([platform_frame, daq, cyl1, cyl2, cyl3, cyl4])
            mesh = geom.generate_mesh()

        return mesh

    def _build_flap_mesh(self, side="left"):
        x0 = -self.flap_center_distance_apart / 2 if side == "left" else self.flap_center_distance_apart / 2

        with pygmsh.geo.Geometry() as geom:
            if side == "left":
                poly = geom.add_polygon(
                    [
                        [x0 - self.flap_thickness_bottom / 2, -self.flap_width / 2, -self.flap_draft],
                        [x0 + self.flap_thickness_bottom / 2, -self.flap_width / 2, -self.flap_draft],
                        [x0 + self.flap_thickness_top / 2, -self.flap_width / 2, self.flap_height - self.flap_draft],
                        [x0 - self.flap_thickness_top / 2, -self.flap_width / 2, self.flap_height - self.flap_draft],
                    ],
                    mesh_size=self.flap_mesh_size,
                )
            else:
                poly = geom.add_polygon(
                    [
                        [x0 - self.flap_thickness_bottom / 2, -self.flap_width / 2, -self.flap_draft],
                        [x0 + self.flap_thickness_bottom / 2, -self.flap_width / 2, -self.flap_draft],
                        [x0 + self.flap_thickness_top / 2, -self.flap_width / 2, self.flap_height - self.flap_draft],
                        [x0 - self.flap_thickness_top / 2, -self.flap_width / 2, self.flap_height - self.flap_draft],
                    ],
                    mesh_size=self.flap_mesh_size,
                )

            geom.extrude(poly, [0, self.flap_width, 0])
            mesh = geom.generate_mesh()

        return mesh

    def _build_platform_body(self):
        body = cpt.FloatingBody(
            mesh=self.platform_mesh,
            name="platform",
            center_of_mass=self.platform_cg,
        )
        body.add_all_rigid_body_dofs()
        body.rotation_center = body.center_of_mass

        inertia_matrix = xr.DataArray(
            data=np.asarray(np.diag([self.m1, self.m1, self.m1, self.J1, self.J1, self.J1])),
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

    def _build_flap_body(self, name, mesh, center_of_mass):
        body = cpt.FloatingBody(
            mesh=mesh,
            name=name,
            center_of_mass=center_of_mass,
        )
        body.add_all_rigid_body_dofs()
        body.rotation_center = body.center_of_mass

        inertia_matrix = xr.DataArray(
            data=np.asarray(np.diag([self.m2, self.m2, self.m2, self.J2, self.J2, self.J2])),
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
    # BEM and M4E reduction
    # ------------------------------------------------------------------
    def load_or_run_bem(self, bem_data_file=None, overwrite=False):
        if bem_data_file is None:
            bem_data_file = f"foswec_f1_{self.f1:.3f}_nf_{self.nfreq}".replace(".", "p") + ".nc"
        if (not overwrite) and os.path.exists(bem_data_file):
            bem_data = wot.read_netcdf(bem_data_file)
        else:
            bem_data = wot.run_bem(self.allbodies, self.freq, depth=self.depth)
            bem_data["radiating_dof"] = bem_data["radiating_dof"].astype(str)
            bem_data["influenced_dof"] = bem_data["influenced_dof"].astype(str)
            wot.write_netcdf(bem_data_file, bem_data)

        body_coord = bem_data.coords["body_name"].values
        bodies = [self.platform, self.flap1, self.flap2]
        com = np.vstack([b.center_of_mass for b in bodies])
        cob = np.vstack([b.center_of_buoyancy for b in bodies])
        volumes = np.array([b.immersed_part().volume for b in bodies], dtype=float)

        bem_data["center_of_mass"] = xr.DataArray(
            com,
            dims=("bod", "xyz"),
            coords={"body": body_coord, "xyz": ["x", "y", "z"]},
            attrs={"units": "m"},
        )
        bem_data["center_of_buoyancy"] = xr.DataArray(
            cob,
            dims=("bod", "xyz"),
            coords={"body": body_coord, "xyz": ["x", "y", "z"]},
            attrs={"units": "m"},
        )
        bem_data["volume"] = xr.DataArray(
            volumes,
            dims=("bod",),
            coords={"body": body_coord},
            attrs={"units": "m^3"},
        )

        self.bem_data = bem_data
        return bem_data

    def set_bem_data(self, bem_data):
        self.bem_data = bem_data

    def setup_m4e_reduction(self):
        if self.bem_data is None:
            raise ValueError("No BEM data available. Run load_or_run_bem(...) first.")

        m0 = [self.m1, self.m2, self.m2]
        J0 = [self.J1, self.J2, self.J2]

        (
            self.impedance_reduced,
            self.excitation_reduced,
            self.hydrostatic_stiffness_reduced,
            self.R0,
            self.M4E_ndof,
        ) = wot.utilities.setup_from_M4E(
            self.m4e_inputs_module,
            self.bem_data,
            [self.platform, self.flap1, self.flap2],
            m0,
            J0,
        )

        return {
            "impedance_reduced": self.impedance_reduced,
            "excitation_reduced": self.excitation_reduced,
            "hydrostatic_stiffness_reduced": self.hydrostatic_stiffness_reduced,
            "R0": self.R0,
            "M4E_ndof": self.M4E_ndof,
        }

    # ------------------------------------------------------------------
    # Additional forcing
    # ------------------------------------------------------------------
    def build_additional_forcing(self):
        if self.R0 is None:
            raise ValueError("Run setup_m4e_reduction() before building forcing terms.")

        mooring_stiffness = wot.utilities.reduce_damping_stiffness_M4E(
            self.mooring_stiffness_full, self.R0
        )
        mooring_damping = wot.utilities.reduce_damping_stiffness_M4E(
            self.mooring_damping_full, self.R0
        )
        flap_linear_damping = wot.utilities.reduce_damping_stiffness_M4E(
            self.flap_linear_damping_full, self.R0
        )

        def f_mooring(wec, x_wec, x_opt, wave, nsubsteps=1):
            pos = wec.vec_to_dofmat(x_wec)
            vel = jnp.dot(wec.derivative_mat, pos)
            time_matrix = wec.time_mat_nsubsteps(nsubsteps)
            mooring = -pos @ mooring_stiffness - vel @ mooring_damping
            mooring_force = jnp.dot(time_matrix, mooring)
            return mooring_force

        def f_linear_damping(wec, x_wec, x_opt, wave, nsubsteps=1):
            pos = wec.vec_to_dofmat(x_wec)
            vel = jnp.dot(wec.derivative_mat, pos)
            time_matrix = wec.time_mat_nsubsteps(nsubsteps)
            damping = -vel @ flap_linear_damping
            linear_damping_force = jnp.dot(time_matrix, damping)
            return linear_damping_force

        return {
            "f_mooring": f_mooring,
            "f_linear_damping": f_linear_damping,
        }

    # ------------------------------------------------------------------
    # PTO creation
    # ------------------------------------------------------------------
    def build_pto(self, P=True, I=False, coupled=False):
        if self.R0 is None:
            raise ValueError("Run setup_m4e_reduction() before building PTO.")

        omega = self.bem_data.omega.values

        drivetrain_impedance_aft = (
            1j * omega * self.drivetrain_inertia
            + self.drivetrain_friction_aft
            + 1 / (1j * omega) * self.drivetrain_stiffness
        )
        drivetrain_impedance_bow = (
            1j * omega * self.drivetrain_inertia
            + self.drivetrain_friction_bow
            + 1 / (1j * omega) * self.drivetrain_stiffness
        )

        winding_impedance = self.winding_resistance + 1j * omega * self.winding_inductance

        pto_impedance_11_aft = -1 * self.gear_ratio**2 * drivetrain_impedance_aft
        pto_impedance_11_bow = -1 * self.gear_ratio**2 * drivetrain_impedance_bow

        off_diag = np.sqrt(3.0 / 2.0) * self.torque_constant * self.gear_ratio
        pto_impedance_12 = -1 * (off_diag + 0j) * np.ones(omega.shape)
        pto_impedance_21 = -1 * (off_diag + 0j) * np.ones(omega.shape)
        pto_impedance_22 = winding_impedance

        pto_impedance = np.zeros((4, 4, self.nfreq), dtype=complex)

        pto_impedance[0, 0, :] = pto_impedance_11_aft
        pto_impedance[1, 1, :] = pto_impedance_11_bow
        pto_impedance[0, 2, :] = pto_impedance_12
        pto_impedance[1, 3, :] = pto_impedance_12
        pto_impedance[2, 0, :] = pto_impedance_21
        pto_impedance[3, 1, :] = pto_impedance_21
        pto_impedance[2, 2, :] = pto_impedance_22
        pto_impedance[3, 3, :] = pto_impedance_22

        names = ["flap 1 PTO", "flap 2 PTO"]
        kinematics_full = np.array(
            [
                [0, 0, -1, 0, 0, 1, 0, 0, 0],
                [0, 0, -1, 0, 0, 0, 0, 0, 1],
            ]
        )
        kinematics = wot.utilities.reduce_PTO_kinematics_M4E(kinematics_full, self.R0)

        pto_ndof = 2
        controller = CoupledDampingController(pto_ndof, P=P, I=I, coupled=coupled)
        loss = None

        self.pto = wot.pto.PTO(
            pto_ndof,
            kinematics,
            controller,
            pto_impedance,
            loss,
            names,
        )
        return self.pto

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    def build_wec(self, f_add):
        self.wec = wot.WEC.from_impedance(
            self.waves.freq.values,
            impedance=self.impedance_reduced,
            exc_coeff=self.excitation_reduced,
            hydrostatic_stiffness=self.hydrostatic_stiffness_reduced,
            constraints=None,
            f_add=f_add,
        )
        return self.wec

    def build_controller_bounds(self, P=True, I=False, coupled=False):
        if I and coupled:
            bounds_opt = (
                (-1e10, 0),
                (-1e10, 0),
                (-1e10, 0),
                (-1e10, 0),
                (-1e10, 1e10),
                (-1e10, 1e10),
                (-1e10, 1e10),
                (-1e10, 1e10),
            )
            nstate_opt = 8
        elif I:
            bounds_opt = (
                (-1e10, 0),
                (-1e10, 0),
                (-1e10, 1e10),
                (-1e10, 1e10),
            )
            nstate_opt = 4
        elif coupled:
            bounds_opt = (
                (-1e10, 0),
                (-1e10, 0),
                (-1e10, 0),
                (-1e10, 0),
            )
            nstate_opt = 4
        else:
            bounds_opt = (
                (-1e10, 0),
                (-1e10, 0),
            )
            nstate_opt = 2

        return nstate_opt, bounds_opt

    def build_initial_conditions(self):
        x_wec_0 = np.zeros((5 * 2 * self.nfreq,))
        return x_wec_0

    def solve_controller_case(self, P=True, I=False, coupled=False):
        if self.bem_data is None:
            raise ValueError("No BEM data available. Run load_or_run_bem(...) first.")
        if self.R0 is None:
            raise ValueError("M4E reduction not initialized. Run setup_m4e_reduction() first.")

        pto = self.build_pto(P=P, I=I, coupled=coupled)
        forcing = self.build_additional_forcing()
        f_add = {
            **forcing,
            "f_PTO": pto.force_on_wec,
        }

        wec = self.build_wec(f_add)

        nstate_opt, bounds_opt = self.build_controller_bounds(P=P, I=I, coupled=coupled)
        x_wec_0 = self.build_initial_conditions()
        x_opt_0 = [-self.pto_gains_init] * nstate_opt
        obj_fun = pto.average_power

        results = wec.solve(
            self.waves,
            obj_fun,
            nstate_opt,
            scale_x_wec=self.scale_x_wec,
            scale_x_opt=self.scale_x_opt,
            scale_obj=self.scale_obj,
            x_wec_0=x_wec_0,
            x_opt_0=x_opt_0,
            bounds_opt=bounds_opt,
            optim_options=self.optim_options,
        )

        wec_fdom_full, wec_tdom_full = wot.utilities.post_process_M4E(
            wec, results, self.waves, self.nsubsteps_post, self.R0
        )
        pto_fdom, pto_tdom = pto.post_process(
            wec, results, self.waves, nsubsteps=self.nsubsteps_post
        )

        avg_power = np.mean(
            pto_tdom["power"][0, 1, :, 0].values + pto_tdom["power"][0, 1, :, 1].values
        )

        summary = {
            "P": P,
            "I": I,
            "coupled": coupled,
            "avg_power": float(avg_power),
            "objective": float(results[0].fun),
            "x_opt": np.array(results[0].x, copy=True),
        }

        return {
            "wec": wec,
            "pto": pto,
            "results": results,
            "wec_fdom_full": wec_fdom_full,
            "wec_tdom_full": wec_tdom_full,
            "pto_fdom": pto_fdom,
            "pto_tdom": pto_tdom,
            "summary": summary,
        }

    def run_standard_controller_study(self):
        cases = [
            ("P", dict(P=True, I=False, coupled=False)),
            ("P_coupled", dict(P=True, I=False, coupled=True)),
            ("PI", dict(P=True, I=True, coupled=False)),
            ("PI_coupled", dict(P=True, I=True, coupled=True)),
        ]

        out = {}
        for name, kwargs in cases:
            logging.info(f"Running controller case: {name}")
            out[name] = self.solve_controller_case(**kwargs)

        return out

    def results_table(self, results_dict):
        rows = []
        for name, data in results_dict.items():
            row = copy.deepcopy(data["summary"])
            row["case"] = name
            rows.append(row)
        try:
            import pandas as pd
            return pd.DataFrame(rows)
        except ImportError:
            return rows