import os
import copy
import logging
import numpy as np
import xarray as xr
import jax.numpy as jnp
import matplotlib.pyplot as plt

import pygmsh
import gmsh
import capytaine as cpt
import wecopttool as wot

from wavespectra.construct.frequency import pierson_moskowitz

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

class UnstructuredForceController(_ControllerBase):
    def __init__(self, ndof_pto: int, nfreq=None):
        self._ndof = ndof_pto
        self.nfreq = nfreq

    @property
    def ndof(self) -> int:
        return self._ndof

    @property
    def ngains(self) -> int:
        return 0

    @property
    def nstate(self) -> int:
        return self.ngains

    def force(self, pto, wec, x_wec, x_opt, wave=None, nsubsteps=1):
        if self.nfreq is None:
            force_x_opt = x_opt
        else:
            n_force_states = 2 * self.nfreq * self.ndof
            force_x_opt = x_opt[:n_force_states]

        force_fdom = jnp.reshape(force_x_opt, (-1, self.ndof))
        time_matrix = wec.time_mat_nsubsteps(nsubsteps)
        force_td = jnp.dot(time_matrix, force_fdom)
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
        wavefreq=1 / 2.61,
        f1=1 / 2.61/8,
        nfreq=50,
        wave_type="irregular",
        amplitude=0.136,
        phase=0.0,
        wavedir=0.0,
        wave_seed=0,
        depth=2.0,
        bem_data_dir="bem_data",
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
        flap1_mass_on_top=0.0,
        flap2_mass_on_top=None,
        added_mass_height_above_hinge=None,
        # drivetrain / PTO
        gear_ratio=3.75,
        torque_constant=1.021,
        winding_resistance=1.028,
        winding_inductance=0.0,
        drivetrain_inertia=0.05,
        drivetrain_friction_aft = 3.27 / 3.75**2,
        drivetrain_friction_bow = 2.7 / 3.75**2,
        drivetrain_stiffness=0.0,
        # flap pitch constraints
        max_flap_pitch_deg=30.0,
        nsubsteps_constraint=4,
        use_flap_pitch_constraints=True,
        # other constraints and opt variables
        max_torque_drive=40.0,
        max_rms_torque_drive=20.0,
        optimize_stiffness=True,
        stiffness_init_factor=6.0,
        fixed_stiffness=0.0,
        # stabilization
        mooring_stiffness_full=None,
        mooring_damping_full=None,
        flap_linear_damping_full=None,
        # optimization
        scale_x_wec=1e1,
        scale_x_opt=1e-2,
        scale_obj=1.0,
        optim_maxiter=300,
        nsubsteps_post=10,
        pto_gains_init=1e2,
        log_level=logging.INFO,
    ):
        logging.getLogger().setLevel(log_level)

        self.m4e_inputs_module = m4e_inputs_module

        # Irregular Wave/frequency
        self.wavefreq = wavefreq

        self.f1 = f1
        self.nfreq = nfreq
        self.freq = wot.frequency(self.f1, self.nfreq, False)
        
        self.amplitude = amplitude
        self.phase = phase
        self.wavedir = wavedir
        self.wave_seed = wave_seed
        self.wave_type = wave_type

        self.waves = self.build_waves(
            amplitude=self.amplitude,
            seed=self.wave_seed,
        )
        
        self.depth = depth

        # Mesh settings
        self.mesh_size_factor = mesh_size_factor
        self.flap_mesh_size = flap_mesh_size
        self.bem_data_dir = bem_data_dir
        os.makedirs(self.bem_data_dir, exist_ok=True)

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

        # Rigid-body properties
        self.m1 = float(m1)
        self.J1 = float(J1)

        # Baseline flap properties before adding top mass
        self.flap_mass_base = float(m2)
        self.flap_inertia_cg_base = float(J2)

        # Added top mass settings
        self.flap1_mass_on_top = float(flap1_mass_on_top)

        if flap2_mass_on_top is None:
            self.flap2_mass_on_top = float(flap1_mass_on_top)
        else:
            self.flap2_mass_on_top = float(flap2_mass_on_top)

        if added_mass_height_above_hinge is None:
            self.added_mass_height_above_hinge = float(self.flap_height)
        else:
            self.added_mass_height_above_hinge = float(added_mass_height_above_hinge)

        # Compute effective flap mass properties and CG locations
        self._apply_added_top_mass_properties()

        # PTO parameters
        self.gear_ratio = gear_ratio
        self.torque_constant = torque_constant
        self.winding_resistance = winding_resistance
        self.winding_inductance = winding_inductance
        self.drivetrain_inertia = drivetrain_inertia
        self.drivetrain_friction_aft = drivetrain_friction_aft
        self.drivetrain_friction_bow = drivetrain_friction_bow
        self.drivetrain_stiffness = drivetrain_stiffness

        self.max_flap_pitch = np.deg2rad(max_flap_pitch_deg)
        self.nsubsteps_constraint = nsubsteps_constraint
        self.use_flap_pitch_constraints = use_flap_pitch_constraints

        self.max_torque = max_torque_drive * torque_constant
        self.max_rms_torque = max_rms_torque_drive * torque_constant

        self.optimize_stiffness = optimize_stiffness
        self.stiffness_init_factor = stiffness_init_factor
        self.fixed_stiffness = fixed_stiffness

        # Stabilization matrices
        if mooring_stiffness_full is None:
            mooring_stiffness_full = np.diag([8e3, 2e5, 2e5, 0, 0, 0, 0, 0, 0])
        if mooring_damping_full is None:
            mooring_damping_full = np.diag([8e2, 1e4, 1e4, 0, 0, 0, 0, 0, 0])
        if flap_linear_damping_full is None:
            flap_linear_damping_full = np.diag([0, 0, 0, 0, 0, 10, 0, 0, 10])

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
    # Added-top-mass helpers
    # ------------------------------------------------------------------
    def compute_flap_mass_properties(self, mass_on_top):
        """
        Compute updated flap rigid-body properties when a point mass is added
        above the hinge.

        The baseline flap is assumed to have:
          - mass self.flap_mass_base
          - CG height self.cg_height_above_hinge above hinge
          - inertia self.flap_inertia_cg_base about its own CG

        The added point mass is located at:
          - self.added_mass_height_above_hinge above the hinge

        Returns properties for one flap.
        """
        mass_on_top = float(mass_on_top)

        m_base = float(self.flap_mass_base)
        J_cg_base = float(self.flap_inertia_cg_base)
        cg_base = float(self.cg_height_above_hinge)
        h_added = float(self.added_mass_height_above_hinge)

        total_mass = m_base + mass_on_top

        if total_mass <= 0.0:
            raise ValueError(
                f"Total flap mass must be positive. Got "
                f"m_base={m_base}, mass_on_top={mass_on_top}, "
                f"total_mass={total_mass}."
            )

        # New CG height above hinge
        cg_height = (
            m_base * cg_base
            + mass_on_top * h_added
        ) / total_mass

        # Baseline inertia about hinge
        I_hinge_base = J_cg_base + m_base * cg_base**2

        # Added point-mass contribution about hinge
        I_hinge_added = mass_on_top * h_added**2

        # New total inertia about hinge
        I_hinge = I_hinge_base + I_hinge_added

        # Convert back to inertia about the new combined CG
        I_cg = I_hinge - total_mass * cg_height**2

        return {
            "mass_on_top": mass_on_top,
            "total_mass": total_mass,
            "cg_height_above_hinge": cg_height,
            "inertia_cg": I_cg,
            "inertia_hinge": I_hinge,
            "added_mass_height_above_hinge": h_added,
        }

    def _apply_added_top_mass_properties(self):
        """
        Update internal flap mass, inertia, and CG properties based on the
        current flap1_mass_on_top and flap2_mass_on_top values.

        This does not rebuild the meshes or bodies by itself.
        """
        self.flap1_mass_props = self.compute_flap_mass_properties(
            self.flap1_mass_on_top
        )
        self.flap2_mass_props = self.compute_flap_mass_properties(
            self.flap2_mass_on_top
        )

        self.m2_flap1 = float(self.flap1_mass_props["total_mass"])
        self.J2_flap1 = float(self.flap1_mass_props["inertia_cg"])

        self.m2_flap2 = float(self.flap2_mass_props["total_mass"])
        self.J2_flap2 = float(self.flap2_mass_props["inertia_cg"])

        self.flap1_cg = [
            -self.flap_center_distance_apart / 2,
            0.0,
            -self.flap_draft + self.flap1_mass_props["cg_height_above_hinge"],
        ]

        self.flap2_cg = [
            self.flap_center_distance_apart / 2,
            0.0,
            -self.flap_draft + self.flap2_mass_props["cg_height_above_hinge"],
        ]

        # Backward-compatible aliases.
        # If both flaps are identical, these are the common values.
        # If not, these refer to flap 1.
        self.m2 = self.m2_flap1
        self.J2 = self.J2_flap1

    def set_added_top_mass(
        self,
        flap1_mass_on_top,
        flap2_mass_on_top=None,
        rebuild_bodies=True,
        invalidate_reduction=True,
    ):
        """
        Update the added top mass on the flaps.

        Parameters
        ----------
        flap1_mass_on_top : float
            Added point mass on top of flap 1.
        flap2_mass_on_top : float or None
            Added point mass on top of flap 2. If None, uses same value as flap 1.
        rebuild_bodies : bool
            If True, rebuilds the Capytaine flap bodies and allbodies object.
        invalidate_reduction : bool
            If True, clears BEM/M4E/WEC/PTO objects that depend on mass properties.

        Returns
        -------
        dict
            Mass-property dictionaries for flap 1 and flap 2.
        """
        self.flap1_mass_on_top = float(flap1_mass_on_top)

        if flap2_mass_on_top is None:
            self.flap2_mass_on_top = float(flap1_mass_on_top)
        else:
            self.flap2_mass_on_top = float(flap2_mass_on_top)

        self._apply_added_top_mass_properties()

        if rebuild_bodies:
            if self.platform_mesh is None or self.flap1_mesh is None or self.flap2_mesh is None:
                raise ValueError(
                    "Meshes have not been built yet. Cannot rebuild bodies."
                )

            self.platform = self._build_platform_body()

            self.flap1 = self._build_flap_body(
                name="flap1",
                mesh=self.flap1_mesh,
                center_of_mass=self.flap1_cg,
                mass=self.m2_flap1,
                inertia_cg=self.J2_flap1,
            )

            self.flap2 = self._build_flap_body(
                name="flap2",
                mesh=self.flap2_mesh,
                center_of_mass=self.flap2_cg,
                mass=self.m2_flap2,
                inertia_cg=self.J2_flap2,
            )

            self.allbodies = self.platform + self.flap1 + self.flap2

        if invalidate_reduction:
            self.bem_data = None
            self.impedance_reduced = None
            self.excitation_reduced = None
            self.hydrostatic_stiffness_reduced = None
            self.R0 = None
            self.M4E_ndof = None
            self.pto = None
            self.wec = None

        return {
            "flap1": copy.deepcopy(self.flap1_mass_props),
            "flap2": copy.deepcopy(self.flap2_mass_props),
        }

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
            mass=self.m2_flap1,
            inertia_cg=self.J2_flap1,
        )

        self.flap2 = self._build_flap_body(
            name="flap2",
            mesh=self.flap2_mesh,
            center_of_mass=self.flap2_cg,
            mass=self.m2_flap2,
            inertia_cg=self.J2_flap2,
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

    def _build_flap_body(self, name, mesh, center_of_mass, mass, inertia_cg):
        body = cpt.FloatingBody(
            mesh=mesh,
            name=name,
            center_of_mass=center_of_mass,
        )

        body.add_all_rigid_body_dofs()

        # Keep your original convention:
        # rotations are about the body's current center of mass.
        body.rotation_center = body.center_of_mass

        inertia_matrix = xr.DataArray(
            data=np.asarray(
                np.diag(
                    [
                        mass,
                        mass,
                        mass,
                        inertia_cg,
                        inertia_cg,
                        inertia_cg,
                    ]
                )
            ),
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
    def _num_for_filename(self, value):
        """
        Convert a number to a filename-safe string.
        """
        return f"{float(value):.6g}".replace("-", "m").replace(".", "p")

    def default_bem_filename(self):
        """
        Default BEM cache filename including frequency grid, depth, and
        added-top-mass values.
        """
        f1_str = self._num_for_filename(self.f1)
        depth_str = self._num_for_filename(self.depth)
        mtop1_str = self._num_for_filename(self.flap1_mass_on_top)
        mtop2_str = self._num_for_filename(self.flap2_mass_on_top)

        return (
            f"foswec_f1_{f1_str}"
            f"_nf_{self.nfreq}"
            f"_depth_{depth_str}"
            f"_mtop1_{mtop1_str}"
            f"_mtop2_{mtop2_str}"
            + ".nc"
        )
    
    def load_or_run_bem(self, bem_data_file=None, overwrite=False):
        os.makedirs(self.bem_data_dir, exist_ok=True)

        if bem_data_file is None:
            bem_data_file = self.default_bem_filename()

        # If user passes only a filename, save/load inside self.bem_data_dir.
        # If user passes a path, respect that path.
        if os.path.dirname(bem_data_file) == "":
            bem_data_file = os.path.join(self.bem_data_dir, bem_data_file)

        if (not overwrite) and os.path.exists(bem_data_file):
            logging.info(f"Loading BEM data from: {bem_data_file}")
            bem_data = wot.read_netcdf(bem_data_file)
        else:
            logging.info(f"Running BEM and saving to: {bem_data_file}")
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

        bem_data = bem_data.fillna(0.0)
        self.bem_data = bem_data
        return bem_data

    def set_bem_data(self, bem_data):
        self.bem_data = bem_data

    # ------------------------------------------------------------------
    # Wave helpers
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Wave helpers
    # ------------------------------------------------------------------
    def build_irregular_waves(self, amplitude=None, seed=None):
        """
        Build long-crested Pierson-Moskowitz irregular waves using the
        current frequency grid.
        """
        if amplitude is None:
            amplitude = self.amplitude

        if seed is None:
            seed = self.wave_seed

        efth = pierson_moskowitz(
            freq=self.freq,
            hs=2 * amplitude,
            fp=self.wavefreq,
        )

        waves = wot.waves.long_crested_wave(
            efth,
            nrealizations=1,
            direction=self.wavedir,
            seed=seed,
        )

        return waves

    def build_regular_waves(self, amplitude=None):
        """
        Build regular waves using the current frequency grid.
        """
        if amplitude is None:
            amplitude = self.amplitude

        waves = wot.waves.regular_wave(
            self.f1,
            self.nfreq,
            self.wavefreq,
            amplitude,
            self.phase,
            self.wavedir,
        )

        return waves

    def build_waves(self, amplitude=None, seed=None):
        """
        Build waves according to self.wave_type.
        """
        if self.wave_type == "irregular":
            return self.build_irregular_waves(
                amplitude=amplitude,
                seed=seed,
            )

        elif self.wave_type == "regular":
            return self.build_regular_waves(
                amplitude=amplitude,
            )

        else:
            raise ValueError(
                f"Unknown wave_type={self.wave_type!r}. "
                "Use 'irregular' or 'regular'."
            )

    def set_wave_amplitude(self, amplitude, seed=None):
        """
        Update wave amplitude without rebuilding geometry/BEM/M4E.
        """
        self.amplitude = float(amplitude)

        if seed is not None:
            self.wave_seed = seed

        self.waves = self.build_waves(
            amplitude=self.amplitude,
            seed=self.wave_seed,
        )

        return self.waves

    def set_wave_type(self, wave_type, amplitude=None, seed=None):
        """
        Switch between 'regular' and 'irregular' waves.
        """
        self.wave_type = wave_type

        if amplitude is not None:
            self.amplitude = float(amplitude)

        if seed is not None:
            self.wave_seed = seed

        self.waves = self.build_waves(
            amplitude=self.amplitude,
            seed=self.wave_seed,
        )

        return self.waves


    def setup_m4e_reduction(self):
        if self.bem_data is None:
            raise ValueError("No BEM data available. Run load_or_run_bem(...) first.")

        m0 = [
            self.m1,
            self.m2_flap1,
            self.m2_flap2,
        ]

        J0 = [
            self.J1,
            self.J2_flap1,
            self.J2_flap2,
        ]

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
    def build_additional_forcing(self, include_drivetrain_stiffness=False):
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

        forcing = {
            "f_mooring": f_mooring,
            "f_linear_damping": f_linear_damping,
        }

        if include_drivetrain_stiffness:
            pto = self.pto
            pto_kinematics = jnp.asarray(self.pto_kinematics_reduced)

            def _get_stiffness_values(x_opt):
                if self.optimize_stiffness:
                    k_dt_1 = x_opt[-2]
                    k_dt_2 = x_opt[-1]
                else:
                    fixed_stiffness = jnp.asarray(self.fixed_stiffness)

                    if fixed_stiffness.ndim == 0:
                        k_dt_1 = fixed_stiffness
                        k_dt_2 = fixed_stiffness
                    else:
                        k_dt_1 = fixed_stiffness[0]
                        k_dt_2 = fixed_stiffness[1]

                return k_dt_1, k_dt_2

            def f_drivetrain_stiffness_flap1(wec, x_wec, x_opt, wave, nsubsteps=1):
                pto_pos = pto.position(wec, x_wec, x_opt, wave, nsubsteps)

                k_dt_1, _ = _get_stiffness_values(x_opt)

                spring_force_pto = jnp.zeros_like(pto_pos)
                spring_force_pto = spring_force_pto.at[:, 0].set(
                    -k_dt_1 * pto_pos[:, 0]
                )

                spring_force_wec = jnp.dot(spring_force_pto, pto_kinematics)

                return spring_force_wec

            def f_drivetrain_stiffness_flap2(wec, x_wec, x_opt, wave, nsubsteps=1):
                pto_pos = pto.position(wec, x_wec, x_opt, wave, nsubsteps)

                _, k_dt_2 = _get_stiffness_values(x_opt)

                spring_force_pto = jnp.zeros_like(pto_pos)
                spring_force_pto = spring_force_pto.at[:, 1].set(
                    -k_dt_2 * pto_pos[:, 1]
                )

                spring_force_wec = jnp.dot(spring_force_pto, pto_kinematics)

                return spring_force_wec

            forcing["f_drivetrain_stiffness_flap1"] = f_drivetrain_stiffness_flap1
            forcing["f_drivetrain_stiffness_flap2"] = f_drivetrain_stiffness_flap2

        return forcing

    # ------------------------------------------------------------------
    # PTO creation
    # ------------------------------------------------------------------
    def build_pto(self, P=True, I=False, coupled=False, unstructured=False):
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

        pto_impedance = np.zeros((4, 4, len(omega)), dtype=complex)

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
        self.pto_kinematics_reduced = kinematics

        pto_ndof = 2
        if unstructured:
            controller = UnstructuredForceController(
                pto_ndof,
                nfreq=len(omega),
            )
        else:
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

    def build_flap_pitch_constraints(self):
        if self.pto is None:
            raise ValueError("Build PTO before building constraints.")

        pto = self.pto
        max_flap_pitch = self.max_flap_pitch
        max_torque = self.max_torque
        max_rms_torque = self.max_rms_torque
        gear_ratio = self.gear_ratio
        nsubsteps = self.nsubsteps_constraint

        def const_flap1_rotation(wec, x_wec, x_opt, wave, nsubsteps=nsubsteps):
            pto_pos = pto.position(wec, x_wec, x_opt, wave, nsubsteps)
            flap1_pos = pto_pos[:, 0]
            return max_flap_pitch - jnp.abs(flap1_pos.flatten())

        def const_flap2_rotation(wec, x_wec, x_opt, wave, nsubsteps=nsubsteps):
            pto_pos = pto.position(wec, x_wec, x_opt, wave, nsubsteps)
            flap2_pos = pto_pos[:, 1]
            return max_flap_pitch - jnp.abs(flap2_pos.flatten())

        def const_motor_torque(wec, x_wec, x_opt, wave):
            pto_force = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
            motor_torque = pto_force / gear_ratio
            return max_torque - jnp.abs(motor_torque.flatten())

        def const_motor_rms_torque(wec, x_wec, x_opt, wave):
            pto_force = pto.force(wec, x_wec, x_opt, wave, nsubsteps)
            motor_torque = pto_force / gear_ratio
            rms_by_flap = jnp.sqrt(jnp.mean(motor_torque**2, axis=0) + 1e-12)
            return max_rms_torque - rms_by_flap

        constraints = [
            {"type": "ineq", "fun": const_flap1_rotation},
            {"type": "ineq", "fun": const_flap2_rotation},
            {"type": "ineq", "fun": const_motor_torque},
            {"type": "ineq", "fun": const_motor_rms_torque},
        ]

        return constraints

    # ------------------------------------------------------------------
    # Optimization helpers
    # ------------------------------------------------------------------
    def build_wec(self, f_add, constraints=None):
        self.wec = wot.WEC.from_impedance(
            self.waves.freq.values,
            impedance=self.impedance_reduced,
            exc_coeff=self.excitation_reduced,
            hydrostatic_stiffness=self.hydrostatic_stiffness_reduced,
            constraints=constraints,
            f_add=f_add,
        )
        return self.wec

    def build_controller_bounds(self, P=True, I=False, coupled=False, unstructured=False):
        if unstructured:
            nfreq_bem = len(self.bem_data.omega)
            n_force_states = 2 * nfreq_bem * 2  # 2 Fourier coeffs/freq * 2 PTO DOFs

            if self.optimize_stiffness:
                nstate_opt = n_force_states + 2

                bounds_opt = (
                    ((-1e8, 1e8),) * n_force_states
                    + ((0.0, 1e10),)
                    + ((0.0, 1e10),)
                )
            else:
                nstate_opt = n_force_states
                bounds_opt = ((-1e8, 1e8),) * nstate_opt

            return nstate_opt, bounds_opt
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
        if self.wec is None:
            raise ValueError("Build WEC before building initial conditions.")

        x_wec_0 = np.zeros(self.wec.nstate_wec)
        return x_wec_0

    def solve_controller_case(self, P=True, I=False, coupled=False, unstructured=False,x_wec_0_override=None,x_opt_0_override=None):
        if self.bem_data is None:
            raise ValueError("No BEM data available. Run load_or_run_bem(...) first.")
        if self.R0 is None:
            raise ValueError("M4E reduction not initialized. Run setup_m4e_reduction() first.")

        pto = self.build_pto(P=P, I=I, coupled=coupled, unstructured=unstructured)
        include_drivetrain_stiffness = unstructured and (
            self.optimize_stiffness or self.fixed_stiffness != 0.0
        )

        forcing = self.build_additional_forcing(
            include_drivetrain_stiffness=include_drivetrain_stiffness
        )
        f_add = {
            **forcing,
            "f_PTO": pto.force_on_wec,
        }

        if self.use_flap_pitch_constraints:
            constraints = self.build_flap_pitch_constraints()
        else:
            constraints = None

        wec = self.build_wec(f_add, constraints=constraints)

        nstate_opt, bounds_opt = self.build_controller_bounds(
            P=P,
            I=I,
            coupled=coupled,
            unstructured=unstructured,
        )
        x_wec_0 = self.build_initial_conditions()
        if x_wec_0_override is None:
            x_wec_0 = self.build_initial_conditions()
        else:
            x_wec_0 = np.asarray(x_wec_0_override)

        if x_opt_0_override is None:
            if unstructured:
                x_opt_0 = np.zeros(nstate_opt)

                if self.optimize_stiffness:
                    k_init_1 = self.stiffness_init_factor * self.flap1_mass_on_top
                    k_init_2 = self.stiffness_init_factor * self.flap2_mass_on_top

                    x_opt_0[-2] = k_init_1
                    x_opt_0[-1] = k_init_2
            else:
                x_opt_0 = -self.pto_gains_init * np.ones(nstate_opt)
        else:
            x_opt_0 = np.asarray(x_opt_0_override)

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

        constraint_metrics = self.evaluate_constraint_margins(
            {
                "pto_tdom": pto_tdom,
            }
        )

        avg_power = np.mean(
            pto_tdom["power"][0, 1, :, 0].values + pto_tdom["power"][0, 1, :, 1].values
        )

        x_full = np.asarray(results[0].x)
        nstate_wec = int(wec.nstate_wec)

        x_wec_sol = np.array(x_full[:nstate_wec], copy=True)
        x_opt_sol = np.array(x_full[nstate_wec:], copy=True)

        if unstructured and self.optimize_stiffness:
            opt_stiffness_flap1 = float(x_opt_sol[-2])
            opt_stiffness_flap2 = float(x_opt_sol[-1])
            opt_stiffness = 0.5 * (opt_stiffness_flap1 + opt_stiffness_flap2)
        else:
            fixed_stiffness = np.asarray(self.fixed_stiffness)

            if fixed_stiffness.ndim == 0:
                opt_stiffness_flap1 = float(fixed_stiffness)
                opt_stiffness_flap2 = float(fixed_stiffness)
            else:
                opt_stiffness_flap1 = float(fixed_stiffness[0])
                opt_stiffness_flap2 = float(fixed_stiffness[1])

            opt_stiffness = 0.5 * (opt_stiffness_flap1 + opt_stiffness_flap2)

        summary = {
            "P": P,
            "I": I,
            "coupled": coupled,
            "unstructured": unstructured,

            "flap1_mass_on_top": float(self.flap1_mass_on_top),
            "flap2_mass_on_top": float(self.flap2_mass_on_top),

            "flap1_total_mass": float(self.m2_flap1),
            "flap2_total_mass": float(self.m2_flap2),

            "flap1_cg_height_above_hinge": float(
                self.flap1_mass_props["cg_height_above_hinge"]
            ),
            "flap2_cg_height_above_hinge": float(
                self.flap2_mass_props["cg_height_above_hinge"]
            ),

            "flap1_inertia_cg": float(self.J2_flap1),
            "flap2_inertia_cg": float(self.J2_flap2),

            "flap1_inertia_hinge": float(
                self.flap1_mass_props["inertia_hinge"]
            ),
            "flap2_inertia_hinge": float(
                self.flap2_mass_props["inertia_hinge"]
            ),

            "avg_power": float(avg_power),
            "objective": float(results[0].fun),

            # Full optimizer vector
            "x_full": np.array(results[0].x, copy=True),

            # Decomposed initial-condition pieces for continuation
            "x_wec": np.array(x_wec_sol, copy=True),
            "x_opt": np.array(x_opt_sol, copy=True),
            "opt_stiffness_flap1": opt_stiffness_flap1,
            "opt_stiffness_flap2": opt_stiffness_flap2,
            "fixed_stiffness": float(self.fixed_stiffness),
            "opt_power": float(results[0].fun),
            "mean_elec_power": float(avg_power),
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
            "constraint_metrics": constraint_metrics,
        }

    def run_standard_controller_study(self):
        cases = [
            ("P", dict(P=True, I=False, coupled=False)),
            ("P_coupled", dict(P=True, I=False, coupled=True)),
            ("PI", dict(P=True, I=True, coupled=False)),
            ("PI_coupled", dict(P=True, I=True, coupled=True)),
            ("unstructured", dict(P=False, I=False, coupled=False, unstructured=True)),
        ]

        out = {}
        for name, kwargs in cases:
            logging.info(f"Running controller case: {name}")
            out[name] = self.solve_controller_case(**kwargs)

        return out

    def evaluate_constraint_margins(self, case_data, realization=0):
        pto_tdom = case_data["pto_tdom"]

        if "pos" in pto_tdom:
            pos_da = pto_tdom["pos"]
        elif "position" in pto_tdom:
            pos_da = pto_tdom["position"]
        else:
            raise KeyError("Could not find PTO position variable in pto_tdom.")

        pos = np.asarray(pos_da.sel(realization=realization).values)
        force = np.asarray(pto_tdom["force"].sel(realization=realization).values)

        pos = np.squeeze(pos)
        force = np.squeeze(force)

        if pos.ndim == 1:
            pos = pos[:, None]
        if force.ndim == 1:
            force = force[:, None]

        # Ensure shape is time x dof
        if pos.shape[-1] != 2 and pos.shape[0] == 2:
            pos = pos.T
        if force.shape[-1] != 2 and force.shape[0] == 2:
            force = force.T

        motor_torque = force / self.gear_ratio

        max_abs_pos = np.max(np.abs(pos), axis=0)
        max_abs_motor_torque = np.max(np.abs(motor_torque), axis=0)
        rms_motor_torque = np.sqrt(np.mean(motor_torque**2, axis=0))

        peak_rotation_margin = self.max_flap_pitch - max_abs_pos
        peak_torque_margin = self.max_torque - max_abs_motor_torque
        rms_torque_margin = self.max_rms_torque - rms_motor_torque

        return {
            "flap1_max_abs_pos": float(max_abs_pos[0]),
            "flap2_max_abs_pos": float(max_abs_pos[1]),

            "flap1_peak_rotation_utilization": float(max_abs_pos[0] / self.max_flap_pitch),
            "flap2_peak_rotation_utilization": float(max_abs_pos[1] / self.max_flap_pitch),

            "flap1_max_abs_motor_torque": float(max_abs_motor_torque[0]),
            "flap2_max_abs_motor_torque": float(max_abs_motor_torque[1]),

            "flap1_rms_motor_torque": float(rms_motor_torque[0]),
            "flap2_rms_motor_torque": float(rms_motor_torque[1]),

            "flap1_peak_torque_utilization": float(max_abs_motor_torque[0] / self.max_torque),
            "flap2_peak_torque_utilization": float(max_abs_motor_torque[1] / self.max_torque),

            "flap1_rms_torque_utilization": float(rms_motor_torque[0] / self.max_rms_torque),
            "flap2_rms_torque_utilization": float(rms_motor_torque[1] / self.max_rms_torque),

            "flap1_peak_rotation_margin": float(peak_rotation_margin[0]),
            "flap2_peak_rotation_margin": float(peak_rotation_margin[1]),

            "flap1_peak_torque_margin": float(peak_torque_margin[0]),
            "flap2_peak_torque_margin": float(peak_torque_margin[1]),

            "flap1_rms_torque_margin": float(rms_torque_margin[0]),
            "flap2_rms_torque_margin": float(rms_torque_margin[1]),

            "all_constraints_satisfied": bool(
                np.all(peak_rotation_margin >= 0.0)
                and np.all(peak_torque_margin >= 0.0)
                and np.all(rms_torque_margin >= 0.0)
            ),
        }

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

    # ------------------------------------------------------------------
    # Frequency-domain plotting utilities
    # ------------------------------------------------------------------
    @staticmethod
    def stem_subplots_from_array(
        freq,
        values,
        labels,
        title,
        ylabel,
        skip_zero=True,
        plot_mode="magnitude",
        figsize_per_subplot=1.9,
    ):
        """
        Make stem subplots for frequency-domain complex data.

        Parameters
        ----------
        freq : array-like, shape (nfreq,)
            Frequency in Hz.
        values : array-like, shape (nfreq, ndof)
            Complex or real frequency-domain values.
        labels : list[str]
            Labels for each DOF/column.
        title : str
            Figure title.
        ylabel : str
            Y-axis label.
        skip_zero : bool
            Whether to omit freq=0.
        plot_mode : {"magnitude", "real", "imag"}
            How to display complex values.
        figsize_per_subplot : float
            Height multiplier for each subplot.

        Returns
        -------
        fig, axes
            Matplotlib figure and axes.
        """
        freq = np.asarray(freq)
        values = np.asarray(values)

        if values.ndim != 2:
            raise ValueError(f"`values` must have shape (nfreq, ndof). Got {values.shape}.")

        if values.shape[0] != len(freq):
            raise ValueError(
                f"First dimension of values must match freq length. "
                f"Got values.shape={values.shape}, len(freq)={len(freq)}."
            )

        if skip_zero:
            mask = freq > 0.0
        else:
            mask = np.ones_like(freq, dtype=bool)

        f_plot = freq[mask]
        v_plot = values[mask, :]

        if plot_mode == "magnitude":
            y_plot = np.abs(v_plot)
        elif plot_mode == "real":
            y_plot = np.real(v_plot)
        elif plot_mode == "imag":
            y_plot = np.imag(v_plot)
        else:
            raise ValueError("plot_mode must be 'magnitude', 'real', or 'imag'.")

        ndof = y_plot.shape[1]

        fig, axes = plt.subplots(
            ndof,
            1,
            figsize=(10, figsize_per_subplot * ndof + 1.0),
            sharex=True,
            constrained_layout=True,
        )

        if ndof == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            markerline, stemlines, baseline = ax.stem(
                f_plot,
                y_plot[:, i],
                basefmt=" ",
            )

            markerline.set_markersize(4)
            stemlines.set_linewidth(1.0)

            ax.grid(True, alpha=0.3)
            ax.set_ylabel(ylabel)
            ax.set_title(labels[i])

        axes[-1].set_xlabel("Frequency [Hz]")
        fig.suptitle(f"{title} — {plot_mode}", fontsize=14)

        return fig, axes

    @staticmethod
    def plot_frequency_domain_case(
        case_data,
        case_name="controller case",
        realization_idx=0,
        skip_zero_for_motion=True,
        skip_zero_for_power=False,
        power_plot_mode="magnitude",
        platform_dofs=None,
    ):
        """
        Plot frequency-domain results for one solved controller case.

        Parameters
        ----------
        case_data : dict
            One output dictionary from solve_controller_case(...), e.g.
            results_dict["PI_coupled"].
        case_name : str
            Name used in plot titles.
        realization_idx : int
            Wave realization index.
        skip_zero_for_motion : bool
            Whether to skip zero-frequency component for motion/force plots.
        skip_zero_for_power : bool
            Whether to skip zero-frequency component for power plot.
        power_plot_mode : {"magnitude", "real", "imag"}
            Plot mode for electrical power.
        platform_dofs : list[str] or None
            Full-coordinate platform DOFs to include. Default is
            ["DOF_0", "DOF_1", "DOF_2"].

        Returns
        -------
        plot_data : dict
            Dictionary containing extracted xarray data and matplotlib figures.
        """
        if platform_dofs is None:
            platform_dofs = ["DOF_0", "DOF_1", "DOF_2"]

        wec_fdom_full = case_data["wec_fdom_full"]
        pto_fdom = case_data["pto_fdom"]

        wec5_labels = [
            "Platform DOF_0",
            "Platform DOF_1",
            "Platform DOF_2",
            "Flap 1 rel. pitch / PTO",
            "Flap 2 rel. pitch / PTO",
        ]

        # ----------------------------
        # Extract frequency coordinates
        # ----------------------------
        wec_freq = wec_fdom_full["freq"].values
        pto_freq = pto_fdom["freq"].values

        # ----------------------------
        # Build 5-DOF WEC position
        # ----------------------------
        platform_pos = (
            wec_fdom_full["pos"]
            .isel(realization=realization_idx)
            .sel(influenced_dof=platform_dofs)
            .transpose("omega", "influenced_dof")
        )

        platform_pos = platform_pos.rename({"influenced_dof": "dof"})
        platform_pos = platform_pos.assign_coords(
            dof=["Platform DOF_0", "Platform DOF_1", "Platform DOF_2"]
        )

        pto_pos_as_wec = (
            pto_fdom["pos"]
            .isel(realization=realization_idx)
            .transpose("omega", "dof")
        )

        pto_pos_as_wec = pto_pos_as_wec.assign_coords(
            dof=["Flap 1 rel. pitch / PTO", "Flap 2 rel. pitch / PTO"]
        )

        wec5_pos = xr.concat(
            [platform_pos, pto_pos_as_wec],
            dim="dof",
        ).transpose("omega", "dof")

        # ----------------------------
        # Build 5-DOF WEC velocity
        # ----------------------------
        platform_vel = (
            wec_fdom_full["vel"]
            .isel(realization=realization_idx)
            .sel(influenced_dof=platform_dofs)
            .transpose("omega", "influenced_dof")
        )

        platform_vel = platform_vel.rename({"influenced_dof": "dof"})
        platform_vel = platform_vel.assign_coords(
            dof=["Platform DOF_0", "Platform DOF_1", "Platform DOF_2"]
        )

        pto_vel_as_wec = (
            pto_fdom["vel"]
            .isel(realization=realization_idx)
            .transpose("omega", "dof")
        )

        pto_vel_as_wec = pto_vel_as_wec.assign_coords(
            dof=["Flap 1 rel. pitch / PTO", "Flap 2 rel. pitch / PTO"]
        )

        wec5_vel = xr.concat(
            [platform_vel, pto_vel_as_wec],
            dim="dof",
        ).transpose("omega", "dof")

        # ----------------------------
        # PTO velocity
        # ----------------------------
        pto_vel = (
            pto_fdom["vel"]
            .isel(realization=realization_idx)
            .transpose("omega", "dof")
        )

        pto_labels = [str(v) for v in pto_vel["dof"].values]

        # ----------------------------
        # PTO force
        # ----------------------------
        pto_force = (
            pto_fdom["force"]
            .isel(realization=realization_idx)
            .transpose("omega", "dof")
        )

        # ----------------------------
        # Electrical power
        # ----------------------------
        pto_power_elec = (
            pto_fdom["power"]
            .isel(realization=realization_idx)
            .sel(type="elec")
            .transpose("omega", "dof")
        )

        # ----------------------------
        # Make plots
        # ----------------------------
        fig_pos, ax_pos = FOSWECM4EStudyBase.stem_subplots_from_array(
            freq=wec_freq,
            values=wec5_pos.values,
            labels=wec5_labels,
            title=f"{case_name}: 5-DOF WEC Position Spectrum",
            ylabel="|position|",
            skip_zero=skip_zero_for_motion,
            plot_mode="magnitude",
        )

        fig_vel, ax_vel = FOSWECM4EStudyBase.stem_subplots_from_array(
            freq=wec_freq,
            values=wec5_vel.values,
            labels=wec5_labels,
            title=f"{case_name}: 5-DOF WEC Velocity Spectrum",
            ylabel="|velocity|",
            skip_zero=skip_zero_for_motion,
            plot_mode="magnitude",
        )

        fig_pto_vel, ax_pto_vel = FOSWECM4EStudyBase.stem_subplots_from_array(
            freq=pto_freq,
            values=pto_vel.values,
            labels=pto_labels,
            title=f"{case_name}: PTO Velocity Spectrum",
            ylabel="|PTO velocity|",
            skip_zero=skip_zero_for_motion,
            plot_mode="magnitude",
        )

        fig_pto_force, ax_pto_force = FOSWECM4EStudyBase.stem_subplots_from_array(
            freq=pto_freq,
            values=pto_force.values,
            labels=pto_labels,
            title=f"{case_name}: PTO Force Spectrum",
            ylabel="|PTO force|",
            skip_zero=skip_zero_for_motion,
            plot_mode="magnitude",
        )

        fig_pto_power, ax_pto_power = FOSWECM4EStudyBase.stem_subplots_from_array(
            freq=pto_freq,
            values=pto_power_elec.values,
            labels=pto_labels,
            title=f"{case_name}: Electrical Power Spectrum",
            ylabel="Electrical power",
            skip_zero=skip_zero_for_power,
            plot_mode=power_plot_mode,
        )

        plt.show()

        return {
            "wec5_pos": wec5_pos,
            "wec5_vel": wec5_vel,
            "pto_vel": pto_vel,
            "pto_force": pto_force,
            "pto_power_elec": pto_power_elec,
            "fig_pos": fig_pos,
            "ax_pos": ax_pos,
            "fig_vel": fig_vel,
            "ax_vel": ax_vel,
            "fig_pto_vel": fig_pto_vel,
            "ax_pto_vel": ax_pto_vel,
            "fig_pto_force": fig_pto_force,
            "ax_pto_force": ax_pto_force,
            "fig_pto_power": fig_pto_power,
            "ax_pto_power": ax_pto_power,
        }
