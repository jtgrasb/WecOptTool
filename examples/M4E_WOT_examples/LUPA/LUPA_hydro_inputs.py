# Imports
import wecopttool as wot
import pygmsh
import gmsh
import capytaine as cpt

# Waves and frequencies definitions
# Frequency object
wavefreq    = 0.4 # Hz
f1          = wavefreq
nfreq       = 2
freq        = wot.frequency(f1, nfreq, False) # False -> no zero frequency

# Define waves object
amplitude   = 0.005 # m
phase       = 0 # degrees
wavedir     = 0 # degrees
waves       = wot.waves.regular_wave(f1, nfreq, wavefreq, amplitude, phase, wavedir)

# Mesh 
# Spar
mesh_size_factor = 0.1
r1 = 0.45/2  # body
r2 = 0.45  # plate
r3 = 0.10/2  # bar
h1 = 1.20  
h2 = 0.01
h3a = 3.684 - 2.05
submergence = 2.05 - h1 - h2
with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
    body = geom.add_cylinder([0, 0, 0], [0, 0, -h1], r1)
    geom.translate(body, [0, 0, -submergence])
    plate = geom.add_cylinder([0, 0, 0], [0, 0, -h2], r2)
    geom.translate(plate, [0, 0, -(submergence+h1)])
    bar = geom.add_cylinder([0, 0, h3a], [0, 0, -(h3a+submergence)], r3)
    geom.boolean_union([bar, body, plate])
    mesh_spar = geom.generate_mesh()

# Float/Torus
mesh_size_factor = 0.3
r1 = 1.0/2  # top radius
r2 = 0.4/2  # bottom radius
h1 = 0.5  
h2 = 0.21
freeboard = 0.3
r3 = 0.05  # hole radius
with pygmsh.occ.Geometry() as geom:
    gmsh.option.setNumber('Mesh.MeshSizeFactor', mesh_size_factor)
    cyl = geom.add_cylinder([0, 0, 0], [0, 0, -h1], r1)
    cone = geom.add_cone([0, 0, -h1], [0, 0, -h2], r1, r2)
    geom.translate(cyl, [0, 0, freeboard])
    geom.translate(cone, [0, 0, freeboard]) 
    tmp = geom.boolean_union([cyl, cone])
    bar = geom.add_cylinder([0, 0, 10], [0,0,-20], r3)
    geom.boolean_difference(tmp, bar)
    mesh_float = geom.generate_mesh()

# Mass and inertia
m_spar = 175.536
J_spar = 250.4558
m_float = 248.721
J_float = 65.3344

# provided by OSU
# float_mass_properties = {
#     'mass': 248.721,
#     'CG': [0.01, 0, 0.06],
#     'MOI': [66.1686, 65.3344, 17.16],
# }

# spar_mass_properties = {
#     'mass': 175.536,
#     'CG': [0, 0, -1.3],
#     'MOI': [253.6344, 250.4558, 12.746],
# }

# Pre-clip meshes at the waterplane (z=0) so compute_hydrostatics() uses only the submerged volume.
# User is responsible for clipping (not done inside HydroLinearMCKF).
fb_spar = cpt.FloatingBody(mesh=mesh_spar, name="spar")
fb_spar = fb_spar.immersed_part()
mesh_spar_clipped = fb_spar.mesh  # clipped Capytaine mesh for hydrostatics/BEM

fb_float = cpt.FloatingBody(mesh=mesh_float, name="float")
fb_float = fb_float.immersed_part()
mesh_float_clipped = fb_float.mesh  # clipped Capytaine mesh for hydrostatics/BEM

# body_inputs: spar first (key=1) then torus (key=2) to match MBD Cartesian ordering [spar, float]
body_inputs = {
    1: {
        "name": "spar",
        "mesh": mesh_spar_clipped,
        "mesh_reference": "absolute",
        "inertia_diag": [m_spar, m_spar, m_spar, J_spar, J_spar, J_spar],
    },
    2: {
        "name": "torus",
        "mesh": mesh_float_clipped,
        "mesh_reference": "absolute",
        "inertia_diag": [m_float, m_float, m_float, J_float, J_float, J_float],
    },
}

water_depth = 2.7 # depth of wave flume

m0 = [m_spar, m_float]
J0 = [J_spar, J_float]