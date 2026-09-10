"""
MACHINERY DYNAMICS LECTURES  (41514)
MEK - DEPARTMENT OF MECHANICAL ENGINEERING
DTU - TECHNICAL UNIVERSITY OF DENMARK

Copenhagen, March 30th, 2021

Ilmar Ferreira Santos

ROTATING MACHINES -- NATURAL FREQUENCIES AND MODES

EXPERIMENTAL RESULTS
13.0 (horizontal)
14.9 (vertical)
33.6 (horizontal)
43.0 (horizontal)
46.0 (vertical)

Converted from MATLAB to Python using ROSS (Rotordynamic Open-Source Software)
12-element version

Using proper ROSS pattern with n_link:
- Bearing at shaft node with n_link to housing node
- Support bearing at housing node (stiffness to ground)
- PointMass at housing node (housing mass)
"""
import sys
sys.path.append(r"C:\Users\Murillo\Documents\ROSS\ross_230\ross")
import numpy as np
import pandas as pd
import ross as rs
from scipy import linalg as la

# =========================================================================
#   DEFINITION OF THE STRUCTURE OF THE MODEL
# =========================================================================
NE = 12   # number of shaft elements
ND = 2    # number of discs
NM = 2    # number of bearings
CD1 = 3   # node - disc 1 (Python 0-indexed: MATLAB CD1=4)
CD2 = 9   # node - disc 2 (Python 0-indexed: MATLAB CD2=10)
CMM1 = 0  # shaft node - bearing 1 (Python 0-indexed: MATLAB CMM1=1)
CMM2 = 12 # shaft node - bearing 2 (Python 0-indexed: MATLAB CMM2=13)

# Housing nodes (extra nodes beyond shaft nodes)
# Shaft has NE+1 = 13 nodes (0 to 12)
# Housing nodes are added as extra nodes
HOUSING1 = NE + 1  # node 13
HOUSING2 = NE + 2  # node 14

# =========================================================================
#   CONSTANTS
# =========================================================================
E = 2.0e11    # elasticity modulus [N/m^2]
RAco = 7800   # steel density [kg/m^3]
RAl = 2770    # aluminum density [kg/m^3]

# =========================================================================
#   OPERATIONAL CONDITIONS
# =========================================================================
Omega = 0 * 2 * np.pi  # angular velocity [rad/s]
Omegarpm = Omega * 60 / 2 / np.pi

# =========================================================================
#   GEOMETRY OF THE ROTATING MACHINE
# =========================================================================

# (A) DISCS
Rd = 6 / 100                                # external radius of the disc [m]
Ri = (5 / 2) / 1000                         # internal radius of the disc [m]
espD = 1.1 / 100                            # disc thickness [m]
MasD = np.pi * Rd**2 * espD * RAl - np.pi * Ri**2 * espD * RAl  # disc mass [kg]
Id = (1/4 * Rd**2 + 1/12 * espD**2) * np.pi * Rd**2 * espD * RAl \
   - (1/4 * Ri**2 + 1/12 * espD**2) * np.pi * Ri**2 * espD * RAl  # transversal MOI [kg*m^2]
Ip = 1/2 * Rd * Rd * (np.pi * Rd**2 * espD * RAl) \
   - 1/2 * Ri * Ri * (np.pi * Ri**2 * espD * RAl)  # polar MOI [kg*m^2]

print(f"Disc mass: {MasD:.4f} kg")
print(f"Disc Id:   {Id:.6f} kg*m^2")
print(f"Disc Ip:   {Ip:.6f} kg*m^2")

# (B) BEARINGS
MasM = 0.40698   # bearing mass [kg] (housing + ball bearings)
h = 1 / 1000     # beam thickness [m]
b = 28.5 / 1000  # beam width [m]
Area = b * h      # beam cross section area [m^2]
I_beam = b * h**3 / 12  # beam moment of inertia of area [m^4]
lr = 7.5 / 100   # beam length [m]
Kty0 = 2 * 12 * E * I_beam / lr**3  # equivalent beam flexural stiffness [N/m]
Ktz0 = 2 * E * Area / lr            # equivalent bar stiffness [N/m]

print(f"\nBearing stiffness Kty0: {Kty0:.2f} N/m")
print(f"Bearing stiffness Ktz0: {Ktz0:.2f} N/m")
print(f"Bearing housing mass: {MasM:.4f} kg")

# (C) SHAFT
Rext = (5 / 2) / 1000  # shaft external radius [m]
Rint = (0 / 2) / 1000  # shaft internal radius [m]

# =========================================================================
#   CREATE MATERIAL
# =========================================================================
aluminum = rs.Material(name="aluminum", rho=RAl, E=70e9, G_s=27e9)
steel = rs.Material(name="steel_DTU", rho=RAco, E=E, G_s=8.0e10)

# =========================================================================
#   CREATE SHAFT ELEMENTS
# =========================================================================
l = np.zeros(NE)
l[0]  = 0.140 / 3
l[1]  = 0.140 / 3
l[2]  = 0.140 / 3
l[3]  = 0.205 / 6
l[4]  = 0.205 / 6
l[5]  = 0.205 / 6
l[6]  = 0.205 / 6
l[7]  = 0.205 / 6
l[8]  = 0.205 / 6
l[9]  = 0.090 / 3
l[10] = 0.090 / 3
l[11] = 0.090 / 3

shaft_elements = []
for i in range(NE):
    shaft_el = rs.ShaftElement(
        L=l[i],
        idl=2 * Rint,
        odl=2 * Rext,
        material=steel,
        n=i,
        shear_effects=True,
        rotary_inertia=True,
        gyroscopic=True,
    )
    shaft_elements.append(shaft_el)

print(f"\nNumber of shaft elements: {len(shaft_elements)}")
print(f"Total shaft length: {np.sum(l):.4f} m")
print(f"Shaft nodes: 0 to {NE} ({NE+1} nodes)")
print(f"Housing nodes: {HOUSING1}, {HOUSING2}")

# =========================================================================
#   CREATE DISK ELEMENTS
# =========================================================================
disk1 = rs.DiskElement.from_geometry(
    n=CD1, material=aluminum, width=espD, i_d=2 * Ri, o_d=2 * Rd
)
disk2 = rs.DiskElement.from_geometry(
    n=CD2, material=aluminum, width=espD, i_d=2 * Ri, o_d=2 * Rd
)

print(f"\nDisk 1 at node {CD1}")
print(f"Disk 2 at node {CD2}")

# =========================================================================
#   CREATE BEARING ELEMENTS (proper ROSS pattern with n_link)
# =========================================================================
# Bearing 1: connects shaft node CMM1 (0) to housing node HOUSING1 (13)
bearing1 = rs.BearingElement(
    n=CMM1, 
    n_link=HOUSING1,
    kxx=Ktz0,  # horizontal stiffness (MATLAB Ktz1)
    kyy=Kty0,  # vertical stiffness (MATLAB Kty1)
    cxx=0.0,
    cyy=0.0,
    tag="Bearing1"
)

# Support bearing 1 at housing node HOUSING1 (stiffness to ground)
support1 = rs.BearingElement(
    n=HOUSING1,
    kxx=Ktz0,
    kyy=Kty0,
    cxx=0.0,
    cyy=0.0,
    tag="Support1"
)

# Bearing 2: connects shaft node CMM2 (12) to housing node HOUSING2 (14)
bearing2 = rs.BearingElement(
    n=CMM2,
    n_link=HOUSING2,
    kxx=Ktz0,
    kyy=Kty0,
    cxx=0.0,
    cyy=0.0,
    tag="Bearing2"
)

# Support bearing 2 at housing node HOUSING2 (stiffness to ground)
support2 = rs.BearingElement(
    n=HOUSING2,
    kxx=Ktz0,
    kyy=Kty0,
    cxx=0.0,
    cyy=0.0,
    tag="Support2"
)

print(f"\nBearing 1 at shaft node {CMM1} -> housing node {HOUSING1}")
print(f"Support 1 at housing node {HOUSING1}")
print(f"Bearing 2 at shaft node {CMM2} -> housing node {HOUSING2}")
print(f"Support 2 at housing node {HOUSING2}")

# =========================================================================
#   CREATE POINT MASSES AT HOUSING NODES
# =========================================================================
point_mass1 = rs.PointMass(n=HOUSING1, m=MasM, tag="HousingMass1")
point_mass2 = rs.PointMass(n=HOUSING2, m=MasM, tag="HousingMass2")

print(f"\nPointMass 1 at housing node {HOUSING1}: {MasM:.4f} kg")
print(f"PointMass 2 at housing node {HOUSING2}: {MasM:.4f} kg")

# =========================================================================
#   ASSEMBLE ROTOR
# =========================================================================
rotor = rs.Rotor(
    shaft_elements=shaft_elements,
    disk_elements=[disk1, disk2],
    bearing_elements=[bearing1, support1, bearing2, support2],
    point_mass_elements=[point_mass1, point_mass2],
)

print(f"\nRotor assembled successfully!")
print(f"Number of nodes: {len(rotor.nodes)}")
print(f"Number of DOFs:  {rotor.ndof}")

# =========================================================================
#   CONVERT TO 4 DOF and run MODAL ANALYSIS
# =========================================================================
from ross.utils import convert_6dof_to_4dof
rotor_4dof = convert_6dof_to_4dof(rotor)

print(f"\nConverted to 4 DOF model")
print(f"Number of DOFs:  {rotor_4dof.ndof}")

# Debug: check mass matrix singularity
M_test = rotor_4dof.M()
print(f"\nMass matrix shape: {M_test.shape}")
print(f"Mass matrix min eigenvalue: {np.min(np.abs(np.linalg.eigvalsh(M_test))):.6e}")
print(f"Mass matrix condition number: {np.linalg.cond(M_test):.6e}")

# =========================================================================
#   MODAL ANALYSIS
# =========================================================================
print("\n" + "=" * 60)
print("RUNNING MODAL ANALYSIS")
print("=" * 60)

modal = rotor_4dof.run_modal(speed=Omega)

wn_hz = modal.wn / (2 * np.pi)

print("\nFirst 10 Natural frequencies (Hz):")
for i, freq in enumerate(wn_hz[:10]):
    print(f"  Mode {i + 1}: {freq:.2f} Hz")

print("\n" + "=" * 60)
print("COMPARISON WITH EXPERIMENTAL RESULTS")
print("=" * 60)
print("Experimental results (Hz): 13.0, 14.9, 33.6, 43.0, 46.0")
print("ROSS results (Hz):")
for i, freq in enumerate(wn_hz[:10]):
    print(f"  Mode {i + 1}: {freq:.2f} Hz")

# =========================================================================
#   PRINT TABLE OF RESULTS
# =========================================================================
print("\n" + "=" * 60)
print("DETAILED MODAL RESULTS")
print("=" * 60)
print(modal.format_table())

# =========================================================================
#   PLOT MODE SHAPES
# =========================================================================
try:
    for mode_num in range(1, min(7, len(wn_hz) + 1)):
        try:
            fig = modal.plot_mode_3d(mode=mode_num)
            fig.update_layout(title=f"Mode {mode_num} - f={wn_hz[mode_num - 1]:.2f} Hz")
            fig.show()
        except Exception as e:
            print(f"  Mode {mode_num}: Could not plot 3D ({e})")
            data = modal.data_mode(mode=mode_num)
            print(f"  Mode shape data available at modal.data_mode(mode={mode_num})")
except Exception as e:
    print(f"\nNote: 3D mode plotting requires plotly display: {e}")
    print("Use modal.data_mode(mode) to get mode shape data")

# =========================================================================
#   RESULTS DATAFRAME
# =========================================================================
results_data = {
    "Mode": range(1, len(wn_hz) + 1),
    "wn (rad/s)": modal.wn,
    "wd (rad/s)": modal.wd,
    "Frequency (Hz)": wn_hz,
    "Damping ratio": modal.damping_ratio,
    "Log decrement": modal.log_dec,
}

df_results = pd.DataFrame(results_data)
print("\n" + "=" * 60)
print("RESULTS DATAFRAME")
print("=" * 60)
print(df_results.to_string(index=False))

# =========================================================================
#   VALIDATION
# =========================================================================
print("\n" + "=" * 60)
print("VALIDATION")
print("=" * 60)
print("The experimental results from DTU are:")
print("  13.0 Hz (horizontal)")
print("  14.9 Hz (vertical)")
print("  33.6 Hz (horizontal)")
print("  43.0 Hz (horizontal)")
print("  46.0 Hz (vertical)")
print()
print("Note: ROSS uses 6 DOFs per node (including axial and torsional),")
print("while the MATLAB code uses 4 DOFs (lateral only).")
print("Using proper ROSS pattern with n_link for bearing housing modeling.")