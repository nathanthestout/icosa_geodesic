"""
Example use of Trusspy to analyze and visualize forces and/or stress in a 
Geodesic Dome
Uses icosa_geodesic library to generate geodesic geometry (github project)
Uses TrussPy library to analyze stress
MIT License

Copyright (c) [2025] [Nathaniel Paul Brand]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Libraries (not included, download source from github or use e.g. pip to install)
icosa_geodesic is licensed MIT License
TrussPy is licensed GNU V3
Numpy is licensed modified BSD
Usage of licensed code must comply with the relevant licenses
"""


import trusspy as tp
import icosa_geodesic
import numpy as np

##############################################################################
# Dome Geometry Variables
##############################################################################
"""
Units used in this example are in, lbf, psi 
(inches, pounds-force, pounds-force per square inch)

Any consistant unit system may be used, e.g. m, N, Pa (meter, Newton, Pascal)
Note that Trusspy does not account for the mass of the struts (a.k.a. elements)
"""


r_dome = 180  # inches
dome_frequency = 5
dome_class = 2
dome_fraction = 2/3

# vector, rotation amounts [x, y, z] in radians
dome_rotation_vector = [0, 0, 0]
# For Triad symmetry use [np.pi-np.arccos(np.sqrt(5)/3)/2,0,0]
# For Cross symmetry use [0,0,0]
# For Pentad symmetry use [[np.arctan(2)/2-np.pi/2,0,0],0,0]

# options include "aa", "ab", "aab", "ba", "bb", "bba", "ca", "cb", "cab", radial
dome_subdivision_method = "aab"
A_strut = 1.5*7.25  # Area in²
# E-Modulus psi, steel = 30*10**6, aluminum = 10*10**6, Doug Fir-Larch (lumber) = 580000
E_strut = 580000

# Loading Conditions
snow_load = 50  # lbf/sqft
snow_load_total = snow_load*np.pi*r_dome**2/(12**2)  # lbf

hanging_load_total = 5000  # lbf total, divided by the number of points
# inches, points above this z height will be loaded with a hanging load
hanging_load_plane = r_dome*.8

# Wind load Per AWWA D100-96; Section 3.1.4, Eq 3-1
# Note: Wind load applied from negative X axis
# Wind load assumed to be equal for all windward side nodes
# very conservative estimate for Cd. Note Cd for a sphere is 0.47, 
# a hemisphere(not on a flat plate/on the ground) is 0.38
Cd = 0.5
# Wind speed in mph, maximum 3 second wind gust condition in IBC 2018 is 170mph
#For a less conservative estimate use 110mph
wind_speed = 110  
wind_pressure = 30/144 * Cd * (wind_speed / 100)**2  # PSI

#for km/h wind uncomment the lines below:
#wind_speed = 180 #km/h
#wind_pressure = 6894.76*30/144 * Cd * (wind_speed / 160.934)**2

wind_load_total = dome_fraction*wind_pressure*np.pi*r_dome**2
tolerance = .01  # in case points aren't exactly where we expect them

fixed_plane = r_dome*.05  # inches, points below this z height will be fixed

# Generate Geodesic Dome
g = icosa_geodesic.geodesic(dome_class, dome_frequency, r_dome,
                            dome_subdivision_method, dome_fraction, dome_rotation_vector)

###############################################
# Trusspy Analysis
###############################################
M = tp.Model(logfile=True)

# create nodes
"""IMPORTANT NOTE: TRUSSPY LABELS SHOULD START AT 1, NOT 0, e.g. use i+1 as label"""

with M.Nodes as MN:
    for i in range(len(g.dome_points)):
        MN.add_node(i+1, coord=g.dome_points[i])


# create elements
element_type = 1  # truss
material_type = 1  # linear-elastic
"""IMPORTANT NOTE: TRUSSPY LABELS SHOULD START AT 1, NOT 0"""
with M.Elements as ME:
    for i in range(int(len(g.dome_edge_indices))):
        ME.add_element(i+1, conn=g.dome_edge_indices[i]+1)
    # ME.add_element(0, conn=[1,2])
    ME.assign_geometry("all", [A_strut])  # Area in²
    ME.assign_material("all", [E_strut])  # E-Modulus psi
    ME.assign_etype("all", element_type)
    ME.assign_mtype("all", material_type)

# create displacement (U) boundary conditions
"""IMPORTANT NOTE: TRUSSPY LABELS SHOULD START AT 1, NOT 0"""
with M.Boundaries as MB:
    for i in range(len(g.dome_points)):
        if (g.dome_points[i][2] <= fixed_plane+tolerance):
            MB.add_bound_U(i+1, (0, 0, 0))  # Fixed Support

# create forces
snow_load_point = snow_load_total / len(g.dome_points)
hanging_load_point = hanging_load_total / \
    len([p for p in g.dome_points if p[2] >= hanging_load_plane-tolerance])
wind_load_point = wind_load_total/len([p for p in g.dome_points if p[0] <= 0])

"""IMPORTANT NOTE: TRUSSPY LABELS SHOULD START AT 1, NOT 0"""

with M.ExtForces as MF:
    for i in range(len(g.dome_points)):
        force_vector = [0, 0, 0]
        if (g.dome_points[i][2] > fixed_plane+tolerance):
            force_vector[2] += -snow_load_point
            if (g.dome_points[i][2] >= hanging_load_plane-tolerance):
                force_vector[2] += -hanging_load_point
            if (g.dome_points[i][0] <= tolerance):
                force_vector[0] += wind_load_point
            MF.add_force(
                i+1, (force_vector[0], force_vector[1], force_vector[2]))


# Solver Settings - adjust as needed to get the solver to work

# TrussPy is a nonlinear truss analysis - it incrementally searches for the equilbrium
# by scaling given external forces. If you are not interested in the deformation path,
# set the number of increments ``incs`` to one, the allowed increase in displacements to
# infinity or a high value ``du`` and the increase of the load-proportionality-factor
#  ``dlpf`` to one.
M.Settings.incs = 1  # evaluate only one increment
M.Settings.du = 1e8  # turn off max. incremental displacement
M.Settings.dlpf = 1  # apply the load proportionality factor in one increment

# M.Settings.stepcontrol = True
# M.Settings.maxfac = 8

# M.Settings.ftol = 8
# M.Settings.xtol = 8
# M.Settings.nfev = 8

# M.Settings.dxtol = 1.25
M.build()

# Plot undeformed
fig, ax = M.plot_model(view="xy", force_scale=0.01, inc=0)

# run simulation
M.run()

# Plot loaded condition, forces in struts
fig, ax = M.plot_model(
    view="xy",
    contour="force",
    lim_scale=1.2,
    force_scale=0.00,
    nodesize=3,
    inc=1,
)

# Plot loaded condition, stress in struts
fig, ax = M.plot_model(
    view="xy",
    contour="stress",
    lim_scale=1.2,
    force_scale=0.00,
    nodesize=3,
    inc=1,
)
