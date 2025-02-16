# icosa_geodesic
A python library for generating geodesic spheres and domes


Author: Nathan Brand, 2025

A Python implementation of the geodesic sphere/dome generation (vertices) methods
described by Christopher J. Kitrick in the International Journal of Space 
Structures, Volume 5, Nos. 3 & 4, 1990. 



This code generates geodesic domes based on an icosahedron, implementing both 
Class I and Class II subdivisions.  All methods generate points from a 
Schwarz Triangle (1/6th of an icosahedron face) projected radially onto a sphere.
This projected triangle is referred to as the spherical LCD (lowest common 
denominator) triangle.

For geodesic frequencies >=2

The Z-axis is defined as the radial axis pointing from the origin to the right 
angle of the LCD triangle (see Fig. 12 in Kitrick's paper). The other vertices 
of the LCD triangle lie on the XZ and YX planes, corresponding to the long and 
short edges, respectively.

Helper functions for scaling, rotation, and plotting included for convenience



Changelog:
v0.0.1 - Initial testing
v0.0.2 - Functional dome generation
v0.0.3 - Re-indexed dome points and vertices for easier truss analysis

Tested in Jupyter Notebook
python: 3.12.7
numpy: 1.26.4
scipy: 1.13.1
matplotlib: 3.10.0

MIT License

Example Usage:
import icosa_geodesic as geo
import numpy as np

# Generate a Class I geodesic dome with frequency 2, radius 10, using the "aa"
# method, a fraction of 1/2, and no rotation.
g = geo.geodesic(geo_class=1, frequency=2, radius=10, method="aa",
                 fraction=1/2, rotation=([0,0,0]))

# Print the dome points and edge indices.
print(g.dome_points)
print(g.dome_edge_indices)

# Plot the dome edges.
geo.plot_3d_line_segments(np.array(g.dome_edges), 90, 90) 