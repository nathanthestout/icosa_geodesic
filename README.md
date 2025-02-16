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

#Example Usage 1:
```

import icosa_geodesic as geo
import numpy as np

# Generate a Class I geodesic dome with frequency 2, radius 10, using the "aa"
# method, a fraction of 1/2, and no rotation (defaults to cross symmetry).
g = geo.geodesic(geo_class=1, frequency=2, radius=10, method="aa",
                 fraction=1/2, rotation=([0,0,0]))

# Print the dome points and edge indices.
print(g.dome_points)
print(g.dome_edge_indices)

# Plot the dome edges.
geo.plot_3d_line_segments(np.array(g.dome_edges), 90, 90) 

```

#Example Usage 2: Points on geodesic sphere, rotated to pentad symmetry
```

import icosa_geodesic as geo
import numpy as np

# Generate a Class II geodesic dome with frequency 6, radius 10, using the "cab"
# method, a fraction of 1/2, and rotation to pentad symmetry.

#note: rotation is sequentially applied [x, y, z] in radians
        #For Triad symmetry use rotation = [np.pi-np.arccos(np.sqrt(5)/3)/2,0,0]
        #For Cross symmetry use rotation = [0,0,0]
        #For Pentad symmetry use rotation = [(np.arctan(2)/2-np.pi/2,0,0),0,0]

g = geo.geodesic(geo_class=2, frequency=6, radius=10, method="cab",
                 fraction=1/2, rotation=([np.arctan(2)/2-np.pi/2,0,0,0,0]))

# Plot the sphere points.
geo.plot_3d_line_segments(np.array(g.sphere_points), 90, 90) 

```

#Example Usage 3: Radial geodesic projection (tolerance needed)
```

#since we're using floating point math, a tolerance is needed (default = 0.0000001)
#to determin whether to consider two points as identical
#Use a tolerance of appx 0.1 to find the average vertices on radial projection, otherwise
#redundant points will be generated and the edge generation method will yield bad results

g = geodesic(geo_class=2, frequency=6, radius=10, method="radial",
                 fraction=1/2, rotation=([np.arctan(2)/2-np.pi/2,0,0]), tolerance = 0.1)

# Plot the dome edges.
geo.plot_3d_line_segments(np.array(g.dome_edges), 90, 90)

```

#Example Usage 4: Truss analysis of loaded geodesic dome
See GeoTruss.py
