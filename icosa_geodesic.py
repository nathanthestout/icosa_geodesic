"""
Icosa_Geodesic

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
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting
import matplotlib.colors as mcolors
from scipy.spatial import cKDTree

# ==================================================================
# Helper Functions
# ==================================================================


def plot_3d_points(points, tilt=90, pan=90, title="3D Point Cloud"):
    """
    Plots a numpy array of 3D points using matplotlib.

    Args:
        points: A numpy array of shape (N, 3) where N is the number of points.
        title: The title of the plot.
    """

    if points.shape[1] != 3:
        raise ValueError("Points array must have shape (N, 3).")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D subplot

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    ax.scatter(x, y, z)  # Use scatter for 3D points

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.view_init(tilt, pan)
    plt.show()


def plot_3d_line_segments(segments, pan=0, tilt=0, tolerance=0.00001):
    """
    Plots an array of 3D line segments in 3D, coloring segments of similar 
    length (within tolerance) the same color.

    Args:
    - segments (list of tuples): Each tuple contains two points (start, end), 
    where each point is a (x, y, z) tuple.
    - tolerance (float): Tolerance for grouping similar-length segments with 
    the same color.
    - pan (float): Azimuth angle for 3D plot (0 to 360 degrees).
    - tilt (float): Elevation angle for 3D plot (-90 to 90 degrees).
    """

    # Initialize 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Calculate lengths of all segments
    lengths = []
    for start, end in segments:
        length = np.linalg.norm(np.array(end) - np.array(start))
        lengths.append(length)

    # Group segments by similar lengths within tolerance
    unique_lengths = []
    for length in lengths:
        for unique_length in unique_lengths:
            if abs(length - unique_length) < tolerance:
                break
        else:
            unique_lengths.append(length)

    # Generate colors for each unique length group
    colors = list(mcolors.TABLEAU_COLORS.values())  # You can use any color map
    color_map = {length: colors[i % len(colors)]
                 for i, length in enumerate(unique_lengths)}

    # Plot each line segment with the corresponding color
    for i, (start, end) in enumerate(segments):
        length = lengths[i]
        for unique_length in unique_lengths:
            if abs(length - unique_length) < tolerance:
                color = color_map[unique_length]
                break
        start_x, start_y, start_z = start
        end_x, end_y, end_z = end
        ax.plot([start_x, end_x], [start_y, end_y],
                [start_z, end_z], color=color)

    # Set the view angle
    ax.view_init(elev=tilt, azim=pan)

    # Show the plot
    plt.show()


def unique_points(points, tolerance=1e-6):
    """
    Removes duplicate points from an n-dimensional numpy array using cKDTree,
    replacing duplicates with their average.

    Args:
        points: A numpy array of shape (N, D) where N is the number of points 
        and D is the dimension.
        tolerance: The distance tolerance for considering two points as duplicates.

    Returns:
        A numpy array with unique points (averages of duplicates)
        Returns an empty array if the input is empty.
    """

    if points.size == 0:  # Handle empty input
        return np.array([]), np.array([])

    n_points = points.shape[0]
    tree = cKDTree(points)  # Build the KD tree

    unique_points = []
    unique_indices = []
    is_unique = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        if is_unique[i]:  # Only check if the current point is still considered unique
            neighbors_indices = tree.query_ball_point(points[i], r=tolerance)
            neighbors_indices = np.array(neighbors_indices)

            # Calculate average of neighbors (including self)
            average_point = np.mean(points[neighbors_indices], axis=0)
            unique_points.append(average_point)

            # Store the index of the "representative" point (first one encountered)
            unique_indices.append(i)

            # Mark duplicates (including self) as non-unique
            is_unique[neighbors_indices] = False

    unique_points_array = np.array(unique_points)
    unique_indices = np.array(unique_indices)
    return unique_points_array


def reflection_matrix_3d(plane_point, plane_normal):
    """
    Calculates the 4x4 transformation matrix for mirroring a 3D point
    about a plane defined by a point and a normal vector.

    Args:
        plane_point: A numpy array of shape (3,) representing a point on the plane.
        plane_normal: A numpy array of shape (3,) representing the normal vector of the plane.

    Returns:
        A numpy array of shape (4, 4) representing the 4x4 transformation matrix.
        Returns None if the normal vector is a zero vector.
    """

    # Normalize the normal vector
    norm = np.linalg.norm(plane_normal)
    if norm == 0:
       return None  # Or raise an exception, as a zero normal is invalid.
    n = plane_normal / norm

    # Create the reflection matrix components
    a = n[0]
    b = n[1]
    c = n[2]

    # Calculate d (distance from origin to plane)
    d = -np.dot(plane_point, n)

    # Construct the 4x4 reflection matrix
    reflection_matrix = np.array([
        [1 - 2*a*a, -2*a*b, -2*a*c, -2*a*d],
        [-2*a*b, 1 - 2*b*b, -2*b*c, -2*b*d],
        [-2*a*c, -2*b*c, 1 - 2*c*c, -2*c*d],
        [0, 0, 0, 1]
    ])

    return reflection_matrix


def rotation_matrix_3d_about_line(point1, point2, angle_rad):
    """
    Generates a 4x4 rotation matrix for rotating points around a line segment in 3D space.

    Args:
        point1: NumPy array (3,) representing the first point of the line segment.
        point2: NumPy array (3,) representing the second point of the line segment.
        angle_rad: The rotation angle in radians (using the right-hand rule).

    Returns:
        A 4x4 NumPy array representing the rotation matrix, or None if the line segment is degenerate (points are too close).
    """

    v = np.array(point2) - np.array(point1)
    mag_v = np.linalg.norm(v)

    if mag_v < 1e-6:  # Check for a near-zero vector (degenerate case)
        return None

    v = v / mag_v  # Normalize the direction vector

    a = point1[0]
    b = point1[1]
    c = point1[2]

    x = v[0]
    y = v[1]
    z = v[2]

    cos_theta = np.cos(angle_rad)
    sin_theta = np.sin(angle_rad)

    # Construct the rotation matrix (Rodrigues' rotation formula adapted for homogeneous coordinates)
    rotation_matrix = np.array([
        [x*x*(1 - cos_theta) + cos_theta, x*y*(1 - cos_theta) - z*sin_theta, x*z*(1 - cos_theta) +
         y*sin_theta, (a*(y*y + z*z) - x*(b*y + c*z))*(1-cos_theta) + (b*z - c*y)*sin_theta],
        [y*x*(1 - cos_theta) + z*sin_theta, y*y*(1 - cos_theta) + cos_theta, y*z*(1 - cos_theta) -
         x*sin_theta, (b*(x*x + z*z) - y*(a*x + c*z))*(1-cos_theta) + (c*x - a*z)*sin_theta],
        [z*x*(1 - cos_theta) - y*sin_theta, z*y*(1 - cos_theta) + x*sin_theta, z*z*(1 - cos_theta) +
         cos_theta, (c*(x*x + y*y) - z*(a*x + b*y))*(1-cos_theta) + (a*y - b*x)*sin_theta],
        [0, 0, 0, 1]
    ])

    return rotation_matrix


def scaling_matrix(scale_factor):
    """
    Generates a 4x4 transformation matrix for uniform (equilateral) scaling
    about the origin in 3D space.

    Args:
        scale_factor: The scaling factor (a positive number).

    Returns:
        A numpy array of shape (4, 4) representing the 4x4 scaling matrix.
    """

    if scale_factor <= 0:
        raise ValueError("Scale factor must be positive.")

    scaling_matrix = np.array([
        [scale_factor, 0, 0, 0],
        [0, scale_factor, 0, 0],
        [0, 0, scale_factor, 0],
        [0, 0, 0, 1]
    ])

    return scaling_matrix


def get_convex_mesh_edges_nearest_neighbors(points):
    """
    Returns the edges and indices of a convex mesh given an array of 3D points,
    connecting nearest neighbors with distance ratio constraints.

    Args:
        points: A numpy array of shape (N, 3) where N is the number of points.

    Returns:
        A tuple containing two lists:
            - edges: A list of tuples, where each tuple contains two 3D points
                     representing an edge.
            - edge_indices: A list of tuples, where each tuple contains the
                            indices of the two points forming the edge in the
                            original 'points' array.
        Returns empty lists if there's an issue (e.g., fewer than 4 points).
    """

    n_points = points.shape[0]
    if n_points < 4:  # Need at least 4 points for a 3D convex hull
        return [], []

    tree = cKDTree(points)  # Use cKDTree for efficient nearest neighbor search
    edges = []
    edge_indices = []

    for i in range(n_points):
        distances, indices = tree.query(points[i], k=min(
            n_points, 7))  # Query k nearest neighbors (max 7)

        # Remove self-neighbor (distance 0 at index 0)
        distances = distances[1:]
        indices = indices[1:]

        num_neighbors = len(distances)

        if num_neighbors < 2:  # Need at least two neighbors.
            continue

        closest_dist = distances[0]
        farthest_dist = distances[-1]

        if farthest_dist / closest_dist >= 1.6:
            continue

        for j in range(num_neighbors):
            neighbor_index = indices[j]
            p1 = tuple(points[i])
            p2 = tuple(points[neighbor_index])
            edge = tuple(sorted((p1, p2)))  # Consistent ordering

            # Check if this edge already exists (avoid duplicates)
            # Check against sorted tuples
            if edge not in [tuple(sorted(e)) for e in edges]:
                edges.append(edge)
                # Store indices, sorted
                edge_indices.append(tuple(sorted((i, neighbor_index))))

    return edges, edge_indices
# ==================================================================
# Geodesic Calculation Classes
# ==================================================================


class lcd():
    """
    Least Common Denominator triangle for the purposes of generating the points of a geodesic sphere via various subdivision methods. For full explaination refer to: A Unified Approach to Class I, II & III Geodesic Domes
    September 1990 International Journal of Space Structures Vol. 5(Nos. 3 & 4):223-246
    DOI:10.1177/026635119000500307
    
    """

    def __init__(self, geo_class, frequency, tolerance = 0.0000001):
        """
        Generate the LCD geometry for one segment of a face of an icosahedron, given a class and frequency.
        """
        if (frequency < 2 or geo_class not in [1, 2]):
            raise ValueError(
                'Bad arguments for geodesic class or frequency. geo_class = 1 or 2, frequency (int) >= 2')
        # elif(geo_class == 2 and frequency%2):
        #    raise ValueError('Bad arguments for geodesic class or frequency. for geo_class = 2, frequency = 2*n (even number)')
        self.geo_class = geo_class
        self.frequency = int(frequency)
        self.tolerance = tolerance
        self.A_rad = np.pi/5
        self.B_rad = np.pi/3
        self.C_rad = np.pi/2
        self.b_rad = np.arctan(2)/2
        self.a_rad = np.arcsin(
            np.sin(self.A_rad)*np.sin(self.b_rad)/np.sin(self.B_rad))
        self.c_rad = np.arcsin(np.sin(self.b_rad)/np.sin(self.B_rad))
        if (self.geo_class == 1):
            self.icosa_b = 2*self.frequency
            self.icosa_c = 0
        elif (self.geo_class == 2):
            # modification to match frequency to standard definition
            self.frequency = int(self.frequency/2)
            self.icosa_b = self.frequency
            self.icosa_c = self.icosa_b
        self.t = self.icosa_b**2+self.icosa_b*self.icosa_c+self.icosa_c**2
        # self.sphere_vertices = 10*self.t+2
        # self.sphere_edges = 20*self.t
        # self.sphere_faces = 30*self.t
        self.point_ac = np.array([0, np.sin(self.a_rad), np.cos(self.a_rad)])
        self.point_ab = np.array([0, 0, 1])
        self.point_b = np.array([np.sin(self.b_rad), 0, np.cos(self.b_rad)])
        self.point_b_prime = np.array(
            [-np.sin(self.b_rad), 0, np.cos(self.b_rad)])
        self.phi = (1+np.sqrt(5))/2
        self.icosa_edge_length = 2/(np.sqrt(self.phi+2))
        self.icosa_height = (np.sqrt(3)/2)*self.icosa_edge_length
        self.dihedral_rad = np.pi - np.arccos(np.sqrt(5)/3)
        self.icosa_angle_a_from_xy = np.arccos(np.sqrt(5)/3)/2
        self.icosa_vertex_y = np.array(
            [0, np.cos(self.b_rad), np.sin(self.b_rad)])
        self.icosa_inradius = (3*np.sqrt(3)+np.sqrt(15))/12
        # edge length 1 icosahedron
        self.icosa_vertices = np.array([
            [1,  0,  self.phi],
            [-1,  0,  self.phi],
            [1,  0, -self.phi],
            [-1,  0, -self.phi],
            [0,  self.phi,  1],
            [0, -self.phi,  1],
            [0,  self.phi, -1],
            [0, -self.phi, -1],
            [self.phi,  1,  0],
            [-self.phi,  1,  0],
            [self.phi, -1,  0],
            [-self.phi, -1,  0]
        ])/np.sqrt(1+self.phi**2)

    def subdivide(self, method="radial"):
        self.points_lcd_list = []
        self.points_lcd_sphere = []
        if (method == "aa"):
            delta_a_rad = self.a_rad/self.frequency
            self.axis = "y"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_arc_ij = np.arctan(
                            np.sin(i*delta_a_rad)*np.tan(self.B_rad))
                        y_arc_ij = j*delta_a_rad
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "ab"):
            delta_a_rad = self.a_rad/self.frequency
            self.axis = "x"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_arc_ij = self.b_rad - \
                            np.arcsin(
                                np.tan(delta_a_rad*(self.frequency-i))/np.tan(self.A_rad))
                        y_arc_ij = j*delta_a_rad
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "aab"):
            delta_a_rad = self.a_rad/self.frequency
            self.axis = "y"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        y_arc_ij = j*delta_a_rad
                        x_arc_ij = np.arctan(np.cos(y_arc_ij)*np.sin(i*delta_a_rad)*np.tan(
                            self.B_rad)/np.sin(np.pi/2-self.a_rad+i*delta_a_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "ba"):
            delta_a_rad = self.a_rad/self.frequency
            delta_b_rad = self.b_rad/self.frequency
            self.axis = "y"
            # self.points_lcd_list.append([self.b_rad,0])
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_arc_ij = i*delta_b_rad
                        y_arc_ij = self.a_rad - \
                            np.arcsin(
                                np.tan(delta_b_rad*(self.frequency-j))/np.tan(self.B_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "bb"):
            delta_a_rad = self.a_rad/self.frequency
            delta_b_rad = self.b_rad/self.frequency
            self.axis = "x"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_arc_ij = i*delta_b_rad
                        y_arc_ij = np.arctan(
                            np.sin(j*delta_b_rad)*np.tan(self.A_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "bba"):
            delta_b_rad = self.b_rad/self.frequency
            self.axis = "y"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_arc_ij = i*delta_b_rad
                        y_arc_ij = np.arctan(np.cos(x_arc_ij)*np.sin(j*delta_b_rad)*np.tan(
                            self.A_rad)/np.sin(np.pi/2-self.b_rad+j*delta_b_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "ca"):
            delta_c_rad = self.c_rad/self.frequency
            self.axis = "y"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_arc_ij = np.arcsin(
                            np.sin(i*delta_c_rad)*np.sin(self.B_rad))
                        y_arc_ij = self.a_rad - \
                            np.arctan(
                                np.tan(delta_c_rad*(self.frequency-j))*np.cos(self.B_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "cb"):
            delta_c_rad = self.c_rad/self.frequency
            self.axis = "x"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        y_arc_ij = np.arcsin(
                            np.sin(j*delta_c_rad)*np.sin(self.A_rad))
                        x_arc_ij = self.b_rad - \
                            np.arctan(
                                np.tan(delta_c_rad*(self.frequency-i))*np.cos(self.A_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "cab"):
            delta_c_rad = self.c_rad/self.frequency
            self.axis = "y"
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        y_arc_ij = self.a_rad - \
                            np.arctan(
                                np.tan(delta_c_rad*(self.frequency-j))*np.cos(self.B_rad))
                        x_arc_ij = self.b_rad - \
                            np.arctan(
                                np.tan(delta_c_rad*(self.frequency-i))*np.cos(self.A_rad))
                        self.points_lcd_list.append((x_arc_ij, y_arc_ij))
        elif (method == "radial"):

            # we'll rotate the LCD such that it is on a face parallel with the xz plane in the up z direction to make the subdivision math easier

            height = np.sqrt(3)/2
            icosa_inradius = (3*np.sqrt(3)+np.sqrt(15))/12
            dihedral = np.arccos(np.sqrt(5)/3)
            rotation = np.pi-dihedral/2
            # Normalize a point to project it onto the unit sphere

            def normalize(v):
                return v / np.linalg.norm(v)

            z = icosa_inradius
            y_min = np.sqrt(3)/6
            delta_y = height/(2*self.frequency)
            x_max = 1/2
            delta_x = x_max/self.frequency
            # self.points_lcd_list = [[x_max, y_min, z]]
            for i in range(self.frequency+1):
                grid_x = self.frequency-i
                for j in range(self.frequency-i+1):
                    grid_y = j
                    if (self.geo_class == 1):
                        if ((grid_x % 2 == 0 and grid_y % 6 == 0) or (grid_x % 2 == 1 and (grid_y+3) % 6 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    elif (self.geo_class == 2):
                        if ((grid_x % 2 == 0 and grid_y % 2 == 0) or ((grid_x+1) % 2 == 0 and (grid_y+1) % 2 == 0)):
                            on_geodesic = True
                        else:
                            on_geodesic = False
                    if (on_geodesic):
                        x_icosa_ij = i*delta_x
                        y_icosa_ij = j*delta_y + y_min
                        self.points_lcd_list.append(
                            (x_icosa_ij, y_icosa_ij, z))
            points = [[p[0], p[1], p[2], 1] for p in self.points_lcd_list]
            self.points_lcd_list = points
            self.points_lcd_list.extend(np.array(
                (self.points_lcd_list)@reflection_matrix_3d([0, 0, 0], [1, 0, 0])).tolist())
            for i in range(2):
                self.points_lcd_list.extend((np.array(points)@rotation_matrix_3d_about_line(
                    np.array([0, 0, 0]), np.array([0, 0, 1]), 2*(i+1)*np.pi/3)).tolist())
            self.points_lcd_list.extend(np.array(
                (self.points_lcd_list)@reflection_matrix_3d([0, 0, 0], [1, 0, 0])).tolist())
            for point in unique_points(np.array(self.points_lcd_list), self.tolerance):
                self.points_lcd_sphere.append(
                    normalize([point[0], point[1], point[2]]))
            self.points_lcd_sphere = [[p[0], p[1], p[2], 1]
                                      for p in self.points_lcd_sphere]
            points_lcd_sphere = self.points_lcd_sphere

            self.points_lcd_sphere = ((np.array(points_lcd_sphere)@rotation_matrix_3d_about_line(
                np.array([0, 0, 0]), np.array([1, 0, 0]), rotation)).tolist())
            # elf.points_lcd_sphere =self.points_lcd_sphere.tolist()

        if (not method == "radial"):
            if (self.axis == "x"):
                for point in self.points_lcd_list:
                    x_sphere = np.sin(point[0])*np.cos(point[1])
                    y_sphere = np.sin(point[1])
                    z_sphere = np.cos(point[0])*np.cos(point[1])
                    self.points_lcd_sphere.append(
                        [x_sphere, y_sphere, z_sphere, 1])
            elif (self.axis == "y"):
                for point in self.points_lcd_list:
                    x_sphere = np.sin(point[0])
                    y_sphere = np.sin(point[1])*np.cos(point[0])
                    z_sphere = np.cos(point[0])*np.cos(point[1])
                    self.points_lcd_sphere.append(
                        [x_sphere, y_sphere, z_sphere, 1])


class geodesic():
    """
    A representation of a geodesic subdivision of a sphere (and dome fraction of the sphere). Geometry is generated on initialization
    useful instance variables:
    sphere_points #list of points for the geodesic vertices
    dome_points #list of points for the top fraction of the geodesic, shifted so that the cut line is at z=0
    sphere_edges #list of edges, e.g. pairs of points, comprising the geodesic sphere
    dome_edges #ditto, but for the fraction forming the dome
    sphere_edge_indices #list of pairs of indices of the sphere_points list for the edges of the sphere
    dome_edge_indices #list of pairs of indices of the dome_points list for the edges of the dome
    edge_lengths #list of lengths of edges of the dome. Useful when deciding which subdivision method to use
    """

    def __init__(self, geo_class, frequency, radius, method="radial", fraction=1, rotation=([0, 0, 0]), tolerance=0.0000001):
        """
        
        class options are 1 or 2
        frequency options are any natural number N for class 1, 2N for class 2
        method options include "aa", "ab", "aab", "ba", "bb", "bba", "ca", "cb", "cab", "radial"
        
        rotation is sequentially applied [x, y, z] in radians
        For Triad symmetry use rotation = [np.pi-np.arccos(np.sqrt(5)/3)/2,0,0]
        For Cross symmetry use rotation = [0,0,0]
        For Pentad symmetry use rotation = [np.arctan(2)/2-np.pi/2,0,0]   

        """
        if (frequency < 1 or geo_class not in [1, 2]):
            raise ValueError(
                'Bad arguments for geodesic class or frequency. geo_class = 1 or 2, frequency > 0')
        if (fraction < 0 or fraction > 1):
            raise ValueError(
                'fraction must be greater than zero and less than or equal to 1')
        # elif(geo_class == 2 and frequency%2):
        #    raise ValueError('Bad arguments for geodesic class or frequency. for geo_class = 2, frequency = 2*n')
        self.geo_class = geo_class
        self.frequency = int(frequency)
        self.radius = radius
        self.method = method
        self.fraction = fraction
        self.rotation = rotation
        #since we're using floating point math, a tolerance is needed
        #to determin whether to consider two points as identical
        #set to appx 0.1 to find the average vertices on radial projection
        self.tolerance = tolerance
        
        self.lcd = lcd(self.geo_class, self.frequency, self.tolerance)
        
        self.generate()
        self.get_edge_lengths()

    def generate(self):
        """
        Pattern the LCD around the sphere to form the full geodesic. Use rotations and reflections to ensure we have all the points
        - Radial projection is inexact so we'll use a large tolerance on the unique points function to get the averages of each projected vertex
        - Due to this averaging, high frequencies may fail to generate valid geodesic subdivisions using the radial method. FortunatelyOther methods
            are more appropriate for practical use.
        """
        self.lcd.subdivide(self.method)
        lcd_rotation1 = np.dot(np.array(self.lcd.points_lcd_sphere), rotation_matrix_3d_about_line(
            np.array([0, 0, 0]), np.array(self.lcd.point_ac), 2*np.pi/3).T)
        lcd_rotation2 = np.dot(np.array(self.lcd.points_lcd_sphere), rotation_matrix_3d_about_line(
            np.array([0, 0, 0]), np.array(self.lcd.point_ac), 4*np.pi/3).T)
        icosa_face = self.lcd.points_lcd_sphere + \
            lcd_rotation1.tolist() + lcd_rotation2.tolist()
        icosa_mirrored = np.dot(np.array(icosa_face), reflection_matrix_3d(
            np.array([0, 0, 0]), np.array([0, -1, 0])))
        self.icosa_mirrored = icosa_face+icosa_mirrored.tolist()
        segment = np.array(icosa_face + icosa_mirrored.tolist())
        segments = segment.tolist()
        for i in range(5):
            segments += (np.array(segment)@rotation_matrix_3d_about_line(
                [0, 0, 0], self.lcd.icosa_vertex_y, (i+1)*np.pi*2/5)).tolist()
        for vector in self.lcd.icosa_vertices:
            for i in range(5):
                segments += (np.array(segment)@rotation_matrix_3d_about_line(
                    [0, 0, 0], vector, (i+1)*np.pi*2/5)).tolist()
            segments += ((np.array(segment)@reflection_matrix_3d(
                [0, 0, 0], vector))@rotation_matrix_3d_about_line([0, 0, 0], vector, np.pi/5)).tolist()
        segments += (segments @
                     reflection_matrix_3d([0, 0, 0], [0, 0, 1])).tolist()

        # segments = segments + (np.array(segments)@rotation_matrix_3d_about_line([0,0,0],self.lcd.icosa_vertex_y,(i+1)*np.pi*2/5)).tolist()
        # segments = segments + ((np.array(segments)@reflection_matrix_3d([0,0,0],self.lcd.icosa_vertex_y))@rotation_matrix_3d_about_line([0,0,0],self.lcd.icosa_vertex_y, np.pi/5)).tolist()

        if (self.method == "radial"):
            sphere_points = unique_points(np.array(segments), self.tolerance)
        else:
            sphere_points = unique_points(np.array(segments), self.tolerance)
        sphere_points = (sphere_points@rotation_matrix_3d_about_line([0, 0, 0], [0, 0, 1], self.rotation[2]))@rotation_matrix_3d_about_line(
            [0, 0, 0], [0, 1, 0], self.rotation[1])@rotation_matrix_3d_about_line([0, 0, 0], [1, 0, 0], self.rotation[0])
        sphere_points = sphere_points@scaling_matrix(self.radius)
        sphere_points = sphere_points.tolist()
        truncate_depth = self.radius - 2*self.radius*self.fraction-self.tolerance
        translation_matrix = [[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0, 1, -truncate_depth],
                              [0, 0, 0, 1]]

        # for point in dome_points:
        # point.pop()
        for point in sphere_points:
            point.pop()
        self.sphere_edges, self.sphere_edge_indices = get_convex_mesh_edges_nearest_neighbors(
            np.array(sphere_points))
        self.sphere_points = sphere_points
        # self.dome_points = [point for point in sphere_points if point[2]>=truncate_depth]
        self.dome_edges = [list(edge) for edge in self.sphere_edges if edge[0]
                           [2] >= truncate_depth and edge[1][2] > truncate_depth]
        dome_edge_indices = np.array([list(index) for index in self.sphere_edge_indices if self.sphere_points[index[0]]
                                     [2] >= truncate_depth and self.sphere_points[index[1]][2] >= truncate_depth])
        unique_indices_dome = np.unique(dome_edge_indices).astype(int)
        new_indices_dome = np.array(range(len(unique_indices_dome)))
        sphere_points = [[p[0], p[1], p[2], 1] for p in self.sphere_points]
        dome_points = (np.array(sphere_points) @
                       translation_matrix)[unique_indices_dome]
        # dome_points = (np.array(sphere_points)@translation_matrix)
        dome_points = [[p[0], p[1], p[2]] for p in dome_points]
        self.dome_points = dome_points
        self.dome_points_sphere_indexed = list(
            zip(unique_indices_dome, dome_points))
        reindexed_dome_edge_indices = np.zeros_like(dome_edge_indices)
        for key, val in zip(unique_indices_dome, new_indices_dome):
            reindexed_dome_edge_indices[dome_edge_indices == key] = val
        # self.dome_edge_indices = [[index_mapping[i] for i in edge]for edge in dome_edge_indices]
        dome_edge_indices = np.array(reindexed_dome_edge_indices)
        dome_edge_indices.sort()
        dome_edge_indices = dome_edge_indices[dome_edge_indices[:, 1].argsort(
        )]
        dome_edge_indices = dome_edge_indices[dome_edge_indices[:, 0].argsort(
            kind='stable')]
        self.dome_edge_indices = dome_edge_indices
        self.dome_edges = [[self.dome_points[i[0]], self.dome_points[i[1]]]
                           for i in self.dome_edge_indices]

    def get_edge_lengths(self, tolerance=1e-9):
        """
        Calculates and returns a list of unique edge lengths from a list of edges.
    
        Args:
            edges: A list of tuples, where each tuple represents an edge and contains
                   two 3D points (e.g., [(p1, p2), (p3, p4), ...]).  Points should be
                   represented as lists or numpy arrays of length 3.
            tolerance: The tolerance for considering two lengths as equal.
    
        Returns:
            A list of unique edge lengths.
        """

        lengths = []
        for edge in self.dome_edges:
            p1 = np.array(edge[0])  # Convert to numpy arrays for calculations
            p2 = np.array(edge[1])
            length = np.linalg.norm(p2 - p1)
            lengths.append(length)

        unique_lengths = []
        for length in lengths:
            is_unique = True
            for existing_length in unique_lengths:
                if abs(length - existing_length) < tolerance:
                    is_unique = False
                    break
            if is_unique:
                unique_lengths.append(length)

        self.edge_lengths = unique_lengths
