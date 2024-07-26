import numpy as np
from stl import mesh


def drawSphere(center, radius, resolution=20):
    """ Creates a sphere mesh around a center point with a given radius. """
    phi = np.linspace(0, np.pi, resolution)
    theta = np.linspace(0, 2 * np.pi, resolution)
    phi, theta = np.meshgrid(phi, theta)

    x = center[0] + radius * np.sin(phi) * np.cos(theta)
    y = center[1] + radius * np.sin(phi) * np.sin(theta)
    z = center[2] + radius * np.cos(phi)

    vertices = np.vstack((x.flatten(), y.flatten(), z.flatten())).T
    faces = []

    for i in range(resolution - 1):
        for j in range(resolution - 1):
            faces.append([i * resolution + j,
                          i * resolution + (j + 1),
                          (i + 1) * resolution + j])
            faces.append([(i + 1) * resolution + j,
                          i * resolution + (j + 1),
                          (i + 1) * resolution + (j + 1)])

    faces = np.array(faces)
    sphere_mesh = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            sphere_mesh.vectors[i][j] = vertices[f[j], :]

    return sphere_mesh

def drawConnection(start,end,thikness):
    vertices = [start,end,np.array(start) - [0,thikness,thikness]]
    faces = np.array([[0,1,2]])

    vertices, faces = generate_cylinder(start, end, thikness)

    allMesh = mesh.Mesh(np.zeros(np.array(faces).shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            allMesh.vectors[i][j] = vertices[f[j]]

    return allMesh


import numpy as np
from stl import mesh

import numpy as np
from stl import mesh

def drawHollowCube(corner, side, thickness=0.2):
    # Function to create a cube with a given corner, side length, and thickness
    def create_cube(corner, side):
        return np.array([
            corner,
            corner + [side, 0, 0],
            corner + [side, side, 0],
            corner + [0, side, 0],
            corner + [0, 0, side],
            corner + [side, 0, side],
            corner + [side, side, side],
            corner + [0, side, side]
        ])
    
    outer_vertices = create_cube(corner, side)
    inner_vertices = create_cube(corner + thickness, side - 2 * thickness)
    
    faces = [
        # Outer cube faces
        [0, 1, 2], [0, 2, 3],  # Bottom
        [4, 5, 6], [4, 6, 7],  # Top
        [0, 1, 5], [0, 5, 4],  # Front
        [2, 3, 7], [2, 7, 6],  # Back
        [0, 3, 7], [0, 7, 4],  # Left
        [1, 2, 6], [1, 6, 5],  # Right

        # Inner cube faces
        [8, 9, 10], [8, 10, 11],  # Bottom
        [12, 13, 14], [12, 14, 15],  # Top
        [8, 9, 13], [8, 13, 12],  # Front
        [10, 11, 15], [10, 15, 14],  # Back
        [8, 11, 15], [8, 15, 12],  # Left
        [9, 10, 14], [9, 14, 13],  # Right

        # Connecting faces
        [0, 1, 9], [0, 9, 8],  # Bottom Front
        [1, 2, 10], [1, 10, 9],  # Bottom Right
        [2, 3, 11], [2, 11, 10],  # Bottom Back
        [3, 0, 8], [3, 8, 11],  # Bottom Left
        
        [4, 5, 13], [4, 13, 12],  # Top Front
        [5, 6, 14], [5, 14, 13],  # Top Right
        [6, 7, 15], [6, 15, 14],  # Top Back
        [7, 4, 12], [7, 12, 15]  # Top Left
    ]
    
    vertices = np.vstack([outer_vertices, inner_vertices])
    
    all_mesh = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            all_mesh.vectors[i][j] = vertices[f[j],:]
    
    return all_mesh






import numpy as np

def generate_cylinder(A, B, D, N=10):
    # Convert A and B to numpy arrays
    A = np.array(A)
    B = np.array(B)
    
    # Calculate the axis vector and radius
    v = B - A
    R = D / 2.0
    
    # Normalize the axis vector
    v_norm = np.linalg.norm(v)
    v = v / v_norm
    
    # Create a vector that is not parallel to v
    if v[0] != 0 or v[1] != 0:
        u = np.array([-v[1], v[0], 0])
    else:
        u = np.array([0, -v[2], v[1]])
    
    # Normalize u
    u = u / np.linalg.norm(u)
    
    # Create another perpendicular vector w using cross product
    w = np.cross(v, u)
    
    # Points on the circles
    theta = np.linspace(0, 2 * np.pi, N, endpoint=False)
    circle_A = np.array([A + R * (np.cos(t) * u + np.sin(t) * w) for t in theta])
    circle_B = np.array([B + R * (np.cos(t) * u + np.sin(t) * w) for t in theta])
    
    # Vertices array
    vertices = np.vstack((circle_A, circle_B))
    
    # Faces array
    faces = []
    for i in range(N):
        next_i = (i + 1) % N
        # Side face (two triangles)
        faces.append([N + i, i,  N + next_i])
        faces.append([N + next_i, i,  next_i])
    
    # Top and bottom faces
    for i in range(1, N - 1):
        faces.append([ i, 0, i + 1])  # Top face
        faces.append([N + i,N,  N + i + 1])  # Bottom face
    
    return vertices, faces

