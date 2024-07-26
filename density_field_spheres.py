import numpy as np
from stl import mesh
import geometryUtils


# Dimensions of the 3D array
dim_x, dim_y, dim_z = 5, 5, 5

# Generate random density values
densities = np.random.randint(3, 11, (dim_x, dim_y, dim_z))

# Calculate the scaling factor to allow the highest density sphere to touch others
max_density = np.max(densities)
scaling_factor = 0.5 / max_density  # Ensure the maximum radius is 0.5

# Scale the densities so the highest density sphere touches its neighbors
scaled_densities = densities * scaling_factor

# Diameter of the cylinders
cylinder_diameter = scaled_densities.max() / 2

print("scaled_densities.max()  "+str(scaled_densities.max() ))
print("cylinder_diameter "+str(cylinder_diameter))

# Initialize an empty mesh
combined_mesh = mesh.Mesh(np.zeros(0, dtype=mesh.Mesh.dtype))
#xAxis=geometryUtils.drawConnection([0,0,0],[1,0,0],0.1)
#combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, xAxis.data]))

#zAxis=geometryUtils.drawConnection([0,0,0],[0,0,1],0.1)
#combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, zAxis.data]))

# Create spheres and cylinders for each point in the 3D array
mid_x=(dim_x-1)/2
mid_y=(dim_y-1)/2
mid_z=(dim_z-1)/2
scale=((dim_x-1))
for x in range(dim_x):
    for y in range(dim_y):
        for z in range(dim_z):
            center = np.array([x, y, z])
            radius = 0.7- ( 
                abs(x-mid_x)/scale
                +abs(y-mid_y)/scale
                +abs(z-mid_z)/scale
            )/3
            # radius = scaled_densities[x, y, z]


            if (x + y + z) % 2 == 0:
                continue
                #sphere = geometryUtils.drawSphere(center, 0.1)#sradius)
            else:
                sphere = geometryUtils.drawSphere(center, radius)


                for i in [-1,0,1]:
                    for j in [-1,0,1]:
                        for k in [-1,0,1]:
                            if not ((x+i) + (y+j) + (z+k) )%2 == 0  :
                                neighbor_center = center + np.array([i, j, k])
                                if 0 <= int(neighbor_center[0]) < dim_x and 0 <= int(neighbor_center[1]) < dim_y and 0 <= int(neighbor_center[2]) < dim_z:
                                    if neighbor_center[0]!=center[0] and neighbor_center[1]!=center[1] :
                                        #print("-------" + str(neighbor_center) + "--" + str(center) )
                                        print("--")
                                    else:
                                        if neighbor_center[2]<center[2]:
                                            print("-------" + str(neighbor_center) + "--" + str(center) )
                                            cylinder = geometryUtils.drawConnection(center, neighbor_center, cylinder_diameter/1.2)
                                            combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, cylinder.data]))

            combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, sphere.data]))

# Save the combined mesh to an STL file
combined_mesh.save('out/newSphres.stl')
