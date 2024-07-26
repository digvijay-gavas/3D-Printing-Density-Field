import numpy as np
from stl import mesh
import geometryUtils
import math



dim_x, dim_y, dim_z = 7, 7, 7
densities = np.random.randint(3, 11, (dim_x, dim_y, dim_z))

max_density = np.max(densities)
scaling_factor = 0.5 / max_density

scaled_densities = densities * scaling_factor
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
            #radius = 0.5- ( 
            #    abs(x-mid_x)/scale
            #    +abs(y-mid_y)/scale
            #    +abs(z-mid_z)/scale
            #)/3
#
            #radius=radius if radius>0 else 0

            radius=0
            centerDist=math.sqrt((x-mid_x)*(x-mid_x) + (y-mid_y)*(y-mid_y) + (z-mid_z)*(z-mid_z))
            #if centerDist<3:
            R=2.7
            maxRadius=0.7
            if centerDist == R:
                radius=maxRadius
            else:
                radius=maxRadius*(1/(1+abs(math.exp(3*(centerDist-R)))))
            # radius = scaled_densities[x, y, z]


            
            sphere = geometryUtils.drawSphere(center, radius)
            combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, sphere.data]))

            if(z<dim_z-1):
                neighbor_center = center + np.array([0, 0, 1])
                cylinder = geometryUtils.drawConnection(center, neighbor_center, cylinder_diameter/1.2)
                combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, cylinder.data]))
        
            if(y<dim_y-1):
                neighbor_center = center + np.array([0, 1, 0])
                cylinder = geometryUtils.drawConnection(center, neighbor_center, cylinder_diameter/1.2)
                combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, cylinder.data]))
            
            if(x<dim_x-1):
                neighbor_center = center + np.array([1, 0, 0])
                cylinder = geometryUtils.drawConnection(center, neighbor_center, cylinder_diameter/1.2)
                combined_mesh = mesh.Mesh(np.concatenate([combined_mesh.data, cylinder.data]))


            
            

# Save the combined mesh to an STL file
combined_mesh.save('out/newSphres.stl')
