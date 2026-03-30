import trimesh
import numpy as np

# Load the mesh (GLB is typically Y-up)
# input_path = "../generated/for_slides.glb"
# output_path = "../generated/for_slides_zup.glb"

input_path = "../generated/mast3r.glb"
output_path = "../generated/mast3r_zup.glb"


mesh = trimesh.load(input_path)

# Rotate -90 degrees around X-axis to convert Y-up to Z-up
# Using -90 to ensure the mesh isn't upside down (standard for GLB to Blender)
transform = trimesh.transformations.rotation_matrix(np.radians(90), [1, 0, 0])
mesh.apply_transform(transform)

# Export the Z-up mesh
mesh.export(output_path)
print(f"Saved Z-up mesh to {output_path}")