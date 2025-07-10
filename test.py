# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llvmlite.binding.ffi")

# you need to import magnum before importing habitat-sim or it will cause a core-dump
import magnum as mn
import numpy as np
from magnum.platform.glfw import Application

import habitat_sim
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sys
from pathlib import Path
from scipy.spatial.transform import Rotation as R

# import functions
import functions

"""
Research about .navmesh
"""

if len(sys.argv) != 3:
    print("\nmanual : python3 simulator.py <scene_id> <region_id>")
    print("example: python3 simulator.py 00800-TEEsavR23oF 1")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]
region_id = int(sys.argv[2])

# make path automatically
base_path = Path(__file__).resolve().parent
scene_path = str(base_path / f"data/scene_datasets/hm3d/val/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_datasets/hm3d/val/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_datasets/hm3d/val/{scene_id}/{scene_name}.semantic.txt")
navmesh_path = str(base_path / f"data/scene_datasets/hm3d/val/{scene_id}/{scene_name}.basis.navmesh")
scene_cfg_path = str(base_path / "data/scene_datasets/hm3d/hm3d_annotated_minival_basis.scene_dataset_config.json")

# open3d objects
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()

mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
print("\n\033[91m============== Debugging ==============\033[0m")

# simulation setting
pitch_deg = -15
pitch_rad = np.radians(pitch_deg)
rotation_vector = np.array([pitch_rad, 0, 0], dtype=np.float32)  

sim_settings = {
    "width": 960,  # Spatial resolution of the observations
    "height": 480,
    "scene": scene_path,  # Scene path
    "scene_cfg_path": scene_cfg_path,
    "navmesh_path": navmesh_path,
    "default_agent": 0,
    "sensor_height": 1.5,  # Height of sensors in meters
    "color_sensor": True,  # RGB sensor
    "semantic_sensor": True,  # Semantic sensor
    "depth_sensor": True,  # Depth sensor
    "quat_xyzw": rotation_vector,
    "seed": 1,
}

# set simulation
cfg = functions.make_custome_cfg_rotation(sim_settings)
sim = habitat_sim.Simulator(cfg)
scene = sim.semantic_scene              # semantic scnene
semantic_color_map = functions.load_semantic_colors(semanticTXT_path)
sim.pathfinder.load_nav_mesh(navmesh_path)


## 1. Filtering based on specific room

# Make RoI Bbox
bbox = functions.create_bbox_from_scene(scene, semantic_color_map, region_id, 'floor')
bboxes = functions.create_bboxes_from_scene(scene, semantic_color_map, region_id)

z_max_list = [bbox.get_max_bound()[2] for bbox in bboxes]

min_bound = bbox[0].get_min_bound()  
max_bound = bbox[0].get_max_bound()  
new_max_bound = np.array([max_bound[0], max_bound[1], max(z_max_list)])
roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, new_max_bound)
roi_bbox.color = bbox[0].color  

# Make Cropped Semantic Mesh
sem_cropped = sem_mesh.crop(roi_bbox) 

# Make Filtered Navigable PCD
points = functions.sample_navmesh_points(sim, num_points=5000)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.2, 0.8, 0.2])  

indices = roi_bbox.get_point_indices_within_bounding_box(pcd.points)
navigable_pcd_filtered = pcd.select_by_index(indices)

## 2. Render and display RGB Image

# Set agent's position based on navigable point
navigable_pcd_filtered = np.asarray(navigable_pcd_filtered.points)
navigable_position = navigable_pcd_filtered[np.random.choice(len(navigable_pcd_filtered))]
sampled_habitat_point = functions.convert_back_coordinate(navigable_position)

agent_state = sim.get_agent(0).get_state()
agent_state.position = [-9.448947, 0.16337794, -1.182925]
agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(0.0, np.array([0, 1.0, 0]))
sim.get_agent(0).set_state(agent_state)

agent = sim.get_agent(0)
state = agent.get_state()

print("Agent Position:", state.position)
print("Agent Rotation (quaternion):", state.rotation)

# Get Camera's Image
observations = sim.get_sensor_observations()
rgb = observations.get("color_sensor", None)
depth = observations.get("depth_sensor", None)
semantic = observations.get("semantic_sensor", None)

images = []
titles = []

if rgb is not None:
    images.append(rgb)
    titles.append("RGB")
if depth is not None:
    images.append(depth)
    titles.append("Depth")
if semantic is not None:
    images.append(semantic)
    titles.append("Semantic")

fig, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 6))
if len(images) == 1:
    axes = [axes]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

print("\n\033[91m============== Finish ==============\033[0m")