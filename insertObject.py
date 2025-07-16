# ignore unnecessary warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="llvmlite.binding.ffi")

# you need to import magnum before importing habitat-sim or it will cause a core-dump
import magnum as mn
import numpy as np
from magnum.platform.glfw import Application

import habitat_sim
import habitat_sim.utils.common as utils
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import sys
from pathlib import Path
import json

# import functions
import functions

"""
Research about .navmesh
"""

if len(sys.argv) != 3:
    print("\nmanual : python3 insertObject.py <scene_id> <region_id>")
    print("example: python3 insertObject.py 00800-TEEsavR23oF 1")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]
region_id = int(sys.argv[2])

# make path automatically
base_path = Path(__file__).resolve().parent
base_path = Path("/home/jinwoo/AMILab")
scene_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.semantic.txt")
navmesh_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.basis.navmesh")
scene_cfg_path = str(base_path / "data/scene_dataset/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json")
json_path = str(base_path / f"data/format/val/{scene_id}_scene_info.json")
# there should be *.object_config.json in same folder
obj_path = "/home/jinwoo/AMILab/3D-Audio-Visual-Understanding/objaverse/SM_LampStand001.glb"

# open3d objects
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()

mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# simulation setting
agent_roll_deg = 0
agent_pitch_deg = 0
agent_yaw_deg = 0
rotation = np.array([np.radians(agent_roll_deg), np.radians(agent_pitch_deg), np.radians(agent_yaw_deg)], dtype=np.float32) 

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
    "rotation": rotation,
    "seed": 1,
}

# set simulation
cfg = functions.make_custome_cfg(sim_settings)
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

# print("\n\033[91m============== Debugging ==============\033[0m")

# Get 3D object
obj_template_mgr = sim.get_object_template_manager()
rigid_obj_mgr = sim.get_rigid_object_manager()
template_ids = obj_template_mgr.load_object_configs(obj_path)
sphere_template_id = template_ids[0]
object = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)    # load 3d object by template_ids(from *.object_config.json)

# Localize 3D object
obj_position = np.array([-7.448947, 0.16337794, -3.682925])
object.translation = obj_position

obj_roll_deg = 0
obj_pitch_deg = 0
obj_yaw_deg = 0
object.rotation = functions.calculate_rotation(obj_roll_deg, obj_pitch_deg, obj_yaw_deg)

# Set agent position
agent_state = sim.get_agent(0).get_state()
agent_position = np.array([-7.448947, 0.16337794, -0.682925])
agent_state.position = agent_position
sim.get_agent(0).set_state(agent_state)

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
    if title == "Semantic":
        ax.imshow(img, cmap="tab20")
    else:
        ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

with open(json_path, 'r') as f:
    region_info = json.load(f)

room_objects = next((r["objects"] for r in region_info["regions"] if r["region_id"] == region_id), [])
id_to_category = {obj["semantic_id"]: obj["category"] for obj in room_objects}

unique_ids = np.unique(semantic)
valid_ids = [sid for sid in unique_ids if sid in id_to_category]

plt.figure(figsize=(10, 8))
plt.imshow(rgb, cmap="tab20")
plt.axis("off")
plt.title(f"Semantic View for Region ID {region_id}")

for sid in valid_ids:
    mask = (semantic == sid)
    y, x = np.argwhere(mask).mean(axis=0).astype(int)
    category = id_to_category[sid]
    plt.text(x, y, f"{sid}\n{category}",
             fontsize=10,
             color='white',
             ha='center', va='center',
             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    plt.plot(x, y, 'ro', markersize=4)

plt.tight_layout()
plt.show()

# print("\n\033[91m============== Finish ==============\033[0m")