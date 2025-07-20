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
import json
from scipy.spatial import cKDTree

import functions

'''
Place the agent at a navigable point within a specific room based on region_id, and render the scene.
Randomly sample a navigable point around the corner of the region specified by region_id.
Re-render the scene so that the agent faces the center of the room from the sampled point, and 
if the number of detected objects is too small, resample a new point.
'''

if len(sys.argv) != 3:
    print("\nmanual : python3 main.py <scene_id> <region_id> <use_default>")
    print("example: python3 main.py 00800-TEEsavR23oF 1")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]
region_id = int(sys.argv[2])

# ------------------------------------ Part 0 ------------------------------------
## Set Hyperparameter
THR_CORNER_NEAR_RANGE = 0.4
THR_CORNER_RADIUS = 0.5
THR_MAX_ATTEMPTS = 20
THR_MIN_OBJECTS_CNT = 5
EXCLUDE_CATEGORIES = ['wall', 'floor', 'stairs', 'unknown']

# ------------------------------------ Part 1 ------------------------------------

## Basic Setting
# make path automatically
base_path = Path(__file__).resolve().parent.parent
scene_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.semantic.txt")
navmesh_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.basis.navmesh")
scene_cfg_path = str(base_path / "data/scene_dataset/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json")
json_path = str(base_path / f"data/format/val/{scene_id}_scene_info.json")

# load mesh
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()
mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()
axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# set simulation
roll_deg = 0
pitch_deg = 0
yaw_deg = 0
rotation = np.array([np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)], dtype=np.float32)   # for sensor

sim_settings = {
    "width": 960,  
    "height": 480,
    "scene": scene_path,  
    "scene_cfg_path": scene_cfg_path,
    "navmesh_path": navmesh_path,
    "default_agent": 0,
    "sensor_height": 1.5,  
    "color_sensor": True, 
    "semantic_sensor": True,  
    "depth_sensor": True,  
    "rotation": rotation,
    "seed": 1,
}

cfg = functions.make_custome_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
semantic_scene = sim.semantic_scene              
semantic_color_map = functions.load_semantic_colors(semanticTXT_path)
sim.pathfinder.load_nav_mesh(navmesh_path)

## Crop based on region_id
bbox_floor = functions.create_bbox_from_scene(semantic_scene, semantic_color_map, region_id, 'floor')
bboxes = functions.create_bboxes_from_scene(semantic_scene, semantic_color_map, region_id)

z_max_list = [bbox.get_max_bound()[2] for bbox in bboxes]
min_bound = bbox_floor[0].get_min_bound()  
max_bound = bbox_floor[0].get_max_bound()  
roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, np.array([max_bound[0], max_bound[1], max(z_max_list)]))
roi_bbox.color = bbox_floor[0].color  # RoI BBox for room

# crop navigable points
points = functions.sample_navmesh_points(sim, num_points=5000)

pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)
pcd.paint_uniform_color([0.2, 0.8, 0.2])  

indices = roi_bbox.get_point_indices_within_bounding_box(pcd.points)
navigable_pcd_filtered = pcd.select_by_index(indices)

# select random point
navigable_pcd_filtered = np.asarray(navigable_pcd_filtered.points)
o_navigable_point = navigable_pcd_filtered[np.random.choice(len(navigable_pcd_filtered))]
h_navigable_point = functions.convert_back_coordinate(o_navigable_point)

# set agent position
agent_state = sim.get_agent(0).get_state()
agent_state.position = h_navigable_point
agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(0.0, np.array([0, 1.0, 0]))
sim.get_agent(0).set_state(agent_state)

## Rendering
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

# ------------------------------------ Part 2 ------------------------------------

## Re-rendering
# select corner points
o_navigable_points = pcd.select_by_index(indices)
cropped_sem_mesh = sem_mesh.crop(roi_bbox) 

xyz = np.asarray(o_navigable_points.points)
x = xyz[:, 0]
y = xyz[:, 1]
x_min, x_max = np.min(x), np.max(x)

# left area
left_mask = np.abs(x - x_min) < THR_CORNER_NEAR_RANGE
left_points = xyz[left_mask]
left_y = left_points[:, 1]

lb = left_points[np.argmin(left_y)]  
lt = left_points[np.argmax(left_y)]  

# right area
right_mask = np.abs(x - x_max) < THR_CORNER_NEAR_RANGE
right_points = xyz[right_mask]
right_y = right_points[:, 1]

rb = right_points[np.argmin(right_y)] 
rt = right_points[np.argmax(right_y)]  

# corner point visualization
corner_points = np.array([lt, rt, lb, rb])
o_corner_pcd = o3d.geometry.PointCloud()
o_corner_pcd.points = o3d.utility.Vector3dVector(corner_points)
o_corner_pcd.paint_uniform_color([1.0, 0.0, 0.0])  
# o3d.visualization.draw_geometries([cropped_sem_mesh, o_navigable_points, o_corner_pcd]) # o3d 기반 visualization 실행 시 Habitat-sim Re-rendering이 정확히 안됨.

# find points near-by corner points
o_navigable_points = pcd.select_by_index(indices)
navigable_xyz = np.asarray(o_navigable_points.points)
kdtree = cKDTree(navigable_xyz)

corner_points = np.asarray(o_corner_pcd.points)
floor_center = bbox_floor[0].get_center()
floor_center_habitat = functions.convert_back_coordinate(floor_center)

with open(json_path, 'r') as f:
    region_info = json.load(f)
room_objects = next((r["objects"] for r in region_info["regions"] if r["region_id"] == region_id), [])
id_to_category = {obj["semantic_id"]: obj["category"] for obj in room_objects}

nearby_samples = []
for corner in corner_points:
    idx = kdtree.query_ball_point(corner, r=THR_CORNER_RADIUS)
    if idx:
        candidates = navigable_xyz[idx]
        sampled = candidates[np.random.choice(len(candidates))]
        nearby_samples.append(sampled)

if not nearby_samples:
    raise RuntimeError("\033[91mThere are no valid navigable points within the radius around any corner.\033[0m")

# random sample navigable point
for attempt in range(1, THR_MAX_ATTEMPTS + 1):
    if not nearby_samples:
        print("\033[91mAll candidate positions have been used. Stopping.\033[0m")
        break

    final_sample = nearby_samples.pop(np.random.choice(len(nearby_samples)))    
    agent_position = functions.convert_back_coordinate(final_sample)

    # calculate rotation : agent_position-floor_center
    forward = mn.Vector3(floor_center_habitat) - mn.Vector3(agent_position)
    forward = forward.normalized()
    rotation_mat = mn.Matrix4.look_at(
        eye=mn.Vector3(agent_position),
        target=mn.Vector3(floor_center_habitat),
        up=mn.Vector3(0.0, 1.0, 0.0)
    ).rotation()
    agent_rot = mn.Quaternion.from_matrix(rotation_mat)
    quat_np = np.array([agent_rot.vector.x, agent_rot.vector.y, agent_rot.vector.z, agent_rot.scalar], dtype=np.float32)

    agent_state = sim.get_agent(0).get_state()
    agent_state.position = agent_position
    agent_state.rotation = quat_np
    sim.get_agent(0).set_state(agent_state)

    obs = sim.get_sensor_observations()
    rgb = obs.get("color_sensor", None)
    semantic = obs.get("semantic_sensor", None)

    # if the number of semantic objects is too small, resample the agent position.
    cnt_semantic_obj = 0
    for sid in np.unique(semantic):
        if sid not in id_to_category:
            continue
        category = id_to_category[sid].lower()
        if category in EXCLUDE_CATEGORIES:
            continue
        if np.count_nonzero(semantic == sid) > 0:
            cnt_semantic_obj += 1

    if cnt_semantic_obj >= THR_MIN_OBJECTS_CNT:
        print("a position with sufficient semantic objects has been found, ", agent_state.position)
        break
    else:
        print("Not enough semantic objects. Resampling")

if cnt_semantic_obj < THR_MIN_OBJECTS_CNT:
    print("Maximum number of attempts reached. Using the last sampled position.")
print(f"cnt semantic objects: {cnt_semantic_obj}")

# visualization
plt.figure(figsize=(10, 8))
plt.imshow(rgb)
plt.axis("off")
plt.title(f"Semantic View for Region ID {region_id}")

for sid in np.unique(semantic):
    if sid not in id_to_category:
        continue
    category = id_to_category[sid].lower()
    if category in EXCLUDE_CATEGORIES:
        continue
    mask = (semantic == sid)
    if np.count_nonzero(mask) == 0:
        continue
    y, x = np.argwhere(mask).mean(axis=0).astype(int)
    plt.text(x, y, f"{sid}\n{category}",
             fontsize=10,
             color='white',
             ha='center', va='center',
             bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
    plt.plot(x, y, 'ro', markersize=4)

plt.tight_layout()
plt.show()