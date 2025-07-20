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

# import functions
import functions

if len(sys.argv) != 4:
    print("\nmanual : python3 calculateDistAngle.py <scene_id> <region_id> <use_default>")
    print("example: python3 calculateDistAngle.py 00800-TEEsavR23oF 1 True")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]
region_id = int(sys.argv[2])
if sys.argv[3] not in ["True", "False"]:
    print("Error: use_default must be True or False")
    sys.exit(1)
use_default = sys.argv[3]

# make path automatically
base_path = Path(__file__).resolve().parent.parent
scene_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.semantic.txt")
navmesh_path = str(base_path / f"data/scene_dataset/hm3d/val/{scene_id}/{scene_name}.basis.navmesh")
scene_cfg_path = str(base_path / "data/scene_dataset/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json")
json_path = str(base_path / f"data/format/val/{scene_id}_scene_info.json")

# open3d objects
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()

mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# simulation setting
roll_deg = 0
pitch_deg = 0
yaw_deg = 0
rotation = np.array([np.radians(roll_deg), np.radians(pitch_deg), np.radians(yaw_deg)], dtype=np.float32) 

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
agent_state = sim.get_agent(0).get_state()

if use_default == "False":
    agent_state.position = sampled_habitat_point
    agent_state.rotation = habitat_sim.utils.common.quat_from_angle_axis(0.0, np.array([0, 1.0, 0]))
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
    ax.imshow(img)
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()

with open(json_path, 'r') as f:
    region_info = json.load(f)

room_objects = next((r["objects"] for r in region_info["regions"] if r["region_id"] == region_id), [])
id_to_center = {obj["semantic_id"]: np.array(obj["center"]) for obj in room_objects}
id_to_category = {obj["semantic_id"]: obj["category"] for obj in room_objects}

unique_ids = np.unique(semantic)
valid_ids = [sid for sid in unique_ids if sid in id_to_center]

agent_pos = np.array([agent_state.position[0], agent_state.position[2]])  

object_infos = []
for sid in valid_ids:
    center = id_to_center[sid]
    center_xz = np.array([center[0], center[2]])

    dist = np.linalg.norm(center_xz - agent_pos)
    heading_diff = functions.calculate_heading(agent_pos, center_xz)

    object_infos.append({
        "id": sid,
        "category": id_to_category[sid],
        "center": center,
        "distance": dist,
        "heading_diff": heading_diff,
    })

nearest_objects = sorted(object_infos, key=lambda x: x["distance"])[:5]

for obj in nearest_objects:
    print(f" - ID {obj['id']} | {obj['category']}")
    print(f"   ↳ center: {obj['center']}")
    print(f"   ↳ distance: {obj['distance']:.2f} m")
    print(f"   ↳ heading diff: {obj['heading_diff']:.2f}°")

plt.figure(figsize=(10, 8))
plt.imshow(rgb, cmap="tab20")
plt.axis("off")
plt.title("Semantic View with ID and Category")

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

# print("\n\033[91m============== Debugging ==============\033[0m")

# import networkx as nx
# import matplotlib.pyplot as plt

# # 그래프 초기화
# G = nx.DiGraph()

# # Agent 노드 추가 (id: 'agent')
# G.add_node("agent", pos=tuple(agent_pos))

# # 객체 노드 및 edge 추가
# for obj in nearest_objects:
#     obj_id = obj['id']
#     center = obj['center']
#     center_xz = (float(center[0]), float(center[2]))

#     G.add_node(obj_id, pos=center_xz, label=obj['category'])
#     G.add_edge("agent", obj_id, distance=obj['distance'], heading=obj['heading_diff'])

# # 위치 추출
# pos = nx.get_node_attributes(G, 'pos')
# labels = nx.get_node_attributes(G, 'label')

# # 시각화
# plt.figure(figsize=(8, 6))
# nx.draw(G, pos, with_labels=True, node_size=800, node_color='skyblue', font_size=10, font_weight='bold')
# nx.draw_networkx_nodes(G, pos, nodelist=["agent"], node_color='red', node_size=900)

# # 방향 및 거리 정보 표시
# edge_labels = {
#     (u, v): f"{G[u][v]['distance']:.2f}m\n{G[u][v]['heading']:.1f}°"
#     for u, v in G.edges()
# }
# nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=9)

# plt.title("Top-5 Nearest Objects in XZ Plane")
# plt.grid(True)
# plt.axis("equal")
# plt.tight_layout()
# plt.show()


# print("\n\033[91m============== Finish ==============\033[0m")