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
import networkx as nx
from scipy.spatial.distance import euclidean
import json

# import functions
import functions

if len(sys.argv) != 2:
    print("\nmanual : python3 makeSceneGraph.py <scene_id>")
    print("example: python3 makeSceneGraph.py 00006-HkseAnWCgqk")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]

# make path automatically
base_path = Path(__file__).resolve().parent.parent
scene_cfg_path = str(base_path / "data/scene_dataset/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
top_folder = functions.find_parent_folder(scene_cfg_path, scene_id)
scene_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.semantic.txt")
navmesh_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.basis.navmesh")
json_path = str(base_path / f"data_origin/format/{top_folder}/{scene_id}_scene_info.json")

# open3d objects
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()

mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

semantic_color_map = functions.load_semantic_colors(semanticTXT_path)

with open(json_path, "r") as f:
    scene_info = json.load(f)

G = nx.DiGraph()
positions = {}

# calculate scene center
all_centers = [
    functions.convert_cordinate(obj['center'])
    for region in scene_info['regions']
    for obj in region['objects']
]
scene_center = np.mean(all_centers, axis=0)

scene_id = scene_info["scene_id"]
total_rooms_cnt = scene_info["total_rooms"]
total_objects_cnt = scene_info["total_objects"]

G.add_node("root_node",
           type="scene",
           scene_id=scene_id,
           total_rooms=total_rooms_cnt,
           total_objects=total_objects_cnt)

positions["root_node"] = scene_center
print(G.nodes["root_node"])

# calculate rooms and objects
for region in scene_info['regions']:
    region_id = region['region_id']
    room_name = f"room_{region_id}"

    room_objects = region['objects']
    room_centers = [functions.convert_cordinate(obj['center']) for obj in room_objects]
    room_center = np.mean(room_centers, axis=0)

    G.add_node(
        room_name,
        type="room",
        total_objects=len(room_objects),
        position=room_center.tolist()
    )
    positions[room_name] = room_center
    G.add_edge("root_node", room_name)
    print(G.nodes[room_name])

    for obj in room_objects:
        obj_name = f"{obj['category']}_{obj['semantic_id']}"
        obj_pos = functions.convert_cordinate(obj['center'])
        obj_size = obj['size']
        obj_volume = obj_size[0] * obj_size[1] * obj_size[2]
        obj_class = obj['category']

        G.add_node(
            obj_name,
            type="object",
            category=obj_class,
            position=obj_pos.tolist(),
            size=obj_size,
            volume=obj_volume,
            class_name=obj_class
        )
        positions[obj_name] = obj_pos
        G.add_edge(room_name, obj_name)

color_map = {
    'scene': [1.0, 0.0, 0.0],   
    'room': [0.0, 0.0, 1.0],   
    'object': [0.0, 1.0, 0.0],
}

spheres = []
for node, attr in G.nodes(data=True):
    pos = positions[node]
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
    sphere.translate(pos)

    # 색상 설정
    if node == "root_node":
        color = color_map['scene']
    elif attr["type"] == "room":
        color = color_map['room']
    elif attr["type"] == "object":
        try:
            semantic_id = int(node.split("_")[-1])
            color = semantic_color_map.get(semantic_id, [0.5, 0.5, 0.5]) 
        except:
            color = [0.5, 0.5, 0.5]
    else:
        color = [0.5, 0.5, 0.5]

    sphere.paint_uniform_color(color)
    spheres.append(sphere)


lines, colors, points = [], [], []
index_map = {}

for i, (node, pos) in enumerate(positions.items()):
    index_map[node] = i
    points.append(pos)

for u, v in G.edges():
    lines.append([index_map[u], index_map[v]])
    colors.append([0.3, 0.3, 0.3])

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines)
)
line_set.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries([sem_mesh] + spheres + [line_set])