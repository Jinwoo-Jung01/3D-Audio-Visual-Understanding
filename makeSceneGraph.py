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

# import functions
import functions

if len(sys.argv) != 3:
    print("\nmanual : python3 makeSceneGraph.py <scene_id> <region_id>")
    print("example: python3 makeSceneGraph.py 00006-HkseAnWCgqk 1")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]
region_id = int(sys.argv[2])

# make path automatically
base_path = Path(__file__).resolve().parent.parent
scene_path = str(base_path / f"data/scene_dataset/hm3d/train/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_dataset/hm3d/train/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_dataset/hm3d/train/{scene_id}/{scene_name}.semantic.txt")
scene_cfg_path = str(base_path / "data/scene_dataset/hm3d/train/hm3d_annotated_train_basis.scene_dataset_config.json")
json_path = str(base_path / f"data/format/train/{scene_id}_scene_info.json")

# open3d objects
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()

mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

semantic_color_map = functions.load_semantic_colors(semanticTXT_path)

# print("\n\033[91m============== Debugging ==============\033[0m")
objects = functions.load_objects_from_json(json_path, region_id)

# 3D Scene Graph
G = nx.Graph()
for obj in objects:
    G.add_node(
        obj["name"],
        pos=obj["center"],
        class_name=obj["class"],
        semantic_id=obj["semantic_id"],
        color=semantic_color_map.get(obj["semantic_id"], [1.0, 0.6, 0.2])
    )

for i in range(len(objects)):
    for j in range(i + 1, len(objects)):
        p1, p2 = objects[i]["center"], objects[j]["center"]
        dist = euclidean(p1, p2)
        if dist < 1.8:
            G.add_edge(objects[i]["name"], objects[j]["name"], distance=dist)

# For visualization
geometries = []

for node, data in G.nodes(data=True):
    pos = np.array(data['pos'])
    color = data['color']

    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.08)
    sphere.translate(pos)
    sphere.paint_uniform_color(color)
    sphere.compute_vertex_normals()
    geometries.append(sphere)
    print(f"[INFO] Added {data['class_name']} at {pos}")

lines = []
colors = []
points = []

for u, v in G.edges():
    u_pos = G.nodes[u]['pos']
    v_pos = G.nodes[v]['pos']
    points.extend([u_pos, v_pos])
    lines.append([len(points) - 2, len(points) - 1])
    colors.append([0.2, 0.2, 1.0])  

line_set = o3d.geometry.LineSet(
    points=o3d.utility.Vector3dVector(points),
    lines=o3d.utility.Vector2iVector(lines),
)
line_set.colors = o3d.utility.Vector3dVector(colors)
geometries.append(line_set)

# Visualization
o3d.visualization.draw_geometries(geometries)
sub_geometries = functions.visualize_subgraph(G, target_node="1_39")
functions.visualize_3D_scene(json_path, sem_mesh, semantic_color_map, region_id, geometries)

# print("\n\033[91m============== Finish ==============\033[0m")