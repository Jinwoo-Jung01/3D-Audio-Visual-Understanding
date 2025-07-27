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

# import functions
import functions

if len(sys.argv) != 2:
    print("\nmanual : python3 viewer.py <scene_id>")
    print("example: python3 viewer.py 00800-TEEsavR23oF")
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
json_path = str(base_path / f"data/format/{top_folder}/{scene_id}_scene_info.json")

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
    "width": 256,  # Spatial resolution of the observations
    "height": 256,
    "scene": scene_path,  # Scene path
    "scene_cfg_path": scene_cfg_path,
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

# Visualization
o3d.visualization.draw_geometries([sem_mesh, axes])

# print("\n\033[91m============== Debugging ==============\033[0m")
# print("\n\033[91m============== Finish ==============\033[0m")