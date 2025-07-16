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
import time

# import functions
import functions

if len(sys.argv) != 2:
    print("\nUsage: python3 makeAllSceneJSON.py <scene_parent_folder>")
    print("Example: python3 makeAllSceneJSON.py data/scene_datasets/hm3d/val")
    sys.exit(1)

scene_parent_dir = Path(sys.argv[1]).resolve()
scene_cfg_path = "/home/jinwoo/AMILab/data/scene_dataset/hm3d/val/hm3d_annotated_val_basis.scene_dataset_config.json"

# Get all valid folders
scene_folders = [d for d in sorted(scene_parent_dir.iterdir()) if d.is_dir()]

print(f"[INFO] Found {len(scene_folders)} scene folders under {scene_parent_dir}")

scene_times = []

for scene_dir in scene_folders:
    scene_id = scene_dir.name  # ex: 00800-TEEsavR23oF
    if "-" not in scene_id:
        print(f"[WARNING] Skipping folder {scene_id} (not a valid scene ID format)")
        continue

    scene_name = scene_id.split("-")[1]

    scene_path = str(scene_dir / f"{scene_name}.basis.glb")
    semantic_path = str(scene_dir / f"{scene_name}.semantic.glb")
    semanticTXT_path = str(scene_dir / f"{scene_name}.semantic.txt")

    if not Path(scene_path).exists() or not Path(semantic_path).exists() or not Path(semanticTXT_path).exists():
        # print(f"[WARNING] Skipping {scene_id} — required files missing.")
        continue

    print(f"[INFO] Processing scene: {scene_id}")
    start_time = time.time()

    # Load meshes
    sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
    sem_mesh.compute_vertex_normals()

    mesh = o3d.io.read_triangle_mesh(scene_path)
    mesh.compute_vertex_normals()

    # Simulation config
    rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    sim_settings = {
        "width": 256,
        "height": 256,
        "scene": scene_path,
        "scene_cfg_path": scene_cfg_path,
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
    scene = sim.semantic_scene
    semantic_color_map = functions.load_semantic_colors(semanticTXT_path)

    # Save to JSON
    output_path = Path(__file__).resolve().parent / "format/val" / f"{scene_id}_scene_info.json"
    functions.make_3D_scene_json(scene, scene_id, scene_name, str(output_path))

    sim.close()
    elapsed = time.time() - start_time
    scene_times.append(elapsed)

    print(f"[✅] Saved: {output_path}")

if scene_times:
    avg_time = sum(scene_times) / len(scene_times)
    print(f"\nProcessed {len(scene_times)} scenes.")
    print(f"Average time per scene: {avg_time:.2f} seconds")
else:
    print("No scenes were processed.")