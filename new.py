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

import functions
from functions import convert_cordinate, load_semantic_colors, load_all_objects_from_json

def get_true_scene_center_from_json(json_path):
    with open(json_path, "r") as f:
        scene_info = json.load(f)

    min_corner = np.array([np.inf, np.inf, np.inf])
    max_corner = np.array([-np.inf, -np.inf, -np.inf])

    for region in scene_info["regions"]:
        for obj in region["objects"]:
            if "center" not in obj or "size" not in obj:
                continue

            center = convert_cordinate(obj["center"])
            size = np.abs(convert_cordinate(obj["size"]))
            half = size / 2.0

            obj_min = center - half
            obj_max = center + half

            min_corner = np.minimum(min_corner, obj_min)
            max_corner = np.maximum(max_corner, obj_max)

    scene_center = (min_corner + max_corner) / 2.0
    return scene_center


def get_room_center(scene, region_id):
    for region in scene.regions:
        idx = int(region.id.split("_")[-1])
        if idx != region_id:
            continue
        centers = [convert_cordinate(obj.aabb.center) for obj in region.objects 
                   if not (np.any(np.isinf(obj.aabb.sizes)) or np.all(obj.aabb.sizes == 0))]
        if not centers:
            continue
        return np.mean(centers, axis=0)
    return None

def crop_room_mesh(scene, sem_mesh, semantic_color_map, region_id):
    bbox = functions.create_bbox_from_scene(scene, semantic_color_map, region_id, 'floor')
    bboxes = functions.create_bboxes_from_scene(scene, semantic_color_map, region_id)
    z_max_list = [b.get_max_bound()[2] for b in bboxes]

    min_bound = bbox[0].get_min_bound()
    max_bound = bbox[0].get_max_bound()
    new_max_bound = np.array([max_bound[0], max_bound[1], max(z_max_list)])
    roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, new_max_bound)
    roi_bbox.color = bbox[0].color

    cropped = sem_mesh.crop(roi_bbox)
    return cropped
def draw_with_custom_view(geometries, lookat=[0,0,0], eye=[5,5,5], up=[0,1,0]):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for g in geometries:
        vis.add_geometry(g)
    vis.poll_events()
    vis.update_renderer()

    ctr = vis.get_view_control()
    ctr.set_lookat(lookat)
    ctr.set_front((np.array(eye) - np.array(lookat)) / np.linalg.norm(np.array(eye) - np.array(lookat)))
    ctr.set_up(up)
    ctr.set_zoom(0.5)  # optional

    vis.run()
    vis.destroy_window()
def build_hierarchical_scene_graph(scene, json_path, semantic_color_map, region_ids):
    G = nx.DiGraph()

    scene_center = get_true_scene_center_from_json(json_path)
    G.add_node("scene_center", pos=scene_center, color=[1, 0, 0], level="scene")

    with open(json_path, 'r') as f:
        scene_info = json.load(f)

    for region in scene_info["regions"]:
        region_id = region["region_id"]
        if region_id not in region_ids:
            continue

        room_center = get_room_center(scene, region_id)
        room_node = f"region_{region_id}"
        G.add_node(room_node, pos=room_center, color=[0, 1, 0], level="room")
        G.add_edge("scene_center", room_node, edge_color=[1, 0, 0])  # 빨간 선

        for obj in region["objects"]:
            obj_name = f"{region_id}_{obj['semantic_id']}"
            center = convert_cordinate(obj["center"])
            color = semantic_color_map.get(obj["semantic_id"], [0.5, 0.5, 1.0])
            G.add_node(obj_name, pos=center, color=color, level="object")
            G.add_edge(room_node, obj_name, edge_color=[0, 0, 1])  # 파란 선

    return G

def visualize_graph_with_mesh(G, cropped_meshes):
    geometries = cropped_meshes[:]

    for node, data in G.nodes(data=True):
        pos = np.array(data["pos"])
        color = data["color"]
        level = data.get("level", "object")
        radius = 0.12 if level == "scene" else 0.1 if level == "room" else 0.08

        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(pos)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    points, lines, colors = [], [], []
    idx_map = {}

    for i, (node, data) in enumerate(G.nodes(data=True)):
        idx_map[node] = i
        points.append(data["pos"])

    for u, v, attr in G.edges(data=True):
        lines.append([idx_map[u], idx_map[v]])
        colors.append(attr.get("edge_color", [0.4, 0.4, 0.8]))

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    o3d.visualization.draw_geometries(geometries)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hierarchicalSceneGraph.py <scene_id>")
        sys.exit(1)

    scene_id = sys.argv[1]
    scene_name = scene_id.split("-")[1]

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
    scene = sim.semantic_scene              
    semantic_color_map = functions.load_semantic_colors(semanticTXT_path)
    sim.pathfinder.load_nav_mesh(navmesh_path)

    # Room 선택
    target_region_ids = [1, 4, 8]

    # Mesh crop for each room
    cropped_meshes = []
    for region_id in target_region_ids:
        cropped = crop_room_mesh(scene, sem_mesh, load_semantic_colors(semanticTXT_path), region_id)
        cropped_meshes.append(cropped)

    # Build graph
    semantic_color_map = load_semantic_colors(semanticTXT_path)
    G = build_hierarchical_scene_graph(scene, json_path, semantic_color_map, target_region_ids)
    
    # Visualize
    visualize_graph_with_mesh(G, cropped_meshes)
