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
import networkx as nx
from matplotlib.patches import Rectangle

def make_custome_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_cfg_path"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.orientation = settings["rotation"]
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_sensor_spec.orientation = settings["rotation"]
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    semantic_sensor_spec.orientation = settings["rotation"]
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.0)
            # "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def print_scene_instances(scene, limit_output=10):
    # Print semantic annotation information (id, category, bounding box details)
    # about levels, regions and objects in a hierarchical fashion
    
    print(f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects")
    print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")

    cnt = 0

    for region in scene.regions:
        for obj in region.objects:
            print(
                f"Object id:{obj.id}, category:{obj.category.name()},"
                f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
            )
            cnt += 1
            if cnt > limit_output:
                break

def convert_cordinate(arr):
    # adapt semantic coordinate system to open3d 
    return np.array([arr[0], -arr[2], arr[1]])

def convert_back_coordinate(arr):
    # Open3D → Habitat
    return np.array([arr[0], arr[2], -arr[1]])

def create_bbox(center_raw, size_raw, color):
    # make bbox of instance
    center = convert_cordinate(center_raw)
    size = np.abs(convert_cordinate(size_raw))
    half_size = size / 2
    box = o3d.geometry.AxisAlignedBoundingBox(center - half_size, center + half_size)
    box.color = color
    return box

def load_semantic_colors(txt_path):
    # instance matching using semantic.txt and id
    # get color information for each instance
    id_to_color = {}
    with open(txt_path, "r") as f:
        for line in f:
            if line.startswith("HM3D") or not line.strip():
                continue
            parts = line.strip().split(",")
            sem_id = int(parts[0])
            hex_color = parts[1]
            # convert to RGB(Regularization)
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
            id_to_color[sem_id] = [r, g, b]
    return id_to_color

def create_bboxes_from_scene(scene, semantic_color_map, region_id, print_info = False):
    """
    Create Open3D BBoxes for all objects in a specified region of the SemanticScene

    Args:
        scene: habitat_sim.semantic_scene object
        semantic_color_map: dict {semantic_id: [r, g, b]}
        region_id: int, the target region index

    Returns:
        bboxes: list of Open3D AxisAlignedBoundingBox
    """
    bboxes = []

    found = False
    for region in scene.regions:
        region_index = int(region.id.split("_")[-1])
        if region_index != region_id:
            continue

        found = True
        for obj in region.objects:
            if np.any(np.isinf(obj.aabb.sizes)) or np.all(obj.aabb.sizes == 0):
                continue
            color = semantic_color_map.get(obj.semantic_id, [1, 0, 0])
            bbox = create_bbox(obj.aabb.center, obj.aabb.sizes, color=color)
            if print_info:
                print(f"Object id: {obj.semantic_id}, category: {obj.category.name()}, "
                    f"center: {obj.aabb.center}, dims: {obj.aabb.sizes}")
            bboxes.append(bbox)

        break

    if not found:
        print(f"[ERROR] Region with index {region_id} not found in the semantic scene")
        exit(1)

    return bboxes

def create_bbox_from_scene(scene, semantic_color_map, region_id, target_categories=None, print_info=False):
    """
    Create Open3D BBoxes for objects of specified categories in a region of the SemanticScene

    Args:
        scene: habitat_sim.semantic_scene object
        semantic_color_map: dict {semantic_id: [r, g, b]}
        region_id: int, the target region index
        target_categories: list of str, object categories to include (e.g., ['chair', 'table'])
        print_info: bool, whether to print debug info

    Returns:
        bboxes: list of Open3D AxisAlignedBoundingBox
    """
    bboxes = []
    found = False

    for region in scene.regions:
        region_index = int(region.id.split("_")[-1])
        if region_index != region_id:
            continue

        found = True
        for obj in region.objects:
            if np.any(np.isinf(obj.aabb.sizes)) or np.all(obj.aabb.sizes == 0):
                continue

            category = obj.category.name()
            if target_categories and category not in target_categories:
                continue  # 카테고리가 리스트에 없으면 건너뛰기

            color = semantic_color_map.get(obj.semantic_id, [1, 0, 0])
            bbox = create_bbox(obj.aabb.center, obj.aabb.sizes, color=color)
            if print_info:
                print(f"[{category}] id: {obj.semantic_id}, center: {obj.aabb.center}, dims: {obj.aabb.sizes}")
            bboxes.append(bbox)

        break

    if not found:
        print(f"[ERROR] Region with index {region_id} not found in the semantic scene")
        exit(1)

    return bboxes

def extract_single_region_info(scene, region_id, save_path):
    # extract room's information per each room
    for region in scene.regions:
        region_index = int(region.id.split("_")[-1])
        if region_index != region_id:
            continue

        objects_info = []
        for obj in region.objects:
            if np.any(np.isinf(obj.aabb.sizes)) or np.all(obj.aabb.sizes == 0):
                continue  # skip invalid objects

            objects_info.append({
                "semantic_id": obj.semantic_id,
                "category": obj.category.name(),
                "center": obj.aabb.center.tolist(),
                "size": obj.aabb.sizes.tolist()
            })

        region_data = {
            "region_id": region_id,
            "object_count": len(objects_info),
            "objects": objects_info
        }

        # save
        with open(save_path, "w") as f:
            json.dump(region_data, f, indent=2)
        print(f"[INFO] Saved region info to: {save_path}")
        return

    print(f"[ERROR] Region with id {region_id} not found.")

def sample_navmesh_points(sim, num_points):
    # get navigable position from .navmesh
    points = []
    for _ in range(num_points):
        p = sim.pathfinder.get_random_navigable_point()
        point = convert_cordinate(p)
        points.append(point)
    return np.array(points)

def make_custome_cfg_rotation(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.scene_dataset_config_file = settings["scene_cfg_path"]

    # Note: all sensors must have the same resolution
    sensor_specs = []

    color_sensor_spec = habitat_sim.CameraSensorSpec()
    color_sensor_spec.uuid = "color_sensor"
    color_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    color_sensor_spec.resolution = [settings["height"], settings["width"]]
    color_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    color_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    color_sensor_spec.orientation = settings["roatation"]
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_sensor_spec.orientation = settings["roatation"]
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    semantic_sensor_spec.orientation = settings["roatation"]
    sensor_specs.append(semantic_sensor_spec)

    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "noop": habitat_sim.agent.ActionSpec(
            "noop", habitat_sim.agent.ActuationSpec(amount=0.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])

def make_3D_scene_json(scene, scene_id, scene_name, save_path):
    regions_data = []
    total_objects = 0

    for region in scene.regions:
        region_index = int(region.id.split("_")[-1])

        # igonore region_id -1
        if region_index == -1:
            continue
        
        objects_info = []

        for obj in region.objects:
            if np.any(np.isinf(obj.aabb.sizes)) or np.all(obj.aabb.sizes == 0):
                continue

            objects_info.append({
                "semantic_id": obj.semantic_id,
                "category": obj.category.name(),
                "center": obj.aabb.center.tolist(),
                "size": obj.aabb.sizes.tolist()
            })

        total_objects += len(objects_info)

        regions_data.append({
            "region_id": region_index,
            "object_count": len(objects_info),
            "objects": objects_info
        })

    full_info = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "total_rooms": len(regions_data),
        "total_objects": total_objects,
        "regions": regions_data
    }

    with open(save_path, "w") as f:
        json.dump(full_info, f, indent=2)

    print(f"[INFO] Saved full scene info to: {save_path}")

def visualize_3D_scene(json_path, sem_mesh, semantic_color_map, target_region_id=1, geometries=None):
    # load json
    with open(json_path, "r") as f:
        scene_info = json.load(f)
    print("1")
    # make 3D scene by json
    for region in scene_info["regions"]:
        region_id = region["region_id"]

        if target_region_id is not None and region_id != target_region_id:
            continue

        print(f"[INFO] Visualizing region_id: {region_id}")

        bbox = create_bbox_from_json(region, semantic_color_map, 'floor')
        bboxes = create_bboxes_from_json(region, semantic_color_map)

        if not bboxes or not bbox:
            print(f"[WARNING] region_id {region_id} has no valid bbox.")
            continue

        z_max_list = [b.get_max_bound()[2] for b in bboxes]
        min_bound = bbox[0].get_min_bound()
        max_bound = bbox[0].get_max_bound()
        new_max_bound = np.array([max_bound[0], max_bound[1], max(z_max_list)])

        roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, new_max_bound)
        roi_bbox.color = bbox[0].color

        cropped = sem_mesh.crop(roi_bbox)

        print(f"[INFO] Cropped region {region_id} — Object count: {region['object_count']}")
        if geometries != None:
            o3d.visualization.draw_geometries([cropped] + bboxes + geometries)
        else:
            o3d.visualization.draw_geometries([cropped] + bboxes)
            

def create_bboxes_from_json(region_data, semantic_color_map, target_categories=None, print_info=False):
    """
    Create Open3D BBoxes for objects listed in a region (from JSON file)

    Args:
        region_data: dict, from JSON — one item of scene_info["regions"]
        semantic_color_map: dict {semantic_id: [r, g, b]} or just use default color
        target_categories: list of str, filter for object categories (optional)
        print_info: bool

    Returns:
        bboxes: list of Open3D AxisAlignedBoundingBox
    """
    bboxes = []

    for obj in region_data["objects"]:
        category = obj["category"]
        if target_categories and category not in target_categories:
            continue

        center = np.array(obj["center"])
        size = np.array(obj["size"])
        color = semantic_color_map.get(int(obj["semantic_id"]), [1, 0, 0])
        bbox = create_bbox(center, size, color=color)
        if print_info:
            print(f"[{category}] id: {obj['semantic_id']}, center: {center}, size: {size}")
        bboxes.append(bbox)

    return bboxes

def create_bbox_from_json(region, semantic_color_map, target_categories=None, print_info=False):
    """
    Create Open3D BBoxes for objects of specified categories in a region from JSON data

    Args:
        region: dict, region data from JSON (scene_info["regions"][i])
        semantic_color_map: dict {semantic_id: [r, g, b]}
        target_categories: list of str, object categories to include (e.g., ['chair', 'table'])
        print_info: bool, whether to print debug info

    Returns:
        bboxes: list of Open3D AxisAlignedBoundingBox
    """
    bboxes = []

    for obj in region["objects"]:
        if "size" not in obj or "center" not in obj:
            continue

        size = np.array(obj["size"])
        if np.any(np.isinf(size)) or np.all(size == 0):
            continue

        category = obj["category"]
        if target_categories and category not in target_categories:
            continue

        color = semantic_color_map.get(obj["semantic_id"], [1, 0, 0])
        bbox = create_bbox(np.array(obj["center"]), size, color=color)

        if print_info:
            print(f"[{category}] id: {obj['semantic_id']}, center: {obj['center']}, dims: {obj['size']}")
        bboxes.append(bbox)

    return bboxes

def calculate_heading(agent_pos_xz, object_pos_xz):
    """
    agent_pos_xz: np.array([x, z])
    object_pos_xz: np.array([x, z])
    """
    vec = object_pos_xz - agent_pos_xz 
    # Agent 기준 -z방향을 바라고 Camear가 생성됨
    # Habitat-Sim 기준 z방향
    forward_2d = np.array([0, -1])      

    unit_v = vec / np.linalg.norm(vec)
    dot = np.clip(np.dot(forward_2d, unit_v), -1.0, 1.0)
    angle_rad = np.arccos(dot)

    cross = forward_2d[0] * unit_v[1] - forward_2d[1] * unit_v[0]
    if cross < 0:
        angle_rad = -angle_rad

    return np.degrees(angle_rad)

def calculate_rotation(obj_roll_deg, obj_pitch_deg, obj_yaw_deg):
    q_x = mn.Quaternion.rotation(mn.Deg(obj_roll_deg), mn.Vector3.x_axis())
    q_y = mn.Quaternion.rotation(mn.Deg(obj_yaw_deg), mn.Vector3.y_axis())
    q_z = mn.Quaternion.rotation(mn.Deg(obj_pitch_deg), mn.Vector3.z_axis())
    
    return q_y * q_z * q_x

def load_objects_from_json(json_path, target_region_id):
    with open(json_path, 'r') as f:
        scene_info = json.load(f)

    objects = []
    for region in scene_info["regions"]:
        if region["region_id"] != target_region_id:
            continue

        for obj in region["objects"]:
            objects.append({
                "name": f"{region['region_id']}_{obj['semantic_id']}",
                "center": convert_cordinate(obj["center"]),
                "class": obj["category"],
                "semantic_id": obj["semantic_id"]
            })

    return objects

def load_all_objects_from_json(json_path):
    with open(json_path, 'r') as f:
        scene_info = json.load(f)

    objects = []
    for region in scene_info["regions"]:

        for obj in region["objects"]:
            objects.append({
                "name": f"{region['region_id']}_{obj['semantic_id']}",
                "center": convert_cordinate(obj["center"]),
                "class": obj["category"],
                "semantic_id": obj["semantic_id"]
            })

    return objects

def visualize_subgraph(G, target_node):

    if target_node not in G:
        print(f"[ERROR] Node {target_node} not found in the graph.")
        return

    neighbors = list(G.neighbors(target_node))
    sub_nodes = [target_node] + neighbors

    # 노드들만 추출
    geometries = []
    for node in sub_nodes:
        pos = np.array(G.nodes[node]['pos'])
        color = G.nodes[node]['color']
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1 if node == target_node else 0.08)
        sphere.translate(pos)
        sphere.paint_uniform_color([1, 0, 0] if node == target_node else color)  # 중심 노드는 빨간색
        sphere.compute_vertex_normals()
        geometries.append(sphere)

    # 엣지들만 추출
    lines = []
    colors = []
    points = []
    point_idx_map = {}

    for i, node in enumerate(sub_nodes):
        point_idx_map[node] = i
        points.append(G.nodes[node]['pos'])

    for neighbor in neighbors:
        lines.append([point_idx_map[target_node], point_idx_map[neighbor]])
        colors.append([1, 0, 0])  # 파란색 선

    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    geometries.append(line_set)

    print(f"[INFO] Visualizing {target_node} and its {len(neighbors)} neighbors.")

    return geometries

def compute_2d_iou(box1, box2):
    """
    Compute 2D IoU in XZ plane between two AABBs.
    Each box is given as (min_x, min_z, max_x, max_z)
    """
    # Intersection
    x_left = max(box1[0], box2[0])
    z_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    z_bottom = min(box1[3], box2[3])

    if x_right <= x_left or z_bottom <= z_top:
        return 0.0  # No overlap

    inter_area = (x_right - x_left) * (z_bottom - z_top)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    return inter_area / union_area

def aabb_to_xz_box(aabb):
    if hasattr(aabb, "get_min_bound"):
        min_b = np.array(aabb.get_min_bound())
        max_b = np.array(aabb.get_max_bound())
    else:
        min_b, max_b = aabb  # assume tuple
        min_b = np.array(min_b)
        max_b = np.array(max_b)

    return (min_b[0], min_b[2], max_b[0], max_b[2])

def draw_xz_boxes(object_box, wall_boxes):
    fig, ax = plt.subplots()

    # Draw object box (red)
    ox1, oz1, ox2, oz2 = object_box
    ax.add_patch(Rectangle((ox1, oz1), ox2 - ox1, oz2 - oz1,
                           linewidth=2, edgecolor='red', facecolor='none', label='Object'))

    # Draw wall boxes (blue)
    for i, wall in enumerate(wall_boxes):
        wx1, wz1, wx2, wz2 = wall
        ax.add_patch(Rectangle((wx1, wz1), wx2 - wx1, wz2 - wz1,
                               linewidth=1.5, edgecolor='blue', facecolor='none'))
        ax.text((wx1 + wx2) / 2, (wz1 + wz2) / 2, f'Obj {i}', fontsize=8, color='blue', ha='center')

    # Center axes lines at (0, 0)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.axvline(0, color='gray', linestyle='--', linewidth=1)

    # Set axis limits to include negative ranges
    all_x = [ox1, ox2] + [wx1 for wx1, _, wx2, _ in wall_boxes] + [wx2 for _, _, wx2, _ in wall_boxes]
    all_z = [oz1, oz2] + [wz1 for _, wz1, _, wz2 in wall_boxes] + [wz2 for _, _, _, wz2 in wall_boxes]
    margin = 0.5

    ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
    ax.set_ylim(min(all_z) - margin, max(all_z) + margin)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("XZ Plane Bounding Boxes (Including Negative Coordinates)")
    ax.set_aspect('equal')
    ax.legend()
    plt.grid(True)
    plt.show()

def create_bboxes_from_scene_exclude(scene, semantic_color_map, region_id, print_info=False, exclude_categories=None):
    """
    Create Open3D BBoxes for all objects in a specified region of the SemanticScene

    Args:
        scene: habitat_sim.semantic_scene object
        semantic_color_map: dict {semantic_id: [r, g, b]}
        region_id: int, the target region index
        print_info: bool, whether to print object info
        exclude_categories: list of category names to exclude (e.g., ['ceiling', 'floor'])

    Returns:
        bboxes: list of Open3D AxisAlignedBoundingBox
    """
    bboxes = []
    found = False
    for region in scene.regions:
        region_index = int(region.id.split("_")[-1])
        if region_index != region_id:
            continue

        found = True
        for obj in region.objects:
            if np.any(np.isinf(obj.aabb.sizes)) or np.all(obj.aabb.sizes == 0):
                continue

            category_name = obj.category.name()
            if exclude_categories and category_name in exclude_categories:
                continue

            color = semantic_color_map.get(obj.semantic_id, [1, 0, 0])
            bbox = create_bbox(obj.aabb.center, obj.aabb.sizes, color=color)
            if print_info:
                print(f"Object id: {obj.semantic_id}, category: {category_name}, "
                      f"center: {obj.aabb.center}, dims: {obj.aabb.sizes}")
            bboxes.append(bbox)
        break

    if not found:
        print(f"[ERROR] Region with index {region_id} not found in the semantic scene")
        exit(1)

    return bboxes
