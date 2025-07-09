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
    sensor_specs.append(color_sensor_spec)

    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    depth_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    sensor_specs.append(depth_sensor_spec)

    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, settings["sensor_height"], 0.0]
    semantic_sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
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