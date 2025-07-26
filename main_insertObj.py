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

if len(sys.argv) != 4:
    print("\nmanual : python3 main_insertObj.py <scene_id> <region_id> <object_path>")
    print("example: python3 main_insertObj.py 00800-TEEsavR23oF 1 objaverse/SM_LampStand001.glb")
    sys.exit(1)

scene_id = sys.argv[1]
scene_name = scene_id.split("-")[1]
region_id = int(sys.argv[2])
object_path = sys.argv[3]

THR_RESAMPLE_CNT = 3

# make path automatically
base_path = Path(__file__).resolve().parent.parent
scene_cfg_path = str(base_path / "data/scene_dataset/hm3d/hm3d_annotated_basis.scene_dataset_config.json")
top_folder = functions.find_parent_folder(scene_cfg_path, scene_id)
scene_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.basis.glb")
semantic_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.semantic.glb")
semanticTXT_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.semantic.txt")
navmesh_path = str(base_path / f"data/scene_dataset/hm3d/{top_folder}/{scene_id}/{scene_name}.basis.navmesh")
json_path = str(base_path / f"data/format/{top_folder}/{scene_id}_scene_info.json")

# there should be *.object_config.json in same folder
obj_path = str(base_path / object_path)

# open3d objects
sem_mesh = o3d.io.read_triangle_mesh(semantic_path)
sem_mesh.compute_vertex_normals()

mesh = o3d.io.read_triangle_mesh(scene_path)
mesh.compute_vertex_normals()

axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])

# simulation setting
agent_roll_deg = -20
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

cfg = functions.make_custome_cfg(sim_settings)
sim = habitat_sim.Simulator(cfg)
semantic_scene = sim.semantic_scene              
semantic_color_map = functions.load_semantic_colors(semanticTXT_path)
sim.pathfinder.load_nav_mesh(navmesh_path)

## Crop based on region_id
o_bbox_floor = functions.create_bbox_from_scene(semantic_scene, semantic_color_map, region_id, 'floor')
# o_bboxes = functions.create_bboxes_from_scene(semantic_scene, semantic_color_map, region_id)
exclude = ['ceiling', 'floor', 'floor_mat']
o_bboxes = functions.create_bboxes_from_scene_exclude(semantic_scene, semantic_color_map, region_id, exclude_categories=exclude)

z_max_list = [bbox.get_max_bound()[2] for bbox in o_bboxes]
min_bound = o_bbox_floor[0].get_min_bound()  
max_bound = o_bbox_floor[0].get_max_bound()  
roi_bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound, np.array([max_bound[0], max_bound[1], max(z_max_list)]))
roi_bbox.color = o_bbox_floor[0].color  # RoI BBox for room

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

# Get 3D object
obj_template_mgr = sim.get_object_template_manager()
rigid_obj_mgr = sim.get_rigid_object_manager()
template_ids = obj_template_mgr.load_object_configs(obj_path)
sphere_template_id = template_ids[0]

h_bboxes = []
for o_bbox in o_bboxes:
    h_bbox_min = np.minimum(functions.convert_back_coordinate(o_bbox.get_min_bound()), functions.convert_back_coordinate(o_bbox.get_max_bound()))
    h_bbox_max = np.maximum(functions.convert_back_coordinate(o_bbox.get_min_bound()), functions.convert_back_coordinate(o_bbox.get_max_bound()))
    h_bboxes.append((h_bbox_min, h_bbox_max))

b_oclusion = True

for attempt in range(THR_RESAMPLE_CNT):

    o_navigable_point_obj = navigable_pcd_filtered[np.random.choice(len(navigable_pcd_filtered))]
    if np.array_equal(o_navigable_point_obj, o_navigable_point):
        continue

    h_navigable_point_obj = functions.convert_back_coordinate(o_navigable_point_obj)

    # object's coordinate -> Open3d based!!
    # # -------------------------------------------- if u want to chage scale --------------------------------------------
    # template = obj_template_mgr.get_template_by_id(sphere_template_id)
    # template.scale = np.array([2.0, 2.0, 2.0])
    # handle = template.handle
    # new_template_id = obj_template_mgr.register_template(template, specified_handle=handle, force_registration=True)
    # # ----------------------------------------------------------------------------------------------------------------

    object = rigid_obj_mgr.add_object_by_template_id(sphere_template_id)
    h_object_aabb = object.root_scene_node.cumulative_bb

    object.translation = mn.Vector3(
        h_navigable_point_obj[0],
        h_navigable_point_obj[1] + h_object_aabb.max[1],
        h_navigable_point_obj[2]
    )

    obj_roll_deg = 0
    obj_pitch_deg = 0
    obj_yaw_deg = 0
    object.rotation = functions.calculate_rotation(obj_roll_deg, obj_pitch_deg, obj_yaw_deg)

    # object info
    h_obj_bbox = (np.array(h_object_aabb.min) + h_navigable_point_obj,
                  np.array(h_object_aabb.max) + h_navigable_point_obj)
    h_obj_xz = functions.aabb_to_xz_box(h_obj_bbox)

    b_oclusion = False
    for h_bbox in h_bboxes:
        h_bbox_xz = functions.aabb_to_xz_box(h_bbox)
        iou = functions.compute_2d_iou(h_obj_xz, h_bbox_xz)
        if iou != 0.0:
            b_oclusion = True
            print(f"[Resample {attempt+1}] Occluded (IoU={iou:.4f})")

            # ---------------------- for debug ----------------------
            if attempt+1 != THR_RESAMPLE_CNT:
                rigid_obj_mgr.remove_object_by_id(object.object_id)
            #--------------------------------------------------------
            break

    if not b_oclusion:
        print(f"[Resample {attempt+1}] Not Occluded")
        break

if b_oclusion:
    print(f"[Warning] All {THR_RESAMPLE_CNT} attempts failed: 여전히 Occluded 상태입니다.")

# Set agent position
agent_position = h_navigable_point
look_at_target = h_navigable_point_obj  # agent가 바라볼 대상

# 방향 벡터 계산 (target - position)
forward = mn.Vector3(look_at_target) - mn.Vector3(agent_position)
forward = forward.normalized()

# 회전 행렬 계산
rotation_mat = mn.Matrix4.look_at(
    eye=mn.Vector3(agent_position),
    target=mn.Vector3(look_at_target),
    up=mn.Vector3(0.0, 1.0, 0.0)
).rotation()

# 회전 행렬 → 쿼터니언
agent_rot = mn.Quaternion.from_matrix(rotation_mat)
quat_np = np.array([
    agent_rot.vector.x,
    agent_rot.vector.y,
    agent_rot.vector.z,
    agent_rot.scalar
], dtype=np.float32)

# 에이전트 상태 적용
agent_state = sim.get_agent(0).get_state()
agent_state.position = agent_position
agent_state.rotation = quat_np
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

functions.draw_xz_boxes(h_obj_xz, [functions.aabb_to_xz_box(w) for w in h_bboxes])


# with open(json_path, 'r') as f:
#     region_info = json.load(f)

# room_objects = next((r["objects"] for r in region_info["regions"] if r["region_id"] == region_id), [])
# id_to_category = {obj["semantic_id"]: obj["category"] for obj in room_objects}

# unique_ids = np.unique(semantic)
# valid_ids = [sid for sid in unique_ids if sid in id_to_category]

# plt.figure(figsize=(10, 8))
# plt.imshow(rgb, cmap="tab20")
# plt.axis("off")
# plt.title(f"Semantic View for Region ID {region_id}")

# for sid in valid_ids:
#     mask = (semantic == sid)
#     y, x = np.argwhere(mask).mean(axis=0).astype(int)
#     category = id_to_category[sid]
#     plt.text(x, y, f"{sid}\n{category}",
#              fontsize=10,
#              color='white',
#              ha='center', va='center',
#              bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.3'))
#     plt.plot(x, y, 'ro', markersize=4)

# plt.tight_layout()
# plt.show()