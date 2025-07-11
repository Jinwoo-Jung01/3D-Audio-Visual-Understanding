import os
import json

def analyze_scene_statistics(json_folder):
    json_files = [f for f in os.listdir(json_folder) if f.endswith(".json")]
    print(f"[INFO] Found {len(json_files)} JSON files\n")

    total_rooms = 0
    total_objects = 0

    room_counts_per_scene = []
    object_counts_per_scene = []
    object_counts_per_room = []

    for fname in json_files:
        path = os.path.join(json_folder, fname)

        with open(path, 'r') as f:
            scene_data = json.load(f)

        rooms = scene_data.get("total_rooms", 0)
        objects = scene_data.get("total_objects", 0)

        total_rooms += rooms
        total_objects += objects

        room_counts_per_scene.append(rooms)
        object_counts_per_scene.append(objects)

        for region in scene_data.get("regions", []):
            object_counts_per_room.append(region.get("object_count", 0))

    def summarize(lst):
        if not lst:
            return (0, 0, 0)
        avg = sum(lst) / len(lst)
        return (avg, min(lst), max(lst))

    avg_obj_per_room, min_obj_per_room, max_obj_per_room = summarize(object_counts_per_room)
    avg_rooms_per_scene, min_rooms_per_scene, max_rooms_per_scene = summarize(room_counts_per_scene)
    avg_objs_per_scene, min_objs_per_scene, max_objs_per_scene = summarize(object_counts_per_scene)

    print(f"[1] 전체 총 방 개수: {total_rooms}")
    print(f"[2] 전체 총 객체 수: {total_objects}")

    print("\n[3] 방 1개당 객체 수 통계 (모든 region 기준)")
    print(f" - 평균: {avg_obj_per_room:.2f}")
    print(f" - 최소: {min_obj_per_room}")
    print(f" - 최대: {max_obj_per_room}")

    print("\n[4] 3D Scene 하나당 방 개수 통계")
    print(f" - 평균: {avg_rooms_per_scene:.2f}")
    print(f" - 최소: {min_rooms_per_scene}")
    print(f" - 최대: {max_rooms_per_scene}")

    print("\n[5] 3D Scene 하나당 객체 수 통계")
    print(f" - 평균: {avg_objs_per_scene:.2f}")
    print(f" - 최소: {min_objs_per_scene}")
    print(f" - 최대: {max_objs_per_scene}")

analyze_scene_statistics("/home/jinwoo/AMILab/data/format/val")