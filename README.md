# 2025 AMI Lab Summer URP

We constructed 3D scenes by loading the [HM3DSem dataset](https://github.com/matterport/habitat-matterport-3dresearch) into the [SoundSpaces 2.0](https://github.com/facebookresearch/sound-spaces) simulator environment. To increase the diversity and complexity of the scenes, we additionally inserted 3D objects from the [Objaverse](https://github.com/allenai/objaverse-xl) dataset. Spatial audio was then generated based on the updated environments, allowing us to build a richer and more comprehensive dataset.

## Code

We provide a set of functionalities for constructing 3D scenes in the simulator, including inserting objects and generating hierarchical 3D scene graphs.

For accurate code execution, we recommend setting up the environment with the following structure. Below is an example configuration for a single 3D scene:

    AMILab/
    ├── habitat-lab/
    ├── habitat-sim/
    ├── sound-spaces/
    ├── 3D-Audio-Visual-Understanding/
    │      └── main.py
    ├── objaverse/
    │      └── boat.json
    └── data/
            ├── format/
            ├── scene_datasets/
            │   └── hm3d/
            │		├── val/
            │		│	└── 00800-TEEsavR23oF/
            │       │        ├── TEEsavR23oF.basis.glb    
            │       │        ├── TEEsavR23oF.basis.navmesh 
            │       │        ├── TEEsavR23oF.semantic.glb 
            │       │        └── TEEsavR23oF.semantic.txt 
            │	    └── hm3d_annotated_basis.scene_dataset_config.json 
            └── mp3d_material_config.json

**You must move the objaverse and data folders out of the 3D-Audio-Visual-Understanding repository and place them at the same directory level.**
For details, please refer to the directory structure shown above.

---
#### main.py
Place the agent at a navigable point within a specific room based on region_id, and render the scene.
Randomly sample a navigable point around the corner of the region specified by region_id.
Re-render the scene so that the agent faces the center of the room from the sampled point, and 
if the number of detected objects is too small, resample a new point.

    python3 main_insertObj.py <scene_id> <region_id>
    python3 main.py 00800-TEEsavR23oF 1

Before running main.py, you must generate a JSON file for each 3D scene.
You can use the script below to generate JSON files for all 3D scenes within a specific folder such as train, val, or test.

    python3 makeAllSceneJSON.py <scene_parent_folder>
    python3 makeAllSceneJSON.py data/scene_dataset/hm3d/val

---
#### main_insertObj.py
We insert Objaverse objects based on navigable points within the environment. To ensure realistic placement, we calculate the 2D IoU between the inserted object and existing objects in the 3D scene to check for occlusion. If occlusion is detected, we perform resampling. The agent's position is also randomly sampled from navigable points, and its orientation is adjusted to face the inserted object.

    python3 main_insertObj.py <scene_id> <region_id>
    python3 main_insertObj.py 00800-TEEsavR23oF 1

Before inserting objects, Objaverse data must be downloaded, and a corresponding *.object_config.json file should be generated using the following code. The input argument should be the file path to the .glb object you wish to insert.

    python3 makeObjectConfig.py path/to/object.glb
    python3 makeObjectConfig.py ./objaverse/boat.glb

---
#### viewer.py
You can visualize the HM3D 3D scenes in the Open3D environment.
The visualization will include semantic annotations applied to the scene.

    python3 viewer.py <scene_id>
    python3 viewer.py 00800-TEEsavR23oF

---
#### analysisNavmesh.py
You can visualize the navigable points of a specific room within the 3D scene.

    python3 analysisNavmesh.py <scene_id> <region_id>
    python3 analysisNavmesh.py 00800-TEEsavR23oF 1

---
#### makeSceneGraph.py
You can make Hierarchical 3D scene graph.

    python3 makeSceneGraph.py <scene_id>
    python3 makeSceneGraph.py 00006-HkseAnWCgqk