import json
from pathlib import Path
import sys

def main(glb_path_str):
    glb_path = Path(glb_path_str)
    config_path = glb_path.with_suffix(".object_config.json")

    config = {
        "render_asset": str(glb_path.name),
        "scale": [1.0, 1.0, 1.0]
    }

    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)

    print(f"[✔] Config generated: {config_path}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("\nmanual : python3 makeObjectConfig.py path/to/object.glb")
        sys.exit(1)

    main(sys.argv[1])