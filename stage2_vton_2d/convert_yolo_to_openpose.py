import json 



def convert_yolo_to_openpose(yolo_json, out_json):
    """Convert YOLO pose.json → CP-VTON++ OpenPose format"""
    with open(yolo_json, "r") as f:
        data = json.load(f)

    kps = data["keypoints"]

    pose = []
    for kp in kps:
        pose += [float(kp["x"]), float(kp["y"]), float(kp.get("conf", 1.0))]

    # CP-VTON++ requires:  pose_keypoints  (NOT pose_keypoints_2d)
    openpose = {
        "version": 1.0,
        "people": [{"pose_keypoints": pose}]
    }

    with open(out_json, "w") as f:
        json.dump(openpose, f, indent=2)

    print(f"✔ YOLO → OpenPose saved at: {out_json}")
    return openpose
