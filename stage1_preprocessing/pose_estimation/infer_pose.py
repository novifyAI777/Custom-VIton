import os
import json
import torch
import cv2
import numpy as np
import sys

# Add openpose_repo to Python path
current_dir = os.path.dirname(__file__)
openpose_path = os.path.join(current_dir, "openpose_repo")
if openpose_path not in sys.path:
    sys.path.append(openpose_path)

from src.body import Body

# ------------------------------
# Convert OpenPose output to your JSON format
# ------------------------------
def save_pose_json(body_kps, json_path):
    keypoints = []
    for i in range(body_kps.shape[0]):
        x, y, conf = body_kps[i]
        keypoints.append({
            "id": i,
            "x": float(x),
            "y": float(y),
            "conf": float(conf)
        })
    with open(json_path, "w") as f:
        json.dump({"keypoints": keypoints}, f, indent=4)


# ------------------------------
# MAIN FUNCTION
# ------------------------------
def run(input_image, json_out):
    img = cv2.imread(input_image)
    if img is None:
        raise RuntimeError(f"Image not found: {input_image}")

    model_path = os.path.join(os.path.dirname(__file__), "openpose_repo", "model", "body_pose_model.pth")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        download_url = "https://www.dropbox.com/s/5v654d2u65fuvyr/body_pose_model.pth"
        error_msg = f"""
Model file not found: {model_path}

Please download the OpenPose body model:
1. Download from: {download_url}
2. Place it at: {model_path}
3. Or run: wget {download_url} -O {model_path}

Make sure the openpose_repo/model directory exists and contains the model file.
        """.strip()
        raise RuntimeError(error_msg)

    print("[OpenPose] Loading body pose model:", model_path)
    body_estimator = Body(model_path)

    # Run OpenPose
    candidate, subset = body_estimator(img)
    if subset.size == 0:
        raise RuntimeError("OpenPose could not detect any person!")

    print(f"[DEBUG] Detected {len(candidate)} keypoints, subset shape: {subset.shape}")
    
    # Convert keypoints
    person = subset[0]
    kps = []
    
    # Determine the number of keypoints based on the model output
    num_keypoints = len(person) - 2  # subtract score and person_id
    print(f"[DEBUG] Processing {num_keypoints} keypoints")
    
    for idx in range(num_keypoints):
        c = int(person[idx])
        if c < 0 or c >= len(candidate):
            kps.append([0, 0, 0])  # missing keypoint
        else:
            kps.append(candidate[c][:3])  # x,y,confidence

    kps = np.array(kps)
    print(f"[DEBUG] Final keypoints shape: {kps.shape}")

    # Save JSON in YOUR format
    save_pose_json(kps, json_out)
    print("✔ Saved:", json_out)

    return json_out


if __name__ == "__main__":
    run("data/stage1_output/person_clean.png", "data/stage1_output/pose_keypoints.json")
