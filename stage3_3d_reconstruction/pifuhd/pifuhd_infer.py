import os
import shutil
import json
import subprocess

# ------------------------------------------------------------
# Convert YOLOv8 JSON → OpenPose JSON format (required by PIFuHD)
# ------------------------------------------------------------
def convert_yolo_to_pifuhd(yolo_json_path, output_json_path):
    import json

    with open(yolo_json_path, "r") as f:
        yolo = json.load(f)

    if len(yolo) != 17:
        raise ValueError("YOLO must output 17 keypoints")

    # helper
    def mid(a, b):
        return {
            "x": (a["x"] + b["x"]) / 2,
            "y": (a["y"] + b["y"]) / 2,
            "conf": (a["conf"] + b["conf"]) / 2,
        }

    # Build 25-keypoint OpenPose format required by PIFuHD
    # PIFuHD expects full OpenPose 25-keypoint format
    k = yolo  # alias

    # YOLO has 17 keypoints, OpenPose needs 25
    # Map YOLO to OpenPose indices
    openpose_kp = [None] * 25
    
    # Body keypoints (0-16 in OpenPose = 0-16 in YOLO)
    openpose_kp[0] = k[0]   # Nose
    openpose_kp[1] = k[1]   # L Eye
    openpose_kp[2] = k[2]   # R Eye
    openpose_kp[3] = k[3]   # L Ear
    openpose_kp[4] = k[4]   # R Ear
    openpose_kp[5] = k[5]   # L Shoulder
    openpose_kp[6] = k[6]   # R Shoulder
    openpose_kp[7] = k[7]   # L Elbow
    openpose_kp[8] = k[8]   # R Elbow
    openpose_kp[9] = k[9]   # L Wrist
    openpose_kp[10] = k[10] # R Wrist
    openpose_kp[11] = k[11] # L Hip
    openpose_kp[12] = k[12] # R Hip
    openpose_kp[13] = k[13] # L Knee
    openpose_kp[14] = k[14] # R Knee
    openpose_kp[15] = k[15] # L Ankle
    openpose_kp[16] = k[16] # R Ankle
    
    # Neck (17) = midpoint of shoulders
    openpose_kp[17] = mid(k[5], k[6])
    
    # Hand keypoints (18-21): use wrist as placeholder
    openpose_kp[18] = k[9]   # L Hand (use L Wrist)
    openpose_kp[19] = k[10]  # R Hand (use R Wrist)
    openpose_kp[20] = k[9]   # L Hand thumb (use L Wrist)
    openpose_kp[21] = k[10]  # R Hand thumb (use R Wrist)
    
    # Foot keypoints (22-24): use ankles as placeholders
    openpose_kp[22] = k[15]  # L Big Toe (use L Ankle)
    openpose_kp[23] = k[16]  # R Big Toe (use R Ankle)
    openpose_kp[24] = mid(k[15], k[16])  # Background (use ankle midpoint)

    # OpenPose JSON - format as flat array
    pose = []
    for kp in openpose_kp:
        pose.extend([kp["x"], kp["y"], kp["conf"]])

    output = {
        "version": 1.0,
        "people": [
            {"pose_keypoints_2d": pose}
        ]
    }

    with open(output_json_path, "w") as f:
        json.dump(output, f, indent=4)

    print("✔ Converted → PIFuHD 25-keypoint OpenPose JSON")
    return output_json_path

# ------------------------------------------------------------
# Main PIFuHD Reconstruction
# ------------------------------------------------------------
def reconstruct(input_image, yolo_keypoints_json, output_dir):
    """
    Runs PIFuHD inference with:
      - input image
      - YOLO keypoints → converted to OpenPose format
    """

    repo = os.path.join(os.path.dirname(__file__), "pifuhd_repo")
    input_dir = os.path.join(repo, "input")
    ckpt = os.path.join(repo, "checkpoints", "pifuhd.pt")

    if not os.path.exists(ckpt):
        raise FileNotFoundError("Missing PIFuHD checkpoint: " + ckpt)

    # Prepare clean input folder inside PIFuHD repo
    if os.path.exists(input_dir):
        shutil.rmtree(input_dir)
    os.makedirs(input_dir)

    # Filenames expected by PIFuHD
    image_name = "input.png"
    keypoints_name = "input_keypoints.json"

    # Copy input image
    shutil.copy(input_image, os.path.join(input_dir, image_name))

    # Convert pose JSON
    openpose_json_path = os.path.join(input_dir, keypoints_name)
    convert_yolo_to_pifuhd(yolo_keypoints_json, openpose_json_path)

    # Ensure output folder exists
    os.makedirs(output_dir, exist_ok=True)

    # Command to run PIFuHD simple_test
    cmd = [
        "python",
        "-m", "apps.simple_test",
        "-i", input_dir,
        "-o", output_dir,
        "-c", ckpt
    ]

    print("Running PIFuHD...")
    subprocess.run(cmd, cwd=repo, check=True)
    print("PIFuHD Complete.")

    # Final recon output path
    recon_obj = os.path.join(output_dir, "pifuhd_output", "recon.obj")
    if not os.path.exists(recon_obj):
        raise FileNotFoundError("PIFuHD did not generate recon.obj")

    return recon_obj
