import os
import cv2
import json
import numpy as np
import shutil
from convert_yolo_to_openpose import convert_yolo_to_openpose

# ---------------------------------------------------------
# Create 18-channel pose heatmap
# ---------------------------------------------------------
def create_pose_map(op_kps, h, w):
    pose_map = np.zeros((18, h, w), dtype=np.float32)

    # CP-VTON++ requires this key
    if "pose_keypoints" in op_kps["people"][0]:
        arr = op_kps["people"][0]["pose_keypoints"]
    else:
        raise KeyError("ERROR: pose_keypoints missing in pose JSON")

    for i in range(18):
        x, y, c = arr[i*3:(i*3)+3]
        if c > 0.1:
            cv2.circle(pose_map[i], (int(x), int(y)), 4, 1, -1)

    return pose_map

def save_parsing_as_rgb(src, dst):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Parsing image missing: {src}")

    # Ensure output is ALWAYS 3 channels
    if len(img.shape) == 2:  # grayscale
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    elif img.shape[2] == 4:  # RGBA
        img = cvv.cvtColor(img, cv2.COLOR_BGRA2BGR)

    elif img.shape[2] == 1:  # single channel weird case
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(dst, img)
    print("✔ Saved Parsing RGB →", dst)

# ---------------------------------------------------------
# Save rendered pose image
# ---------------------------------------------------------
def save_pose_render(pose_map, out_path):
    img = np.sum(pose_map, axis=0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(out_path, img)
    print("✔ Pose rendered →", out_path)


# ---------------------------------------------------------
# Force-saving an image as 3-channel RGB PNG
# ---------------------------------------------------------
def save_png_rgb(src, dst):
    img = cv2.imread(src, cv2.IMREAD_COLOR)
    cv2.imwrite(dst, img)
    print("✔ Saved RGB →", dst)


# ---------------------------------------------------------
# Save parsing OR mask always as RGB (fixes channel mismatch)
# ---------------------------------------------------------
def save_parsing_as_rgb(src, dst):
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)

    if img is None:
        raise ValueError(f"Missing parsing image: {src}")

    # grayscale → RGB
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # RGBA → RGB
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    cv2.imwrite(dst, img)
    print("✔ Saved Parsing RGB →", dst)


# ---------------------------------------------------------
# MAIN — build CP-VTON++ test dataset
# ---------------------------------------------------------
def prepare(stage1_dir, dataset_root):

    print("### Preparing CP-VTON++ dataset...")

    # Stage 1 output files
    person = os.path.join(stage1_dir, "person_clean.png")
    cloth = os.path.join(stage1_dir, "cloth_clean.png")
    cloth_mask = os.path.join(stage1_dir, "cloth_mask.png")
    parsing = os.path.join(stage1_dir, "parsing.png")
    pose_json = os.path.join(stage1_dir, "pose.json")

    person_img = cv2.imread(person)
    H, W = person_img.shape[:2]

    # dataset_root = cp_vton_pp/dataset
    test = os.path.join(dataset_root, "test")

    folders = {
        "image": os.path.join(test, "image"),
        "cloth": os.path.join(test, "cloth"),
        "cloth-mask": os.path.join(test, "cloth-mask"),
        "openpose-json": os.path.join(test, "openpose-json"),
        "pose": os.path.join(test, "pose"),
        "image-parse": os.path.join(test, "image-parse"),
        "image-mask": os.path.join(test, "image-mask")
    }

    for p in folders.values():
        os.makedirs(p, exist_ok=True)

    name = "00001"

    # -----------------------------------------------------
    # Convert YOLO→OpenPose
    # -----------------------------------------------------
    op_json_path = os.path.join(folders["openpose-json"], f"{name}.json")
    openpose_kps = convert_yolo_to_openpose(pose_json, op_json_path)
    print("Parsing shape:", cv2.imread(parsing, cv2.IMREAD_UNCHANGED).shape)

    # -----------------------------------------------------
    # Pose map + render
    # -----------------------------------------------------
    pose_map = create_pose_map(openpose_kps, H, W)

    pose_render = os.path.join(folders["pose"], f"{name}_rendered.png")
    save_pose_render(pose_map, pose_render)

    pose_json_final = os.path.join(folders["pose"], f"{name}_keypoints.json")
    shutil.copy(op_json_path, pose_json_final)
    print("✔ Saved pose JSON →", pose_json_final)

    # -----------------------------------------------------
    # Save all PNG images as RGB
    # -----------------------------------------------------
    save_png_rgb(person, os.path.join(folders["image"], f"{name}.png"))
    save_png_rgb(cloth, os.path.join(folders["cloth"], f"{name}.png"))
    save_png_rgb(cloth_mask, os.path.join(folders["cloth-mask"], f"{name}.png"))

    # parsing: convert grayscale→RGB
    save_parsing_as_rgb(parsing, os.path.join(folders["image-parse"], f"{name}.png"))

    # parsing-mask for CP-VTON++: must ALSO be RGB
    save_parsing_as_rgb(parsing, os.path.join(folders["image-mask"], f"{name}.png"))

    # Create image-parse-new folder for CP-VTON++ compatibility
    folders["image-parse-new"] = os.path.join(test, "image-parse-new")
    os.makedirs(folders["image-parse-new"], exist_ok=True)

    # Save parsing again as image-parse-new
    save_parsing_as_rgb(parsing, os.path.join(folders["image-parse-new"], f"{name}.png"))
    print("✔ Saved Parsing (image-parse-new) →", os.path.join(folders["image-parse-new"], f"{name}.png"))


    # -----------------------------------------------------
    # test_pairs.txt
    # -----------------------------------------------------
    pairs_path = os.path.join(dataset_root, "test_pairs.txt")
    with open(pairs_path, "w") as f:
        f.write(f"{name}.png {name}.png\n")

    print(f"✔ Created test_pairs.txt → {pairs_path}")
    print("### CP-VTON++ dataset READY:", dataset_root)
