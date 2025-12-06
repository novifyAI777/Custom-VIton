"""
Preprocessing for VITON-HD inputs
Creates agnostic images, pose heatmaps, and proper directory structure
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw
import cv2


def load_pose_keypoints(json_path):
    """Load OpenPose keypoints from JSON"""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Handle different formats
    if isinstance(data, list):
        # Format: [{"x": ..., "y": ..., "conf": ...}, ...]
        keypoints = np.array([[kp['x'], kp['y']] for kp in data], dtype=np.float32)
    elif 'people' in data and len(data['people']) > 0:
        # OpenPose format: {"people": [{"pose_keypoints_2d": [x,y,c,...]}]}
        keypoints_flat = data['people'][0]['pose_keypoints_2d']
        keypoints = np.array(keypoints_flat, dtype=np.float32).reshape(-1, 3)[:, :2]
    elif 'pose_keypoints_2d' in data:
        # Direct format: {"pose_keypoints_2d": [x,y,c,...]}
        keypoints = np.array(data['pose_keypoints_2d'], dtype=np.float32).reshape(-1, 3)[:, :2]
    else:
        raise ValueError("Invalid pose keypoints format")
    
    return keypoints


def render_pose_image(pose_data, img_size=(768, 1024), line_color=(255, 255, 255), point_color=(0, 255, 0)):
    """
    Render pose keypoints as an RGB image
    img_size: (width, height)
    """
    # Create blank image
    pose_img = Image.new('RGB', img_size, (0, 0, 0))
    draw = ImageDraw.Draw(pose_img)
    
    # OpenPose 25-point skeleton connections
    # [start_point, end_point]
    connections = [
        [0, 1], [1, 2], [2, 3], [3, 4],  # Head
        [1, 5], [5, 6], [6, 7],  # Right arm
        [1, 8], [8, 9], [9, 10],  # Left arm
        [1, 11], [11, 12], [12, 13],  # Right leg
        [1, 14], [14, 15], [15, 16],  # Left leg
        [0, 15], [0, 16], [15, 17], [16, 18],  # Face
        [5, 17], [6, 18],  # Additional connections
    ]
    
    # Draw lines
    for connection in connections:
        start_idx, end_idx = connection
        if start_idx < len(pose_data) and end_idx < len(pose_data):
            start = pose_data[start_idx]
            end = pose_data[end_idx]
            # Check if both points are valid
            if not (start[0] == 0 and start[1] == 0) and not (end[0] == 0 and end[1] == 0):
                draw.line([tuple(start), tuple(end)], fill=line_color, width=3)
    
    # Draw keypoints
    for point in pose_data:
        if not (point[0] == 0 and point[1] == 0):
            x, y = point
            r = 3
            draw.ellipse((x-r, y-r, x+r, y+r), fill=point_color)
    
    return pose_img


def create_agnostic_person(person_img, parse_img, pose_data):
    """
    Create agnostic person image (person with upper clothes masked out)
    
    Args:
        person_img: PIL Image of person
        parse_img: PIL Image of parsing segmentation (grayscale with class indices)
        pose_data: numpy array of pose keypoints (N, 2)
    
    Returns:
        PIL Image of agnostic person
    """
    parse_array = np.array(parse_img)
    
    # Define parsing classes (adjust based on your parsing model)
    # Typically: background=0, hair=1, face=2, upper=3, pants=4, arms=5,6, etc.
    parse_head = ((parse_array == 1) | (parse_array == 2) | (parse_array == 4) | (parse_array == 13)).astype(np.float32)
    parse_lower = ((parse_array == 9) | (parse_array == 12) | (parse_array == 16) | 
                   (parse_array == 17) | (parse_array == 18) | (parse_array == 19)).astype(np.float32)
    
    r = 20
    agnostic = person_img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)
    
    # Adjust hip keypoints to match shoulder width
    if len(pose_data) >= 13:
        length_a = np.linalg.norm(pose_data[5] - pose_data[2])
        length_b = np.linalg.norm(pose_data[12] - pose_data[9])
        if length_b > 0:
            point = (pose_data[9] + pose_data[12]) / 2
            pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
            pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    
    # Mask arms
    if len(pose_data) >= 8:
        # Right arm: shoulder to elbow to wrist
        for i in [2, 5]:
            if i < len(pose_data):
                x, y = pose_data[i]
                if not (x == 0 and y == 0):
                    agnostic_draw.line([tuple(pose_data[2]), tuple(pose_data[5])], fill='gray', width=r*10)
                    agnostic_draw.ellipse((x-r*5, y-r*5, x+r*5, y+r*5), fill='gray')
        
        # Left arm
        for i in [3, 4, 6, 7]:
            if i < len(pose_data) and i-1 < len(pose_data):
                curr = pose_data[i]
                prev = pose_data[i-1]
                if not (curr[0] == 0 and curr[1] == 0) and not (prev[0] == 0 and prev[1] == 0):
                    agnostic_draw.line([tuple(prev), tuple(curr)], fill='gray', width=r*10)
                    agnostic_draw.ellipse((curr[0]-r*5, curr[1]-r*5, curr[0]+r*5, curr[1]+r*5), fill='gray')
    
    # Mask torso
    if len(pose_data) >= 13:
        for i in [9, 12]:
            if i < len(pose_data):
                x, y = pose_data[i]
                if not (x == 0 and y == 0):
                    agnostic_draw.ellipse((x-r*3, y-r*6, x+r*3, y+r*6), fill='gray')
        
        # Draw torso polygon
        agnostic_draw.line([tuple(pose_data[2]), tuple(pose_data[9])], fill='gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[5]), tuple(pose_data[12])], fill='gray', width=r*6)
        agnostic_draw.line([tuple(pose_data[9]), tuple(pose_data[12])], fill='gray', width=r*12)
        agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], fill='gray')
    
    # Mask neck
    if len(pose_data) >= 2:
        x, y = pose_data[1]
        if not (x == 0 and y == 0):
            agnostic_draw.rectangle((x-r*7, y-r*7, x+r*7, y+r*7), fill='gray')
    
    # Paste back head and lower body
    agnostic.paste(person_img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(person_img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    
    return agnostic


def create_agnostic_parse(parse_img, pose_data):
    """
    Create agnostic parsing (parsing with upper clothes/arms removed)
    
    Args:
        parse_img: PIL Image of parsing segmentation
        pose_data: numpy array of pose keypoints (N, 2)
    
    Returns:
        PIL Image of agnostic parsing
    """
    parse_array = np.array(parse_img)
    parse_upper = ((parse_array == 5) | (parse_array == 6) | (parse_array == 7)).astype(np.float32)
    parse_neck = (parse_array == 10).astype(np.float32)
    
    r = 10
    agnostic = parse_img.copy()
    
    # Mask arms (classes 14 and 15 for left/right arms)
    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        mask_arm = Image.new('L', parse_img.size, 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        
        i_prev = pose_ids[0]
        for i in pose_ids[1:]:
            if i >= len(pose_data) or i_prev >= len(pose_data):
                continue
            if (pose_data[i_prev, 0] == 0 and pose_data[i_prev, 1] == 0) or \
               (pose_data[i, 0] == 0 and pose_data[i, 1] == 0):
                continue
            
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10)
            x, y = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((x-radius, y-radius, x+radius, y+radius), 'white', 'white')
            i_prev = i
        
        parse_arm = (np.array(mask_arm) / 255) * (parse_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
    
    # Mask torso & neck
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_upper * 255), 'L'))
    agnostic.paste(0, None, Image.fromarray(np.uint8(parse_neck * 255), 'L'))
    
    return agnostic


def setup_viton_dataset(stage1_dir, viton_dataset_dir, img_size=(768, 1024)):
    """
    Create VITON-HD dataset structure from Stage 1 outputs
    
    Args:
        stage1_dir: Path to Stage 1 outputs
        viton_dataset_dir: Path where VITON-HD dataset will be created
        img_size: (width, height) for resizing
    
    Returns:
        Path to created dataset
    """
    print("Setting up VITON-HD dataset structure...")
    
    # Create directory structure
    test_dir = os.path.join(viton_dataset_dir, 'test')
    dirs = ['image', 'cloth', 'cloth-mask', 'image-parse', 'openpose-img', 'openpose-json']
    for d in dirs:
        os.makedirs(os.path.join(test_dir, d), exist_ok=True)
    
    # Load inputs from Stage 1
    person_path = os.path.join(stage1_dir, 'person_clean.png')
    cloth_path = os.path.join(stage1_dir, 'cloth_clean.png')
    cloth_mask_path = os.path.join(stage1_dir, 'cloth_mask.png')
    parsing_path = os.path.join(stage1_dir, 'parsing.png')
    pose_json_path = os.path.join(stage1_dir, 'pose_keypoints.json')
    
    # Standard names (use .jpg as VITON-HD expects)
    person_name = 'person.jpg'
    cloth_name = 'cloth.jpg'
    
    # 1. Copy and resize person image
    print("Processing person image...")
    person_img = Image.open(person_path).convert('RGB')
    person_img = person_img.resize(img_size, Image.BILINEAR)
    person_img.save(os.path.join(test_dir, 'image', person_name))
    
    # 2. Copy and resize cloth image
    print("Processing cloth image...")
    cloth_img = Image.open(cloth_path).convert('RGB')
    cloth_img = cloth_img.resize(img_size, Image.BILINEAR)
    cloth_img.save(os.path.join(test_dir, 'cloth', cloth_name))
    
    # 3. Copy and resize cloth mask
    print("Processing cloth mask...")
    cloth_mask = Image.open(cloth_mask_path).convert('L')
    cloth_mask = cloth_mask.resize(img_size, Image.NEAREST)
    cloth_mask.save(os.path.join(test_dir, 'cloth-mask', cloth_name))
    
    # 4. Copy and resize parsing (preserve class indices)
    print("Processing parsing...")
    parsing = Image.open(parsing_path)
    # Convert palette image to numpy array to preserve indices
    parse_array = np.array(parsing)
    # Resize using cv2 with NEAREST interpolation to preserve indices
    parse_resized = cv2.resize(parse_array, img_size, interpolation=cv2.INTER_NEAREST)
    parse_save_name = person_name.replace('.jpg', '.png')
    Image.fromarray(parse_resized).save(os.path.join(test_dir, 'image-parse', parse_save_name))
    
    # 5. Load and process pose
    print("Processing pose...")
    pose_data = load_pose_keypoints(pose_json_path)
    
    # Scale pose keypoints to match resized image
    orig_size = Image.open(person_path).size  # (width, height)
    scale_x = img_size[0] / orig_size[0]
    scale_y = img_size[1] / orig_size[1]
    pose_data[:, 0] *= scale_x
    pose_data[:, 1] *= scale_y
    
    # 6. Render pose image
    print("Rendering pose image...")
    pose_img = render_pose_image(pose_data, img_size)
    pose_rendered_name = person_name.replace('.png', '_rendered.png')
    pose_img.save(os.path.join(test_dir, 'openpose-img', pose_rendered_name))
    
    # 7. Save pose JSON in VITON-HD format (with confidence values)
    print("Saving pose keypoints...")
    pose_json_name = person_name.replace('.jpg', '_keypoints.json')
    
    # Create flat list with confidence: [x1, y1, conf1, x2, y2, conf2, ...]
    pose_with_conf = []
    for point in pose_data:
        pose_with_conf.extend([float(point[0]), float(point[1]), 1.0])  # confidence = 1.0
    
    viton_pose = {
        'people': [{
            'pose_keypoints_2d': pose_with_conf
        }]
    }
    with open(os.path.join(test_dir, 'openpose-json', pose_json_name), 'w') as f:
        json.dump(viton_pose, f)
    
    # 8. Create pair list
    print("Creating pair list...")
    pair_list_path = os.path.join(viton_dataset_dir, 'test_pairs.txt')
    with open(pair_list_path, 'w') as f:
        f.write(f'{person_name} {cloth_name}\n')
    
    print(f"[OK] VITON-HD dataset created at {test_dir}")
    return viton_dataset_dir


if __name__ == '__main__':
    import sys
    stage1_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/stage1_output'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else 'stage2_vton_2d/viton_hd_repo/datasets'
    
    setup_viton_dataset(stage1_dir, output_dir)
