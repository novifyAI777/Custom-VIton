"""
Auto-generated template file
Pose wrapper
"""

from .yolov8_pose.yolov8_pose_infer import YOLOv8Pose
def run(input_path, out_path, weights=None):
    model = YOLOv8Pose(weights)
    kps = model.predict(input_path)
    model.save_keypoints(kps, out_path)
    print(f"Saved pose to {out_path}")

if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv)>1 else '../../data/stage1_output/person_clean.png'
    outp = sys.argv[2] if len(sys.argv)>2 else '../../data/stage1_output/pose_keypoints.json'
    run(inp, outp)
