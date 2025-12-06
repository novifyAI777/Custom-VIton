"""
3D Mesh Visualization Utilities
Supports interactive viewing of OBJ, GLB, and other 3D mesh formats
"""

import os
import numpy as np


def visualize_mesh_open3d(mesh_path):
    """
    Visualize a 3D mesh using Open3D.
    Supports OBJ, STL, PLY, GLB formats.
    
    Args:
        mesh_path: Path to the mesh file
    """
    try:
        import open3d as o3d
    except ImportError:
        print("Error: open3d not installed. Install with: pip install open3d")
        return False
    
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        return False
    
    print(f"Loading mesh: {mesh_path}")
    
    # Load mesh
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    
    if mesh.is_empty():
        print("Error: Mesh is empty!")
        return False
    
    # Print mesh info
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    
    # Compute normals if not present
    mesh.compute_vertex_normals()
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"3D Mesh Viewer - {os.path.basename(mesh_path)}")
    
    vis.add_geometry(mesh)
    
    # Set render options
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.1, 0.1, 0.1])
    opt.line_width = 1.0
    
    # Center view on mesh
    vis.reset_view_point(True)
    
    print("\nViewer Controls:")
    print("  Left click + drag: Rotate")
    print("  Right click + drag: Zoom")
    print("  Middle click + drag: Pan")
    print("  P: Capture screenshot")
    print("  ESC or Q: Close viewer")
    print("\nStarting viewer...")
    
    vis.run()
    vis.destroy_window()
    
    return True


def visualize_mesh_trimesh(mesh_path):
    """
    Visualize a 3D mesh using trimesh and pyrender.
    Supports OBJ, STL, PLY, GLB formats.
    
    Args:
        mesh_path: Path to the mesh file
    """
    try:
        import trimesh
        import pyrender
    except ImportError:
        print("Error: trimesh or pyrender not installed.")
        print("Install with: pip install trimesh pyrender")
        return False
    
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        return False
    
    print(f"Loading mesh: {mesh_path}")
    
    # Load mesh
    mesh = trimesh.load(mesh_path)
    
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.faces)}")
    
    # Create scene
    scene = pyrender.Scene([pyrender.Mesh.from_trimesh(mesh)])
    
    # Add light
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    scene.add(light)
    
    print("\nViewer Controls:")
    print("  Left click + drag: Rotate")
    print("  Right click + drag: Zoom")
    print("  Middle click + drag: Pan")
    print("  ESC or Q: Close viewer")
    print("\nStarting viewer...")
    
    pyrender.Viewer(scene)
    
    return True


def visualize_mesh(mesh_path, viewer='open3d'):
    """
    Main visualization function.
    
    Args:
        mesh_path: Path to the 3D mesh file
        viewer: 'open3d', 'trimesh', or 'auto' (try open3d first)
    """
    if not os.path.exists(mesh_path):
        print(f"Error: Mesh file not found: {mesh_path}")
        return False
    
    if viewer == 'auto':
        # Try open3d first
        try:
            import open3d
            viewer = 'open3d'
        except ImportError:
            try:
                import trimesh
                viewer = 'trimesh'
            except ImportError:
                print("Error: Neither open3d nor trimesh is installed.")
                print("Install one with:")
                print("  pip install open3d")
                print("  OR")
                print("  pip install trimesh pyrender")
                return False
    
    if viewer == 'open3d':
        return visualize_mesh_open3d(mesh_path)
    elif viewer == 'trimesh':
        return visualize_mesh_trimesh(mesh_path)
    else:
        print(f"Error: Unknown viewer '{viewer}'. Use 'open3d' or 'trimesh'")
        return False


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python vis.py <mesh_file> [viewer]")
        print("\nExamples:")
        print("  python vis.py data/stage3_output/result_person_clean_256.obj")
        print("  python vis.py data/stage4_output/model.glb open3d")
        print("  python vis.py data/stage4_output/model.obj trimesh")
        print("\nViewers: open3d, trimesh, auto (default)")
        sys.exit(1)
    
    mesh_file = sys.argv[1]
    viewer_type = sys.argv[2] if len(sys.argv) > 2 else 'auto'
    
    visualize_mesh(mesh_file, viewer_type)
