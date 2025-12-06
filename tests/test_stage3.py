"""
Test Stage 3: 3D Reconstruction
Tests PIFuHD reconstruction from 2D try-on image to 3D mesh
"""

import os
import sys
import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stage3_3d_reconstruction.run_stage3 import run_stage3


class TestStage3:
    """Test cases for Stage 3 - 3D Reconstruction"""
    
    @pytest.fixture
    def setup_paths(self, tmp_path):
        """Setup test paths"""
        input_img = "data/stage2_output/tryon_result.png"
        output_dir = tmp_path / "stage3_output"
        output_dir.mkdir(exist_ok=True)
        return input_img, str(output_dir)
    
    def test_stage3_basic(self, setup_paths):
        """Test basic Stage 3 execution"""
        input_img, output_dir = setup_paths
        
        # Skip if input image doesn't exist
        if not os.path.exists(input_img):
            pytest.skip(f"Input image not found: {input_img}")
        
        # Run Stage 3
        run_stage3(input_img, output_dir)
        
        # Check output mesh exists
        output_mesh = os.path.join(output_dir, "model.obj")
        assert os.path.exists(output_mesh), f"Output mesh not created: {output_mesh}"
        assert os.path.getsize(output_mesh) > 0, "Output mesh is empty"
    
    def test_stage3_output_format(self, setup_paths):
        """Test that output mesh is in OBJ format"""
        input_img, output_dir = setup_paths
        
        if not os.path.exists(input_img):
            pytest.skip(f"Input image not found: {input_img}")
        
        run_stage3(input_img, output_dir)
        
        output_mesh = os.path.join(output_dir, "model.obj")
        assert output_mesh.endswith(".obj"), "Output should be OBJ format"
        
        # Check file contains OBJ format markers
        with open(output_mesh, 'r') as f:
            content = f.read(1000)
            assert 'v ' in content or 'vn ' in content or 'f ' in content, \
                "Output file doesn't appear to be valid OBJ format"
    
    def test_stage3_directory_creation(self):
        """Test that output directory is created if it doesn't exist"""
        input_img = "data/stage2_output/tryon_result.png"
        output_dir = "data/stage3_output_test"
        
        if not os.path.exists(input_img):
            pytest.skip(f"Input image not found: {input_img}")
        
        # Remove directory if it exists
        if os.path.exists(output_dir):
            import shutil
            shutil.rmtree(output_dir)
        
        # Run Stage 3
        run_stage3(input_img, output_dir)
        
        # Check directory was created
        assert os.path.exists(output_dir), "Output directory not created"
        
        # Cleanup
        import shutil
        shutil.rmtree(output_dir)


def test_import_stage3():
    """Test that Stage 3 modules can be imported"""
    try:
        from stage3_3d_reconstruction.run_stage3 import run_stage3
        from stage3_3d_reconstruction.pifuhd.pifuhd_infer import reconstruct
        assert callable(run_stage3)
        assert callable(reconstruct)
    except ImportError as e:
        pytest.fail(f"Failed to import Stage 3 modules: {e}")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
