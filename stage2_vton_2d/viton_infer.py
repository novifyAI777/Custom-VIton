import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# --- Import VITON-HD modules ---
from viton_hd_repo.networks import ALIASGenerator, GMM, SegGenerator
from viton_hd_repo.utils import save_images


class VITONHD:

    def __init__(self, ckpt_dir="stage2_vton_2d/viton_hd_repo/checkpoints", device="cpu"):
        self.device = torch.device(device)

        # Create opt object with required parameters
        class Opt:
            def __init__(self):
                self.semantic_nc = 13
                self.init_type = 'xavier'
                self.init_variance = 0.02
                self.grid_size = 5
                self.norm_G = 'spectralaliasinstance'
                self.ngf = 64
                self.num_upsampling_layers = 'most'
                self.load_height = 1024
                self.load_width = 768
        
        self.opt = Opt()

        # Load checkpoints
        gmm_path = os.path.join(ckpt_dir, "gmm_final.pth")
        alias_path = os.path.join(ckpt_dir, "alias_final.pth")
        seg_path = os.path.join(ckpt_dir, "seg_final.pth")

        # -----------------------
        # Load Segmentation Module
        # -----------------------
        self.seg = SegGenerator(self.opt, input_nc=self.opt.semantic_nc + 8, output_nc=self.opt.semantic_nc)
        self.seg.load_state_dict(torch.load(seg_path, map_location=self.device))
        self.seg.to(self.device).eval()

        # -----------------------
        # Load GMM
        # -----------------------
        self.gmm = GMM(self.opt, inputA_nc=7, inputB_nc=3)
        self.gmm.load_state_dict(torch.load(gmm_path, map_location=self.device))
        self.gmm.to(self.device).eval()

        # -----------------------
        # Load ALIAS Generator
        # -----------------------
        self.opt.semantic_nc = 7  # Change for ALIAS
        self.alias = ALIASGenerator(self.opt, input_nc=9)
        self.alias.load_state_dict(torch.load(alias_path, map_location=self.device))
        self.alias.to(self.device).eval()
        self.opt.semantic_nc = 13  # Restore

    # -----------------------
    # Transform inputs
    # -----------------------
    def preprocess(self, person, cloth):
        tf = transforms.Compose([
            transforms.Resize((512, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        p = tf(Image.open(person).convert("RGB"))
        c = tf(Image.open(cloth).convert("RGB"))
        return p.unsqueeze(0), c.unsqueeze(0)

    # -----------------------
    # Inference Pipeline
    # -----------------------
    def infer(self, person_img, cloth_img, save_path="tryon_result.png"):
        p, c = self.preprocess(person_img, cloth_img)
        p, c = p.to(self.device), c.to(self.device)

        # Step 1 — Predict segmentation
        with torch.no_grad():
            seg_mask = self.seg(p)

        # Step 2 — GMM warps clothing onto body
        with torch.no_grad():
            grid, theta = self.gmm(p, c)
            warped_cloth = torch.nn.functional.grid_sample(c, grid, padding_mode="border")

        # Step 3 — ALIAS Generator produces final try-on
        with torch.no_grad():
            output = self.alias(p, warped_cloth, seg_mask)

        # Convert tensor to image and save
        tensor = (output[0].clone() + 1) * 0.5 * 255
        tensor = tensor.cpu().clamp(0, 255)
        array = tensor.detach().numpy().astype('uint8')
        
        if array.shape[0] == 3:
            array = array.transpose(1, 2, 0)
        
        img = Image.fromarray(array)
        img.save(save_path)

        print(f"[OK] VITON-HD Output Saved → {save_path}")
        return save_path


# -------------------------------
# Standalone usage
# -------------------------------
if __name__ == "__main__":
    import sys
    person = sys.argv[1]
    cloth = sys.argv[2]
    out = sys.argv[3] if len(sys.argv) > 3 else "tryon_result.png"

    viton = VITONHD()
    viton.infer(person, cloth, out)
