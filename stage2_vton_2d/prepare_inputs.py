import os
import shutil
from PIL import Image

def prepare_inputs(person_img, cloth_img, viton_root):
    """
    Prepares input images for VITON-HD.
    
    Arguments:
        person_img (str): path to cleaned person image
        cloth_img (str): path to cleaned cloth image
        viton_root (str): path to viton_hd_repo/
    
    Output:
        viton_hd_repo/test_person/person.png
        viton_hd_repo/test_clothes/cloth.png
    """

    test_person_dir = os.path.join(viton_root, "test_person")
    test_clothes_dir = os.path.join(viton_root, "test_clothes")

    os.makedirs(test_person_dir, exist_ok=True)
    os.makedirs(test_clothes_dir, exist_ok=True)

    # Standard names expected by inference script
    dst_person = os.path.join(test_person_dir, "person.png")
    dst_cloth = os.path.join(test_clothes_dir, "cloth.png")

    # Copy inputs
    shutil.copy(person_img, dst_person)
    shutil.copy(cloth_img, dst_cloth)

    # Validate images
    try:
        Image.open(dst_person).verify()
        Image.open(dst_cloth).verify()
        print("✔ Input images copied & validated for VITON-HD.")
    except Exception as e:
        print("❌ Image validation failed:", e)

    return dst_person, dst_cloth
