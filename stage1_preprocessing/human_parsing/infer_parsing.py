"""
Auto-generated template file
Parsing wrapper
"""

from PIL import Image
import os
from schp.schp_infer import SCHPParser

def run(input_path, out_parsing_path, checkpoint=None):
    img = Image.open(input_path).convert('RGB')
    parser = SCHPParser(checkpoint)
    labels = parser.parse(img)
    parser.save_parsing(labels, out_parsing_path)
    print(f"Saved parsing map to {out_parsing_path}")

if __name__ == '__main__':
    import sys
    inp = sys.argv[1] if len(sys.argv)>1 else '../../data/stage1_output/person_clean.png'
    outp = sys.argv[2] if len(sys.argv)>2 else '../../data/stage1_output/parsing.png'
    run(inp, outp)
