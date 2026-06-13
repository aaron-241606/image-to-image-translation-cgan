"""
Task-04: Translate a single image using trained pix2pix Generator
Prodigy Infotech Internship
"""

import argparse
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
from model import UNetGenerator


def predict(model_path, input_path, output_path, img_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G = UNetGenerator().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    img = Image.open(input_path).convert("RGB")
    tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = G(tensor)

    save_image(output * 0.5 + 0.5, output_path)
    print(f"✅ Translated image saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate image with pix2pix")
    parser.add_argument("--model",    type=str, required=True,          help="Path to generator checkpoint")
    parser.add_argument("--input",    type=str, required=True,          help="Input image path")
    parser.add_argument("--output",   type=str, default="result.jpg",   help="Output image path")
    parser.add_argument("--img_size", type=int, default=256,            help="Image size")
    args = parser.parse_args()
    predict(args.model, args.input, args.output, args.img_size)
