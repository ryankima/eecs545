import os
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
import models.transformer as transformer
import models.StyTR as StyTR

# Define transformations
def test_transform(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])

def style_transform(h, w):
    return transforms.Compose([
        transforms.Resize((h, w)),  # Ensure style matches content size
        transforms.ToTensor()
    ])

def load_model_weights(model, path):
    state_dict = torch.load(path, map_location="cpu")  # Ensure compatibility across devices
    new_state_dict = {k: v for k, v in state_dict.items()}  # Handle possible prefix issues
    model.load_state_dict(new_state_dict)

# Function to perform stylization
def stylize_image(content_path, style_path, output_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    # Model paths (modify if needed)
    vgg_path = "./experiments/vgg_normalised.pth"
    decoder_path = "experiments/decoder_iter_160000.pth"
    trans_path = "experiments/transformer_iter_160000.pth"
    embedding_path = "experiments/embedding_iter_160000.pth"

    # Load pre-trained models
    vgg = StyTR.vgg
    vgg.load_state_dict(torch.load(vgg_path, map_location="cpu"))
    vgg = nn.Sequential(*list(vgg.children())[:44])  # Keep only first 44 layers

    decoder = StyTR.decoder
    Trans = transformer.Transformer()
    embedding = StyTR.PatchEmbed()

    decoder.eval()
    Trans.eval()
    vgg.eval()

    # Load model weights
    load_model_weights(decoder, decoder_path)
    load_model_weights(Trans, trans_path)
    load_model_weights(embedding, embedding_path)

    # Initialize network
    network = StyTR.StyTrans(vgg, decoder, embedding, Trans, args=None)
    network.eval().to(device)

    # Load content and style images
    content_img = Image.open(content_path).convert("RGB")
    style_img = Image.open(style_path).convert("RGB")

    # Resize style image to match content size
    content_tf = test_transform(512)
    content = content_tf(content_img)
    h, w = content.shape[1:]  # Extract content image dimensions

    style_tf = style_transform(h, w)  # Ensure style matches content dimensions
    style = style_tf(style_img)

    # Move tensors to device
    content = content.to(device).unsqueeze(0)
    style = style.to(device).unsqueeze(0)

    # Perform stylization
    with torch.no_grad():
        output = network(content, style)[0].cpu()  # Extract first element from tuple

    # Ensure output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Save the stylized image
    output_name = os.path.join(output_path, f"{Path(content_path).stem}_stylized_{Path(style_path).stem}.jpg")
    save_image(output, output_name)

    print(f"Stylized image saved to {output_name}")
    return output_name