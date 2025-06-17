# handler.py
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO
import base64
import os

# Load the model only once (global)
model = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

def decode_base64_image(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str)))

def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def handler(event):
    """RunPod-compatible handler function"""
    prompt = event.get("prompt", "")
    image_b64 = event.get("image")
    mask_b64 = event.get("mask")

    if not prompt or not image_b64 or not mask_b64:
        return {"error": "Missing prompt, image or mask."}

    image = decode_base64_image(image_b64).convert("RGB")
    mask = decode_base64_image(mask_b64).convert("RGB")

    result_image = model(prompt=prompt, image=image, mask_image=mask).images[0]
    result_b64 = encode_base64_image(result_image)

    return {"output_image": result_b64}
