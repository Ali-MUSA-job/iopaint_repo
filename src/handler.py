# src/handler.py

import runpod
import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from io import BytesIO
import base64

# --- Load model at startup (cold start optimization) ---
pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16
).to("cuda")

# --- Helper functions ---
def decode_base64_image(b64_str):
    return Image.open(BytesIO(base64.b64decode(b64_str)))

def encode_base64_image(image):
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

# --- RunPod handler function ---
def handler(job):
    job_input = job["input"]
    prompt = job_input.get("prompt")
    image_b64 = job_input.get("image")
    mask_b64 = job_input.get("mask")

    if not prompt or not image_b64 or not mask_b64:
        return {"error": "Missing input: 'prompt', 'image', or 'mask'."}

    try:
        image = decode_base64_image(image_b64).convert("RGB")
        mask = decode_base64_image(mask_b64).convert("RGB")
        result_image = pipe(prompt=prompt, image=image, mask_image=mask).images[0]
        return {"output_image": encode_base64_image(result_image)}
    except Exception as e:
        return {"error": str(e)}

# --- Start RunPod Serverless ---
runpod.serverless.start({"handler": handler})
