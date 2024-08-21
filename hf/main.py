import cv2
import numpy as np
import torch
from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                       UniPCMultistepScheduler)
from diffusers.utils import load_image, make_image_grid
from PIL import Image


def to_canny(original_image, low_threshold=100, high_threshold=200):
    image = np.array(original_image)
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image


def run_inference(pipe, frame_no, prompt="night"):
    original_image = load_image(
        f"/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref/GOPR0356_frame_000{frame_no}_ref_rgb_anon.png"
    )
    canny_image = to_canny(original_image)
    gt = load_image(
        f"/data/matthew/dark_zurich/val/rgb_anon/val/night/GOPR0356/GOPR0356_frame_000{frame_no}_rgb_anon.png"
    )
    output = pipe(prompt, image=canny_image).images[0]
    out_image = make_image_grid(
        [original_image, canny_image, output, gt], rows=1, cols=4
    )
    out_image.save(f"out_{frame_no}.png")


if __name__ == "__main__":
    controlnet = ControlNetModel.from_pretrained(
        "lllyasviel/sd-controlnet-canny",
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        controlnet=controlnet,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_model_cpu_offload()
    prompt = "night"
    run_inference(pipe, 321, prompt)
    run_inference(pipe, 414, prompt)
    run_inference(pipe, 488, prompt)

    # original_image = load_image(
    #     # "/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref/GOPR0356_frame_000321_ref_rgb_anon.png"
    #     # "/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref/GOPR0356_frame_000414_ref_rgb_anon.png"
    #     # "/data/matthew/dark_zurich/val/rgb_anon/val_ref/day/GOPR0356_ref/GOPR0356_frame_000488_ref_rgb_anon.png"
    #     # "https://hf.co/datasets/huggingface/documentation-images/resolve/main/diffusers/input_image_vermeer.png"
    # )
