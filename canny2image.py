import random
from functools import partial

import cv2
import einops
import numpy as np
import torch
from pytorch_lightning import seed_everything
from glob import glob
from torch.utils.data import DataLoader, Subset
from torchvision.utils import save_image, make_grid
import os
import shutil

import config
from annotator.canny import CannyDetector
from bdd100k import BDD100K
from cldm.ddim_hacked import DDIMSampler
from cldm.model import create_model, load_state_dict
from share import *
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--image_grid", default=False)
    args = parser.parse_args()
    print(args)

    apply_canny = CannyDetector()
    ckpt = glob("experiments/day_and_night/*.ckpt")[0]
    input_img = "../BDD_processed/train/bdd100k/images/100k/train/0000f77c-6257be58.jpg"
    prompt = "street at daytime"

    model = create_model("./models/cldm_v21.yaml").cpu()
    model.load_state_dict(load_state_dict(ckpt, location="cuda"))
    model = model.cuda()
    ddim_sampler = DDIMSampler(model)

    renorm = lambda x: (x + 1) / 2
    grid = partial(make_grid, nrow=8)

    dataset = Subset(BDD100K(mode="val"), range(args.num_samples))
    loader = DataLoader(dataset, batch_size=args.batch_size)

    # Set up output folders
    OUT_DIR = "experiments/day_and_night"
    shutil.rmtree(os.path.join(OUT_DIR, "output/realA"), ignore_errors=True)
    os.makedirs(os.path.join(OUT_DIR, "output/realA"))
    shutil.rmtree(os.path.join(OUT_DIR, "output/fakeB"), ignore_errors=True)
    os.makedirs(os.path.join(OUT_DIR, "output/fakeB"))

    for i, batch in enumerate(loader):
        batch["txt"] = np.repeat(["street at night"], len(batch["txt"])).tolist()
        log = model.log_images(batch, N=len(batch["txt"]), ddim_steps=20, ddim_eta=0)
        input_imgs = renorm(log["reconstruction"]).detach().cpu()
        canny_imgs = log["control"].detach().cpu()
        translated_imgs = renorm(log["samples_cfg_scale_9.00"]).detach().cpu()

        if args.image_grid:
            save_image(
                input_imgs,
                os.path.join(OUT_DIR, "output/realA/%04d.png" % (i + 1)),
            )
            save_image(
                translated_imgs,
                os.path.join(OUT_DIR, "output/fakeB/%04d.png" % (i + 1)),
            )
        else:
            for j in range(len(input_imgs)):
                save_image(
                    input_imgs[j],
                    os.path.join(
                        OUT_DIR, "output/realA/%04d.png" % (args.batch_size * i + j + 1)
                    ),
                )
                save_image(
                    translated_imgs[j],
                    os.path.join(
                        OUT_DIR, "output/fakeB/%04d.png" % (args.batch_size * i + j + 1)
                    ),
                )

    exit()

    input_image = cv2.imread(input_img)
    print(input_image.shape)

    a_prompt = "best quality, extremely detailed"
    n_prompt = "longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
    eta = 0.0
    scale = 9.0
    low_threshold = 100
    high_threshold = 200
    strength = 1.0
    guess_mode = False
    image_resolution = 512
    num_samples = 1
    ddim_steps = 20
    seed = -1
    output_image, canny = process(
        input_image,
        prompt,
        a_prompt,
        n_prompt,
        num_samples,
        image_resolution,
        ddim_steps,
        guess_mode,
        strength,
        scale,
        seed,
        eta,
        low_threshold,
        high_threshold,
    )
    cv2.imwrite("experiments/day_and_night/input.jpg", input_image)
    print(output_image[0].shape)
    cv2.imwrite("experiments/day_and_night/output.jpg", output_image[0])
    cv2.imwrite("experiments/day_and_night/canny.jpg", canny)
