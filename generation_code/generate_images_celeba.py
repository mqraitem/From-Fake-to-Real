import os 
from diffusers import StableDiffusionPipeline
import torch
import random
import argparse
from PIL import ImageStat
import numpy as np 
import random 

def is_grayscale(im):
    stat = ImageStat.Stat(im)
    print(np.abs(sum(stat.sum)/3 - stat.sum[0]))
    if np.abs(sum(stat.sum)/3 - stat.sum[0]) < 1500000 : #check the avg with any element value
        return True #if grayscale
    else:
        return False #else its colour

parser = argparse.ArgumentParser()
parser.add_argument(
    "--gender",
    type=str,
    default="Male",
)
parser.add_argument(
    "--attr",
    type=str,
    default="Smiling",
)
parser.add_argument(
    "--class_idx",
    type=int,
    default=0,
)
parser.add_argument(
    "--idx",
    type=int,
    default=0,
)

args, _ = parser.parse_known_args()


target_negative_dict = {    
    "Smiling": [["Smiling"], ["with serious face", "with straight face"]],
}

num_images_max = 10000
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")

dir_path = f"generated_images/{args.attr}/{args.class_idx}/{args.gender}"
os.makedirs(dir_path, exist_ok=True) 

count = args.idx*num_images_max
num_images_max = count + num_images_max

num_images = 10
while count < num_images_max: 

    prompts = [] 
    for _ in range(num_images):
        if args.class_idx == 1:
            prompt = f"a headshot of {args.gender} celebrity {random.choice(target_negative_dict[args.attr][0])}"
        else: 
            prompt = f"a headshot of {args.gender} celebrity {random.choice(target_negative_dict[args.attr][1])}"

        prompts.append(prompt)

    print(prompts)
    generator = [torch.Generator(device="cuda").manual_seed(random.randint(0, 100000000) ) for _ in range(num_images)]
    images = pipe(prompts, num_inference_steps=25, guidance_scale=7.5, generator=generator).images

    for image in images: 

        if is_grayscale(image):
            continue

        print(f'SAVED: {count}')

        image.save(f"{dir_path}/{count}.png")
        count += 1

