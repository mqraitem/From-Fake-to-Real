import os 
from diffusers import StableDiffusionPipeline
import torch
import random
import argparse
from PIL import ImageStat
import numpy as np 

def is_grayscale(im):
    stat = ImageStat.Stat(im)
    print(np.abs(sum(stat.sum)/3 - stat.sum[0]))
    if np.abs(sum(stat.sum)/3 - stat.sum[0]) < 200000 : #check the avg with any element value
        return True #if grayscale
    else:
        return False #else its colour

parser = argparse.ArgumentParser()
parser.add_argument(
    "--animal",
    type=str,
    default="big_dog",
)
parser.add_argument(
    "--bias",
    type=str,
    default="Indoors",
)
parser.add_argument(
    "--idx",
    type=int,
    default=0,
)

args, _ = parser.parse_known_args()

animal = args.animal
bias = args.bias

"##############################################"

#copied from https://arxiv.org/pdf/2306.11957
breeds = open("generation_code/animal_breeds.txt", "r")
breeds = breeds.readlines()
breeds_dict = {} 
for line in breeds: 
    animal = line.split(":")[0] 
    animal_breeds = line.split(":")[1].split(",")
    breeds_dict[animal] = animal_breeds

this_animal_breeds = breeds_dict[animal]

"##############################################"

locations = {
    "Indoors": ["indoors", "in the living room", "in the bedroom", "in the kitchen", "in the attic", "on the table", "under the table", "on the chair", "under the chair", "on the kitchen top", "in the sink"],
    "Outdoors": ["outdoors", "in the garden", "in the park", "in the forest", "in the field", "in the backyard", "in the front yard", "in the driveway", "on the street", "on the sidewalk"],
    "Land": ["dirt", "grass", "rock", "sand", "soil", "mud", "gravel", "asphalt", "concrete", "pavement", "brick", "wood", "leaves", "snow", "ice", "clay", "dust", "cement", "metal", "tile"],
    "Water": ["water", "lake", "river", "pond", "sea", "ocean", "pool", "fountain", "stream", "waterfall"]
}

gen_folder = "generated_images"
num_images_max = 20000

pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16).to("cuda")
dir_path = f"{gen_folder}/{animal}_{bias}"
os.makedirs(dir_path, exist_ok=True) 

count = args.idx*num_images_max
num_images_max = count + num_images_max

while count < num_images_max: 


    num_images = 10
    generator = [torch.Generator(device="cuda").manual_seed(random.randint(0, 100000000) ) for _ in range(num_images)]
    
    prompts = [] 
    animals = [] 
    for _ in range(num_images): 

        prompt = f"a photo of {random.choice(this_animal_breeds)} in {random.choice(locations[bias])}"
        print(prompt)

        prompts.append(prompt)
        animals.append(animal)
    
    images = pipe(prompts, num_inference_steps=50, guidance_scale=7.5, generator=generator).images
    
    for image, animal in zip(images, animals): 

        if is_grayscale(image):
            continue

        print(f'SAVED: {count}')

        image.save(f"{dir_path}/{animal}_{count}.png")
        count += 1

