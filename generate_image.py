# FOR MACBOOK PRO 2019

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import string
import random
import torch
from diffusers import StableDiffusionPipeline
from transformers import pipeline
from RealESRGAN import RealESRGAN

def generate_prompt():
    generator = pipeline('text-generation', model="mrm8488/bloom-560m-finetuned-sd-prompts")
    prompt_seeds = ['inspiring domes in a heavenly green forest in the'] # , 'highly detailed vivid view of inspiring temples in a heavenly green jungle', 'highly detailed matte view of inspiring victorian greenhouse in a heavenly green park']
    print('generating prompt... ⏳')
    prompt = generator(random.choice(prompt_seeds), max_length=200, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"📣 prompt: {prompt}")
    return prompt

def generate_image(prompt):
    model_id = "stabilityai/stable-diffusion-2" # "CompVis/stable-diffusion-v1-4"
    num_inference_steps = 50
    guidance_scale = 7.5 
    image_height = 768
    image_width = 512

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    # pipe.safety_checker = lambda images, **kwargs: (images, False)
    print('generating image... ⏳')
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    print(f"done ✅")
    return image

def upscale_image(image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    model.load_weights(f"weights/RealESRGAN_x4.pth", download=True)
    print('upscaling image... ⏳')
    upscaled_image = model.predict(image)
    print('done ✅')
    return upscaled_image

def save_image(image):
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
    image_path = f"./images/{image_name}.png"
    print('saving image... ⏳')
    image.save(image_path)
    print(f"{image_name}.png saved 💾")

def create_batch_of_images(quantity):
  for i in range(quantity):
    print(f"🖼 generating image: {i+1}/{quantity}")
    image_prompt = generate_prompt()
    image = generate_image(image_prompt)
    upscaled_image = upscale_image(image)
    save_image(upscaled_image)
    print("")  

create_batch_of_images(7)