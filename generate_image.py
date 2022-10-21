import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import string
import random
from diffusers import StableDiffusionPipeline
from transformers import pipeline

import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

import tensorflow as tf
# print(tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# * step 1) use text-generator model to create prompt
def generate_prompt():
    generator = pipeline('text-generation', model="Gustavosta/MagicPrompt-Stable-Diffusion")
    prompt_seeds=['amazing complex huge green space architecture in', 'elegant grand building set in a green space nature', 'complex green space byzantine church in the', 'elegant ancient architecture in the jungle', 'elegant future architecture in the jungle', 'elegant botanical garden greenhouse architecture in the', 'the heavenly english countryside at', 'huge complex elegant scifi architecture in the forest', 'huge complex elegant scifi architecture in the jungle', ' grand elegant countryside mansion in heavenly']
    prompt = generator(random.choice(prompt_seeds), max_length=90, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"ℹ️ prompt: {prompt}")
    return prompt

# * step 2) feed prompt into image-to-text model
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    num_inference_steps = 100 # default = 50, sweetspot = 100
    guidance_scale = 7.5 # default = 7.5
    image_height = 768 # 768 # 1024
    image_width = 512 # 512 # 768
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    image_path = f"./images/{image_name}_1x.png"
    image.save(image_path)
    print(f"image saved ✅")

    # * upscale image, you could chain them, i.e. upscale the upscaled image
    def upscale_image(image_name, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        image = Image.open(image_path).convert('RGB')
        print('upscaling image x4... ⏳')
        upscaled_image = model.predict(image)
        print('✅')
        upscaled_image_path = f"./images/{image_name}_4x.png"
        upscaled_image.save(upscaled_image_path)
        print(f"x4 image saved ✅")

    upscale_image(image_name, image_path)

# image_prompt = generate_prompt()
# image = generate_image(image_prompt)

for _ in range(10):
    image_prompt = generate_prompt()
    generate_image(image_prompt)