import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import string
import random
from diffusers import StableDiffusionPipeline
from transformers import pipeline

import torch
from RealESRGAN import RealESRGAN

# * GPU checks
# import tensorflow as tf
# print(tf.test.is_gpu_available())
# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# * step 1) use text-generator model to create prompt
def generate_prompt():
    generator = pipeline('text-generation', model="Gustavosta/MagicPrompt-Stable-Diffusion")
    prompt_seeds = ['highly detailed matte oil painting of english countryside, heavenly, vivid, at', 'highly detailed matte oil painting of english forest, heavenly, vivid, at', 'highly detailed matte oil painting of elegant english countryside estate, heavenly, vivid', 'beautiful architecture in']
    prompt = generator(random.choice(prompt_seeds), max_length=200, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"ℹ️ prompt: {prompt}")
    return prompt

# * step 2) feed prompt into image-to-text model
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    num_inference_steps = 100 # default = 50, sweetspot = 75, max = 100
    guidance_scale = 7.5 # default = 7.5
    image_height = 768 # 1024
    image_width = 512 # 768
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    print('generating image... ⏳')
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    print('done ✅')

    def upscale_image(image):
        device = torch.device('cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights(f"weights/RealESRGAN_x4.pth", download=True)
        print('upscaling image x4... ⏳')
        upscaled_image = model.predict(image)
        print('done ✅')
        return upscaled_image

    image = upscale_image(image)
    image_path = f"./images/{image_name}.png"
    image.save(image_path)
    print(f"image saved ✅") 

for _ in range(1):
    image_prompt = generate_prompt()
    generate_image(image_prompt)