import string
import random
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed

import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

# * step 1) use text-generator model to create prompt
def generate_prompt():
    generator = pipeline('text-generation', model="Gustavosta/MagicPrompt-Stable-Diffusion")
    # set_seed(42)
    # create a few different prompt seeds and then serve a random one to the prompt-generator
    prompt_seeds=['amazing architecture in a', 'beautiful landscape of', 'an inspirational view of nature', 'impressive complex natural structures in the', 'profound view of nature', 'impossible view of nature', 'satisfying vistas of the astral dimension', 'grand buildings set in a', 'massive byzantine church in a', 'huge complex buildings from a', 'colourful natural scene set in a', 'beautiful natural geometry in a forest', 'a mysterious forest with huge trees', 'a magical wild forest with', 'beautiful view of the sky', 'heavenly clouds', 'the radiant sun', 'healing view of nature']
    prompt = generator(random.choice(prompt_seeds), max_length=90, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"ℹ️ prompt: {prompt}")
    return prompt

# * step 2) feed prompt into image-to-text model
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    num_inference_steps = 50 # default = 50
    guidance_scale = 7.5 # default = 7.5
    image_height = 768 # 512 # 1024
    image_width = 512 # 256 # 768
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    image_path = f"./images/{image_name}_1x.png"
    image.save(image_path)

    # * upscale image
    def upscale_image(image_name, image_path):
        model = RealESRGAN(torch.device('cpu'), scale=8)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        image = Image.open(image_path).convert('RGB')
        upscaled_image = model.predict(image)
        upscaled_image_path = f"./images/{image_name}_4x.png"
        upscaled_image.save(upscaled_image_path) 

    upscale_image(image_name, image_path)


image_prompt = generate_prompt()
image = generate_image(image_prompt)