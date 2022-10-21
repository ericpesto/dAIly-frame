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
    prompt_seeds=['amazing architecture in a', 'beautiful landscape of', 'an inspirational view of nature', 'impressive complex natural structures in the', 'impossible view of nature', 'grand buildings set in a', 'massive byzantine church in a', 'huge complex buildings from a', 'colourful natural scene set in a', 'beautiful natural geometry in a forest', 'a mysterious forest with huge trees', 'a magical wild forest with', 'beautiful view of the sky', 'inside a huge complex building', 'dream like view of a', 'psychedelic wonderland in the forest', 'huge plane in the sky', 'giant cave with a forest', 'mega architecture in the sky', 'inspirational architecture in a', 'beautiful spiritual realm', 'stagering mountains in the', 'impressive trees', 'snow tundra', 'the sun shining', 'geometric clouds', 'lush clouds', 'green grass lawns in the suburbs', 'imaginary forest', 'cliffs and the sea', 'huge cave opening to a city', 'big skies in heaven', 'the sea with waves', 'peaceful lake ripples', 'vast lake scotland', 'forest in a anlen planet', 'peaceful pond in a garden with', 'botanical garden with huge greenhouse in a', 'a magical garden with lots of flowers', 'beautiful ancient city with trees', 'huge cathedral in a', 'latge mosque in a', 'lovely little village peaceful beautiful', 'fire campsite forest pine trees', 'beautiful reflective water lake in a park', 'golden summer in a beautful park', 'winter wonderland in the', 'october in a west sussex forest', 'huge beautiful complex water fountain', 'huge waterfall in the', 'big river floating on clouds', 'valley of mirrors river of wind', 'beautiful country side with animals in the', 'a huge castle in scotland', 'huge clouds in  heaven']
    prompt = generator(random.choice(prompt_seeds), max_length=90, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"ℹ️ prompt: {prompt}")
    return prompt

# * step 2) feed prompt into image-to-text model
def generate_image(prompt):
    # 'CompVis/stable-diffusion-v1-4'
    # 'johnslegers/stable-diffusion-v1-5'
    model_id = "CompVis/stable-diffusion-v1-4"
    num_inference_steps = 50 # default = 50
    guidance_scale = 7.5 # default = 7.5
    image_height = 512 # 768 # 1024
    image_width = 512 # 512 # 768
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    image_path = f"./images/{image_name}_1x.png"
    image.save(image_path)

    # * upscale image
    def upscale_image(image_name, image_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = RealESRGAN(device, scale=4)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
        image = Image.open(image_path).convert('RGB')
        print('upscaling image x4...')
        upscaled_image = model.predict(image)
        print('✅')
        upscaled_image_path = f"./images/{image_name}_4x.png"
        upscaled_image.save(upscaled_image_path) 

    upscale_image(image_name, image_path)


image_prompt = generate_prompt()
image = generate_image(image_prompt)