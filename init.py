import os
# os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import string
import random
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed
# from super_image import MdsrModel, ImageLoader
# from PIL import Image

# * step 1) use text-generator model to create prompt
def generate_prompt():
    generator = pipeline('text-generation', model="Gustavosta/MagicPrompt-Stable-Diffusion")
    # set_seed(42)
    # create a few different prompt seeds and then serve a random one to the prompt-generator
    prompt_seeds=['amazing architecture in a', 'beautiful landscape of', 'a transcendent image of']
    prompt = generator(random.choice(prompt_seeds), max_length=90, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"ℹ️ prompt: {prompt}")
    return prompt

# * step 2) feed prompt into image-to-text model
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    num_inference_steps = 75 # default = 50
    guidance_scale = 7.5 # default = 7.5
    image_height = 1024 # 512 # 1024
    image_width = 768 # 256 # 768
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    image = pipe(prompt, guidance_scale=guidance_scale, height=image_height, width=image_width, num_inference_steps=num_inference_steps).images[0]
    image.save(f"./images/{image_name}.png")
    print(image)
    # return image

# * step 2) upscale image
# todo https://huggingface.co/sberbank-ai/Real-ESRGAN
# def upscale_image(image):
#     image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))
#     image_to_scale = Image.open(image)

#     model = MdsrModel.from_pretrained('eugenesiow/mdsr', scale=2)      # scale 2, 3 and 4 models available
#     inputs = ImageLoader.load_image(image_to_scale)
#     preds = model(inputs)

#     ImageLoader.save_image(preds, f"./images/{image_name}_scaled_2x.png")                        # save the output 2x scaled image to `./scaled_2x.png`
#     ImageLoader.save_compare(inputs, preds, f"./images/{image_name}_scaled_2x_compare.png") 

image_prompt = generate_prompt()
image = generate_image(image_prompt)
# upscale_image(image)