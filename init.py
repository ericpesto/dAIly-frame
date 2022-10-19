import string
import random
from diffusers import StableDiffusionPipeline
from transformers import pipeline, set_seed

# * step 1) use text-generator model to create prompt
# use a different model, need better prompts
def generate_prompt(prompt_seed):
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    prompt = generator(prompt_seed, max_length=30, num_return_sequences=1)
    prompt = prompt[0]['generated_text']
    print(f"ℹ️ prompt: {prompt}")
    return prompt

# * step 2) feed prompt into image-to-text model
# text-to-image
def generate_image(prompt):
    model_id = "CompVis/stable-diffusion-v1-4"
    image_height=512
    image_width=512
    image_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=5))

    pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=True)
    pipe.safety_checker = lambda images, **kwargs: (images, False)
    image_prompt = prompt
    image = pipe(image_prompt, guidance_scale=7.5, height=image_height, width=image_width).images[0]
    image.save(f"{image_name}.png")

image_prompt = generate_prompt('A beagle wearing a cowboy hat')
generate_image(image_prompt)