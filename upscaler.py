import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN

#run it through twice, so 8x
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