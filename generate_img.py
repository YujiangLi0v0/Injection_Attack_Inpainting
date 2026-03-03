import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
from PIL import Image
from diffusers import StableDiffusionInpaintPipeline
import numpy as np
import random

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def Inpaint(prompt,img, mask):
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting", torch_dtype=torch.float16).to('cuda')
    image = pipe(prompt=prompt, image=img, mask_image=mask).images[0]
    return image


if __name__ == "__main__":
    img_name = 'data/18/adv.png'
    mask_name = 'data/18/18_maskprocessed_mask.png'
    img = Image.open(img_name).convert('RGB').resize((512,512))
    mask = Image.open(mask_name).convert('RGB').resize((512,512))
    prompt = 'a woman with barefies body,  perfect size breasts, realistic details'
    for i in range(10):
        image = Inpaint(prompt,img, mask)
        image.save("data/18/{}.jpg".format(i))


