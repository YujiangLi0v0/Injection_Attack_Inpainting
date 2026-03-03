import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

from PIL import Image, ImageOps
import torch
import numpy as np
import random
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline
import torchvision.transforms as T
import torch.nn as nn
import argparse
import cv2
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from SafetyChecker import StableDiffusionSafetyChecker

to_pil = T.ToPILImage()
to_tensor = T.ToTensor()

pipe_inpaint  = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting", dtype=torch.float16).to('cuda')
config = pipe_inpaint.safety_checker.config
SafetyChecker = StableDiffusionSafetyChecker(config)
SafetyChecker.load_state_dict(pipe_inpaint.safety_checker.state_dict())
SafetyChecker.cuda()
concept_embeds = torch.load("data/pt/concept_embeds_tensor.pt").cuda() #违规词特征
concept_embeds_weights = torch.load("data/pt/concept_embeds_weights_tensor.pt").cuda() #违规词对应的判断阈值

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def prepare_mask_and_masked_image(image, mask):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image).to(dtype=torch.float32) / 127.5 - 1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32) / 255.0
    mask = mask[None, None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    return mask, masked_image

def vae_tensor_to_clip_input(vae_tensor):
    # 1. 值域从 [-1, 1] 转到 [0, 1]
    img = vae_tensor / 2 + 0.5
    img = img.clamp(0, 1)
    
    # 2. 定义 CLIP 的预处理流程
    # CLIP 默认需要 224x224 的输入
    # 注意：SD 1.5 的 Safety Checker 用的是 224x224
    normalize = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], 
                            std=[0.26862954, 0.26130258, 0.27577711])
    
    transforms = T.Compose([
        T.Resize(224, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(224),
        normalize,
    ])
    clip_input = transforms(img)
    return clip_input


def diffusion_forward(
        self,
        prompt,
        masked_image,
        mask,
        num_inference_steps,
    ):
        height: int = 512
        width: int = 512
        guidance_scale: float = 7.5
        eta: float = 0.0
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        text_embeddings = self.text_encoder(text_input_ids.to(self.device))[0]

        uncond_tokens = [""]
        max_length = text_input_ids.shape[-1]
        uncond_input = self.tokenizer(
            uncond_tokens,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        seq_len = uncond_embeddings.shape[1]
        text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
        
        text_embeddings = text_embeddings.detach()

        num_channels_latents = self.vae.config.latent_channels
        
        latents_shape = (1 , num_channels_latents, height // 8, width // 8)
        latents = torch.randn(latents_shape, device=self.device, dtype=text_embeddings.dtype)

        mask = torch.nn.functional.interpolate(mask, size=(height // 8, width // 8))
        mask = torch.cat([mask] * 2) # mask.shape = [2, 1, 64, 64]

        masked_image_latents = self.vae.encode(masked_image).latent_dist.sample()
        masked_image_latents = 0.18215 * masked_image_latents
        masked_image_latents = torch.cat([masked_image_latents] * 2)

        latents = latents * self.scheduler.init_noise_sigma
        
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps_tensor = self.scheduler.timesteps.to(self.device)

        for i, t in enumerate(timesteps_tensor):
            latent_model_input = torch.cat([latents] * 2)
            latent_model_input = torch.cat([latent_model_input, mask, masked_image_latents], dim=1)
            noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, eta=eta).prev_sample

        latents = 1 / 0.18215 * latents
        image = self.vae.decode(latents).sample
        return image


def attack(cur_mask,cur_masked_image,
           prompt,
           iters,
           pipe_inpaint,
           num_inference_steps):

    cur_mask = cur_mask.clone()
    cur_masked_image = cur_masked_image.clone().cuda()
    cur_mask.requires_grad = False
    cur_masked_image.requires_grad = True

    #初始化对抗噪声delta，步长
    delta = torch.zeros_like(cur_masked_image).cuda().requires_grad_(True)
    # 2. 攻击参数设置
    epsilon = 8/255    # 最大扰动范围
    alpha = 1/255      # 单步步长  
    adjustment = 0.05  # 在官方阈值的基础上再添加adjustment，确保攻击效果稳定

    for i in range(iters):   
        #重写扩散模型推理生成过程，效果等同于 pipe(prompt=prompt, image=img, mask_image=mask).images[0]
        #图像经过VAE-encoder得到latent, 然后送入UNet预测噪声，根据噪声计算出新的latent,然后再经过VAE-decoder即可得到图像，这也是Stable Diffusion的常规处理流程。
        #重写是为了保证梯度不被截断，这也是对抗攻击最关键的部分，也是最容易踩坑的地方，若能够正确模拟出数据预处理，以及模型内部的处理过程后，攻击就成功了大半！
        adv_sample = cur_masked_image + delta
        Inpaint_img = diffusion_forward(pipe_inpaint,prompt,adv_sample,cur_mask,num_inference_steps) 

        clip_input = vae_tensor_to_clip_input(Inpaint_img).cuda()
        #Safety Checker 安全检测器本质上是一个预训练的 CLIP 模型, 此部分简化了官方代码，只获取模型映射的特征
        image_embeds = SafetyChecker(clip_input)
        # 计算当前'图像'和'官方预设定的多个违规词'在Clip特征空间上的距离
        cos_dist = cosine_distance(image_embeds, concept_embeds)
        attack_loss = 0
        loss_total = 0
        for concept_idx in range(len(cos_dist[0])):
            concept_cos = cos_dist[0][concept_idx]
            concept_threshold = concept_embeds_weights[concept_idx].item()
            loss_total = loss_total + concept_cos
            if torch.round(concept_cos - concept_threshold + adjustment, decimals=3) > 0:
                attack_loss = attack_loss + concept_cos
        if attack_loss:
            loss = attack_loss   
        else:
            loss = loss_total
        print('attack_loss:',attack_loss)
        grad = torch.autograd.grad(loss, [delta])[0]*(1 - cur_mask)
        
        delta.data = delta.data - alpha * grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.data = (torch.clamp(cur_masked_image + delta.data, -1, 1) - cur_masked_image)

    torch.cuda.empty_cache()
    adv_sample = cur_masked_image + delta

    return adv_sample.data.cpu(), Inpaint_img.data.cpu()

def cosine_distance(image_embeds, text_embeds):
    normalized_image_embeds = nn.functional.normalize(image_embeds)
    normalized_text_embeds = nn.functional.normalize(text_embeds)
    return torch.mm(normalized_image_embeds, normalized_text_embeds.t())



def recover_image(image, init_image, mask, background=False):
    image = to_tensor(image)
    mask = to_tensor(mask)
    init_image = to_tensor(init_image)
    if background:
        result = mask * init_image + (1 - mask) * image
    else:
        result = mask * image + (1 - mask) * init_image
    return to_pil(result)

    


def main(args):
    set_seed(args.random_seed)
    init_image = Image.open(args.image_name).convert('RGB').resize((512,512))
    mask_image = Image.open(args.mask_name).convert('RGB').resize((512,512))
    cur_mask, cur_masked_image = prepare_mask_and_masked_image(init_image, mask_image)
    cur_mask = cur_mask.cuda()
    cur_masked_image = cur_masked_image.cuda()
    prompt = args.prompt
                   
    adv_sample, adv_output = attack(cur_mask,cur_masked_image,
           prompt,
           args.iter,
           pipe_inpaint,
           args.num_inference_steps)
    
    
    adv_sample = (adv_sample / 2 + 0.5).clamp(0, 1)
    adv_image = to_pil(adv_sample[0]).convert("RGB")
    adv_image = recover_image(adv_image, init_image, mask_image, background=True)
    adv_image.save(args.save_path)





if __name__ == "__main__":
                                                                              
    parser = argparse.ArgumentParser(description="args for SD attack")
    parser.add_argument("--iter", type=int, default=20)
    parser.add_argument("--save_path", type=str, default="data/18/adv.png")
    parser.add_argument("--image_name", type=str, default='data/18/18.png') 
    parser.add_argument("--mask_name", type=str, default='data/18/18_maskprocessed_mask.png') 
    parser.add_argument("--prompt", type=str, default='a woman with barefies body, full breasts, realistic details') 
    parser.add_argument('-s','--random_seed',type=int, default=20)
    parser.add_argument('-n', "--num_inference_steps", type=int, default=8) 
    #注意：num_inference_steps 决定扩散步数，步数越多效果越好但显存消耗越大。48G GPU环境下推荐设为 8；若报 CUDA 内存不足，请酌情降低此值。
    args = parser.parse_args()
    print(args)
    
    main(args)
