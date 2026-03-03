import os
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


from PIL import Image
from transformers import CLIPImageProcessor
import torch
from SafetyChecker import StableDiffusionSafetyChecker
from diffusers import StableDiffusionInpaintPipeline

def get_embeds_from_inpainting_model(image_path, inpainting_model_id):
 
    # 步骤 A: 加载 Inpainting 管线，目的是为了获取它的 Safety Checker 权重和预处理器
    print(f"正在加载 Inpainting 模型: {inpainting_model_id} ...")
    pipe = StableDiffusionInpaintPipeline.from_pretrained("stable-diffusion-v1-5/stable-diffusion-inpainting", dtype=torch.float16).to('cuda')

    # 步骤 B: 实例化你自己的简化版模型
    print("正在初始化简化版 Safety Checker 并加载权重...")
    # 从官方管线里把 Config 拿出来
    config = pipe.safety_checker.config
    # 用这个 Config 初始化你的类
    my_model = StableDiffusionSafetyChecker(config)
    
    # 【关键】把官方管线里的权重加载到你的模型里
    # 因为你的类定义的层名字 (vision_model, visual_projection) 和官方完全一样，所以可以直接 load
    my_model.load_state_dict(pipe.safety_checker.state_dict())
    
    # 切换到评估模式
    my_model.eval()

    # 步骤 C: 准备图片处理器 (CLIPImageProcessor)
    # 直接复用管线里的 feature_extractor，保证预处理逻辑一致
    processor = pipe.feature_extractor

    # 步骤 D: 加载图片并预处理
    print(f"正在处理图片: {image_path} ...")
    image = Image.open(image_path).convert("RGB")
    
    # 这就生成了你要的 clip_input
    inputs = processor(images=image, return_tensors="pt")
    clip_input = inputs.pixel_values

    # 步骤 E: 推理获取 embeddings
    print("正在推理获取 Image Embeds ...")
    with torch.no_grad():
        image_embeds = my_model(clip_input)

    return image_embeds


# ==========================================
# 3. 执行示例
# ==========================================
if __name__ == "__main__":
    # 配置你的路径
    MY_INPAINTING_MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting" # 或者是你本地的文件夹路径
    MY_IMAGE = "data/18.png" # 替换为你要测试的图片

    # 运行
    embeds = get_embeds_from_inpainting_model(MY_IMAGE, MY_INPAINTING_MODEL)

    print("\n✅ 成功!")
    print(f"Image Embeds 形状: {embeds.shape}")
    print(f"Image Embeds 内容 (前5个值): {embeds[0, :5]}")