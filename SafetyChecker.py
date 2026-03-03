import numpy as np
import torch
import torch.nn as nn
from transformers import CLIPConfig, CLIPVisionModel, PreTrainedModel
from packaging import version
import transformers

def check_transformers_version(target: str):
    """
    简化官方代码 is_transformers_version(">", "4.57.3")
    对比当前 transformers 版本。
    """
    current_version = version.parse(transformers.__version__)
    target_version = version.parse(target)

    return current_version > target_version


class StableDiffusionSafetyChecker(PreTrainedModel):
    config_class = CLIPConfig
    main_input_name = "clip_input"

    _no_split_modules = ["CLIPEncoderLayer"]

    def __init__(self, config: CLIPConfig):
        super().__init__(config)

        self.vision_model = CLIPVisionModel(config.vision_config)
        self.visual_projection = nn.Linear(config.vision_config.hidden_size, config.projection_dim, bias=False)

        self.concept_embeds = nn.Parameter(torch.ones(17, config.projection_dim), requires_grad=False)
        self.special_care_embeds = nn.Parameter(torch.ones(3, config.projection_dim), requires_grad=False)

        self.concept_embeds_weights = nn.Parameter(torch.ones(17), requires_grad=False)
        self.special_care_embeds_weights = nn.Parameter(torch.ones(3), requires_grad=False)
        # Model requires post_init after transformers v4.57.3
        if check_transformers_version("4.57.3"):
            self.post_init()


    def forward(self, clip_input):
        pooled_output = self.vision_model(clip_input)[1]  # pooled_output
        image_embeds = self.visual_projection(pooled_output)

        return image_embeds
