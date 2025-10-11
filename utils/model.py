import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from segment_anything import sam_model_registry
# from segment_anything_2.model_registry import sam2_model_registry
from .sam_lora import LoRA_Sam

class Model(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.image_embeddings = None

    def get_checkpoint(self, model_type):
        if model_type == "vit_b":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_b_01ec64.pth")
        elif model_type == "vit_l":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_l_0b3195.pth")
        elif model_type == "vit_h":
            checkpoint = os.path.join(self.cfg.model.checkpoint, "sam_vit_h_4b8939.pth")

        return checkpoint

    def setup(self):
        if self.cfg.model.type in ['vit_b','vit_l','vit_h']:
            checkpoint = self.get_checkpoint(self.cfg.model.type)
            self.model = sam_model_registry[self.cfg.model.type](checkpoint=checkpoint)
            self.base = 'sam'
        elif self.cfg.model.type in ["hiera_b",'hiera_l']:
            self.model = sam2_model_registry[self.cfg.model.type]()
            self.base = 'sam2'
        else:
            raise ValueError("Model type error!")
            
        # for param in self.model.parameters():
        #     param.requires_grad = False  
        if self.cfg.model.freeze.image_encoder:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
        if self.cfg.model.freeze.prompt_encoder:
            try:
                for param in self.model.prompt_encoder.parameters():
                    param.requires_grad = False
            except:
                for param in self.model.sam_prompt_encoder.parameters():
                    param.requires_grad = False

        if self.cfg.model.freeze.mask_decoder:
            try:
                for param in self.model.mask_decoder.parameters():
                    param.requires_grad = False
            except:
                for param in self.model.sam_prompt_encoder.parameters():
                    param.requires_grad = False
        
        self.model.train()
        self.finetune()

    def finetune(self):
        LoRA_Sam(self.model, self.cfg.lora_rank, lora_layer=list(range(self.cfg.start_lora_layer, len(self.model.image_encoder.blocks))))
        # self.set_adapter_layer()
        # self.set_norm_layer()
        # print(self.model)
        
    def set_norm_layer(self):
        for name, param in self.model.image_encoder.named_parameters():
            if "norm" in name:
                param.requires_grad = True

    def set_adapter_layer(self):
        for block in self.model.image_encoder.blocks:
            if hasattr(block, "Space_Adapter"):
                for param in block.Space_Adapter.parameters():
                    param.requires_grad = True
            if hasattr(block, "MLP_Adapter"):
                for param in block.MLP_Adapter.parameters():
                    param.requires_grad = True

    def reset_parameters(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                if "linear_a" in name:
                    nn.init.kaiming_uniform_(param, a=math.sqrt(5))
                if "linear_b" in name:
                    nn.init.zeros_(param)

    def forward(self, images, prompts):
        _, _, H, W = images.shape#[n, 3, 1024, 1024]

        image_embeddings = self.model.image_encoder(images)
        pred_masks, ious, res_masks = self.decode((H, W), prompts, image_embeddings)
        return image_embeddings, pred_masks, ious, res_masks

    # def encode(self, images):
    #     self.image_embeddings = self.model.image_encoder(images)
    #     return self.image_embeddings 

    def decode(self, image_shape, prompts, image_embeddings):
        if self.base == 'sam2':
            _bb_feat_sizes = [
                (256, 256),
                (128, 128),
                (64, 64),
            ]
            _, vision_feats, _, _ = self.model._prepare_backbone_features(image_embeddings)

            feats = [feat.permute(1, 2, 0).view(1, -1, *feat_size)
                    for feat, feat_size in zip(vision_feats[::-1], _bb_feat_sizes[::-1])][::-1] 
            self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]}
            image_embeddings = feats[-1]
            high_res_features = feats[:-1]
        
        if image_embeddings == None:
            raise "No image embeddings"

        pred_masks = []
        ious = []
        res_masks = []
        for prompt, embedding in zip(prompts, image_embeddings):

            if self.base =="sam":
                if isinstance(prompt, torch.Tensor):
                    prompt = prompt.to(device=embedding.device)
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=None,
                )
                elif isinstance(prompt, tuple):
                    sparse_embeddings, dense_embeddings = self.model.prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=None,
                )
                low_res_masks, iou_predictions = self.model.mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.model.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                )
            else:
                if isinstance(prompt, torch.Tensor):
                    prompt = prompt.to(device=embedding.device)
                    sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                    points=None,
                    boxes=prompt,
                    masks=None,
                )
                elif isinstance(prompt, tuple):
                    sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
                    points=prompt,
                    boxes=None,
                    masks=None,
                )
                low_res_masks, iou_predictions,_,_ = self.model.sam_mask_decoder(
                    image_embeddings=embedding.unsqueeze(0),
                    image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image = True, 
                    high_res_features=high_res_features,
                )

            masks = F.interpolate(
                low_res_masks,
                image_shape,
                mode="bilinear",
                align_corners=False,
            )
            pred_masks.append(masks.squeeze(1))
            ious.append(iou_predictions)
            res_masks.append(low_res_masks)
        return pred_masks, ious, res_masks