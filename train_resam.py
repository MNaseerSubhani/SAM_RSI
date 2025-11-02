import os
import time
import argparse
import random
# from abc import ABC

import cv2
import numpy as np
import torch
# import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from matplotlib import cm

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.fabric import _FabricOptimizer

from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.losses import DiceLoss, FocalLoss, Matching_Loss
from utils.eval_utils import AverageMeter, validate, get_prompts, calc_iou, validate_sam2
from utils.tools import copy_model, create_csv, reduce_instances
from utils.utils import *

import  csv, copy
import torch
import torch.nn.functional as F
from collections import deque

# vis = False


def sort_entropy_(model, target_pts):

    # save_dir = "entropy_sorted"
    # os.makedirs(save_dir, exist_ok=True)

    collected = []
    with torch.no_grad():
        for i, batch in enumerate(tqdm(target_pts, desc='Computing per-sample entropy', ncols=100)):
            imgs, boxes, masks, img_paths = batch
            prompts = get_prompts(cfg, boxes, masks)
            embeds, masks_pred, _, _ = model(imgs, prompts)

            batch_size = imgs.shape[0]
            for b in range(batch_size):
                img_np = (imgs[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                p_b = masks_pred[b].clamp(1e-6, 1 - 1e-6)
                if p_b.ndim == 2:
                    p_b = p_b.unsqueeze(0)
                gt_b = masks[b]
                if gt_b.ndim == 2:
                    gt_b = gt_b.unsqueeze(0)

                entropy_scalar = 0
                num_inst = p_b.shape[0]
                for j in range(num_inst):
                    p_inst = p_b[j]
                    entropy_map_inst = - (p_inst * torch.log(p_inst) + (1 - p_inst) * torch.log(1 - p_inst))
                    entropy_scalar += float(entropy_map_inst.mean().cpu().item())

                entropy_scalar /= num_inst
                render = {
                    'real': img_np,
                    'prompt': prompts
                }
                img_path = img_paths[b] if isinstance(img_paths, (list, tuple)) else img_paths
                collected.append((entropy_scalar, img_path, render))

            if i>10:
                break

    collected.sort(key=lambda x: x[0], reverse=True)

    return collected
def create_entropy_mask(entropy_maps, threshold=0.5, device='cuda'):
    """
    Create a mask to reduce learning from high entropy regions.
    
    Args:
        entropy_maps: List of entropy maps for each instance
        threshold: Entropy threshold above which to mask out regions
        device: Device to place the mask on
    
    Returns:
        List of entropy masks (0 for high entropy, 1 for low entropy)
    """
    entropy_masks = []
    
    for entropy_map in entropy_maps:
        # Create binary mask: 1 for low entropy, 0 for high entropy
        entropy_mask = (entropy_map < threshold).float()
        entropy_masks.append(entropy_mask)
    
    return entropy_masks


def process_forward(img_tensor, prompt, model):
    with torch.no_grad():
        _, masks_pred, _, _ = model(img_tensor, prompt)
    entropy_maps = []
    pred_ins = []
    for i, mask_p in enumerate( masks_pred[0]):

        p = mask_p.clamp(1e-6, 1 - 1e-6)
        if p.ndim == 2:
            p = p.unsqueeze(0)

        entropy_map = entropy_map_calculate(p)
        entropy_maps.append(entropy_map)
        pred_ins.append(p)

    return entropy_maps, pred_ins
        
        
        

def edge_corner_score(x, y, x_c, y_c, w, h, gamma=0.7):
    dx = 2 * torch.abs(x - x_c) / w
    dy = 2 * torch.abs(y - y_c) / h
    dx = torch.clamp(dx, 0, 1)
    dy = torch.clamp(dy, 0, 1)
    # high on edges + corners, low at center
    score = (dx + dy - dx * dy) ** gamma
    return score

        
def entropy_map_calculate(p):
    eps = 1e-8
    p = p.clamp(eps, 1 - eps)  # Safe!
    entropy_map = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    # entropy_map = entropy_map.max(dim=0)[0]
    return entropy_map# / torch.log(torch.tensor(2.0))

def prompt_calibration(cfg, entrop_map, prompts, point_status):
    point_list = []
    point_labels_list = []
    num_points = cfg.num_points

    for m in range(len(entrop_map)):
        point_coords = prompts[0][0][m][:].unsqueeze(0)
        point_coords_lab = prompts[0][1][m][:].unsqueeze(0)

        # Find high-entropy location
        max_idx = torch.argmax(entrop_map[m])
        y = max_idx // entrop_map[m].shape[1]
        x = max_idx % entrop_map[m].shape[1]
        neg_point_coords = torch.tensor([[x.item(), y.item()]], device=point_coords.device).unsqueeze(0)


        # Combine positive and negative points
        point_coords_all = torch.cat((point_coords, neg_point_coords), dim=1)
        
        # Append a new label (1) to the label tensor
        point_labels_all = torch.cat(
            (point_coords_lab, torch.tensor([[point_status]], device=point_coords.device, dtype=point_coords_lab.dtype)),
            dim=1
        )
        
        point_list.append(point_coords_all)
        point_labels_list.append(point_labels_all)





    point_ = torch.cat(point_list).squeeze(1)
    point_labels_ = torch.cat(point_labels_list)
    new_prompts = [(point_, point_labels_)]
    return new_prompts


def get_bbox_feature(embedding_map, bbox, stride=16, pooling='avg'):
    """
    Extract a feature vector from an embedding map given a bounding box.
    
    Args:
        embedding_map (torch.Tensor): Shape (C, H_feat, W_feat) or (B, C, H_feat, W_feat)
        bbox (list or torch.Tensor): [x1, y1, x2, y2] in original image coordinates
        stride (int): Downscaling factor between image and feature map
        pooling (str): 'avg' or 'max' pooling inside the bbox region
        
    Returns:
        torch.Tensor: Feature vector of shape (C,)
    """
    # If batch dimension exists, assume batch size 1
    if embedding_map.dim() == 4:
        embedding_map = embedding_map[0]

    C, H_feat, W_feat = embedding_map.shape
    x1, y1, x2, y2 = bbox

    # Map bbox to feature map coordinates
    fx1 = max(int(x1 / stride), 0)
    fy1 = max(int(y1 / stride), 0)
    fx2 = min(int((x2 + stride - 1) / stride), W_feat)  # ceil division
    fy2 = min(int((y2 + stride - 1) / stride), H_feat)

    # Crop the feature map to bbox region
    region = embedding_map[:, fy1:fy2, fx1:fx2]

    if region.numel() == 0:
        # fallback to global feature if bbox is too small
        region = embedding_map

    # Pool to get a single feature vector
    if pooling == 'avg':
        feature_vec = region.mean(dim=(1,2))
    elif pooling == 'max':
        feature_vec = region.amax(dim=(1,2))
    else:
        raise ValueError("pooling must be 'avg' or 'max'")

    return feature_vec

import torch
import torch.nn.functional as F

def info_nce_loss(features, temperature=0.07):
    # Normalize features
    features = F.normalize(features, dim=1)

    # Cosine similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature

    # Remove self-similarity (set diagonal to -inf)
    mask = torch.eye(sim_matrix.size(0), device=sim_matrix.device).bool()
    sim_matrix.masked_fill_(mask, -float('inf'))

    # Softmax across each row, pick the max as pseudo-positive
    probs = F.softmax(sim_matrix, dim=1)
    # Encourage one feature to have one strong positive
    loss = -torch.log(probs.max(dim=1).values + 1e-8).mean()
    return loss

def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    init_iou
):

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    best_iou = init_iou
    best_state = copy.deepcopy(model.state_dict())
    no_improve_count = 0
    max_patience = cfg.get("patience", 3)  # stop if no improvement for X validations
    match_interval = cfg.match_interval
    eval_interval = int(len(train_dataloader) * 1)

    window_size = 100
    embedding_queue = []
    ite_em = 0

    # Prepare output dirs
    os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
    csv_path = os.path.join(cfg.out_dir, "training_log.csv")

    # Initialize CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Iteration", "Val_IoU", "Best_IoU", "Status"])

    fabric.print(f"Training with rollback enabled. Logging to: {csv_path}")


    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        sim_losses = AverageMeter()
        end = time.time()

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes_gt, gt_masks, img_paths= data
            del data

            slice_step = 50
            for j in range(0, len(gt_masks[0]), slice_step):
                
                gt_masks_new = gt_masks[0][j:j+slice_step].unsqueeze(0)
                prompts = get_prompts(cfg, bboxes_gt, gt_masks_new)

                batch_size = images_weak.size(0)

                entropy_maps, preds = process_forward(images_weak, prompts, model)
                entropy_maps = torch.stack(entropy_maps, dim=0)
                pred_stack = torch.stack(preds, dim=0)
                pred_binary = ((pred_stack>0.5 ) )#& (entropy_maps < 0.1)).float()

                
                # pred_stack = torch.stack(preds, dim=0)
                # entropy_maps_mask = ((entropy_maps))
                # entropy_maps_mask = (entropy_maps < 0.1)
                # pred_filt = pred_stack * entropy_maps_mask
               

                # pred_binary = (entropy_maps < 0.1).float()# (pred_stack > 0.5).float() 
               
                overlap_count = pred_binary.sum(dim=0)
                overlap_map = (overlap_count > 1).float()
                invert_overlap_map = 1.0 - overlap_map


                bboxes = []
                point_list = []
                point_labels_list = []
               
                for i,  pred in enumerate( pred_binary):
                    point_coords = prompts[0][0][i][:].unsqueeze(0)
                    point_coords_lab = prompts[0][1][i][:].unsqueeze(0)
                    # print(entropy_map.shape, pred.shape)
                    # pred = pred * entropy_map.unsqueeze(0)#(pred[0]>0.5)
                    
                    pred_w_overlap = (pred[0]) * invert_overlap_map[0]
                    


                    ys, xs = torch.where(pred_w_overlap> 0.5)
                    if len(xs) > 0 and len(ys) > 0:
                        x_min, x_max = xs.min().item(), xs.max().item()
                        y_min, y_max = ys.min().item(), ys.max().item()
                        bboxes.append(torch.tensor([x_min, y_min , x_max, y_max], dtype=torch.float32))

                        point_list.append(point_coords)
                        point_labels_list.append(point_coords_lab)
                
                if len(bboxes) == 0:
                    continue  # skip if no valid region
               
                        
                        
                        # print("No 1s found in mask")
                point_ = torch.cat(point_list).squeeze(1)
                point_labels_ = torch.cat(point_labels_list)
                new_prompts = [(point_, point_labels_)]
            
                bboxes = torch.stack(bboxes)

                with torch.no_grad():
                    embeddings, soft_masks, _, _ = model(images_weak, bboxes.unsqueeze(0))
  
                _, pred_masks, iou_predictions, _= model(images_strong, new_prompts)
                del _


              

                num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
                loss_focal = torch.tensor(0., device=fabric.device)
                loss_dice = torch.tensor(0., device=fabric.device)
                loss_iou = torch.tensor(0., device=fabric.device)
                loss_sim = torch.tensor(0., device=fabric.device)

                for i, (pred_mask, soft_mask, iou_prediction, bbox) in enumerate(
                        zip(pred_masks, soft_masks, iou_predictions, bboxes  )
                    ):  
                        embed_feats = get_bbox_feature( embeddings, bbox)
                        embed_feats = F.normalize(embed_feats, p=2, dim=0)
                        embedding_queue.append(embed_feats)
                        loss_match = 0
                        
                            
                        features = torch.stack(embedding_queue, dim=0)
                        eps = 1e-8
                        # cos_sim_matrix = F.cosine_similarity(
                        #     features.unsqueeze(1),
                        #     features.unsqueeze(0),
                        #     dim=2,
                        #     eps=eps  # prevent division by zero
                        # )
                        # num = features.size(0)
                        # # device = features.device 
                        # # mask = (1 - torch.eye(num, device=device))
                        # # cos_sim_matrix = cos_sim_matrix * mask
                        # # if mask.sum() > 0:
                        # loss_match = 1 - (cos_sim_matrix.mean() )
                        # else:
                        #     loss_match = torch.tensor(0.0, device=features.device)

                        # Rescale to [0,1]
                        # cos_sim_matrix = (cos_sim_matrix + 1) / 2

                        # Temperature
                        # tau = 0.07
                        # sim_soft = torch.exp(cos_sim_matrix / tau)
                        # prob_matrix = sim_soft / sim_soft.sum(dim=1, keepdim=True)

                      
                        # alpha = 0.7
                        # loss_global = 1 - cos_sim_matrix.mean()
                        # loss_local = ((1 - cos_sim_matrix) * prob_matrix).mean()
                        # loss_sim = alpha * loss_global + (1 - alpha) * loss_local
                        
                        # loss_sim = info_nce_loss(features)




                        # Weighted alignment loss
                        # loss_sim = ((1 - cos_sim_matrix) * prob_matrix).mean() 
                        # loss_sim = (1 - cos_sim_matrix.mean())


                        if len(embedding_queue) > 1:
                            # Stack all embeddings (num_instances, feature_dim)
                            features = torch.stack(embedding_queue, dim=0)  # [N, D]
                            eps = 1e-8

                            # Compute cosine similarity matrix
                            cos_sim_matrix = F.cosine_similarity(
                                features.unsqueeze(1),  # [N, 1, D]
                                features.unsqueeze(0),  # [1, N, D]
                                dim=2,
                                eps=eps
                            )  # shape [N, N]

                            # Remove self-similarity bias
                            num = features.size(0)
                            mask = (1 - torch.eye(num, device=features.device))
                            cos_sim_matrix = cos_sim_matrix * mask

                            # ---- Soft alignment (SSAL) ----
                            # Step 1. Rescale cosine to [0,1]
                            cos_sim_matrix = (cos_sim_matrix + 1) / 2

                            # Step 2. Compute temperature-scaled soft distribution
                            tau = 0.07  # you can tune in [0.03–0.1]
                            sim_soft = torch.exp(cos_sim_matrix / tau)
                            prob_matrix = sim_soft / (sim_soft.sum(dim=1, keepdim=True) + eps)

                            # Step 3. Soft Semantic Alignment Loss
                            loss_sim = ((1 - cos_sim_matrix) * prob_matrix).sum(dim=1).mean()

                        else:
                            loss_sim = torch.tensor(0.0, device=embeddings.device)

                        soft_mask = (soft_mask > 0.).float()
                        # Apply entropy mask to losses
                        loss_focal += focal_loss(pred_mask, soft_mask)  #, entropy_mask=entropy_mask
                        loss_dice += dice_loss(pred_mask, soft_mask)   #, entropy_mask=entropy_mask
                        batch_iou = calc_iou(pred_mask, soft_mask)
                        loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

                        if len(embedding_queue) > window_size:
                            embedding_queue.pop(0)
            
                del  pred_masks, iou_predictions 
                # loss_dist = loss_dist / num_masks
                # loss_dice = loss_dice #/ num_masks
                # loss_focal = loss_focal #/ num_masks
                # loss_sim = loss_sim


                loss_total =  20 * loss_focal +  loss_dice  + loss_iou #+ loss_sim #+ loss_iou  +  +



                fabric.backward(loss_total)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                del  prompts, soft_masks

                batch_time.update(time.time() - end)
                end = time.time()

                focal_losses.update(loss_focal.item(), batch_size)
                dice_losses.update(loss_dice.item(), batch_size)
                iou_losses.update(loss_iou.item(), batch_size)
                total_losses.update(loss_total.item(), batch_size)
                sim_losses.update(loss_sim.item(), batch_size)
            
                del loss_dice, loss_iou, loss_focal

            if (iter+1) % match_interval==0:
                fabric.print(
                    f"Epoch [{epoch}] Iter [{iter + 1}/{len(train_dataloader)}] "
                    f"| Focal {focal_losses.avg:.4f} | Dice {dice_losses.avg:.4f} | "
                    f"IoU {iou_losses.avg:.4f} | Sim_loss {sim_losses.avg:.4f} | Total {total_losses.avg:.4f}"
                )
            if (iter+1) % eval_interval == 0:
                val_iou, _ = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)

                status = ""
                if val_iou > best_iou:
                    best_iou = val_iou
                    best_state = copy.deepcopy(model.state_dict())
                    torch.save(best_state, os.path.join(cfg.out_dir, "save", "best_model.pth"))
                    status = "Improved → Model Saved"
                    no_improve_count = 0
                else:
                    model.load_state_dict(best_state)
                    no_improve_count += 1
                    status = f"Rollback ({no_improve_count})"

                # Write log entry
                with open(csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, iter + 1, val_iou, best_iou, status])

                fabric.print(f"Validation IoU={val_iou:.4f} | Best={best_iou:.4f} | {status}")

                # Stop if model fails to stabilize
                if no_improve_count >= max_patience:
                    fabric.print(f"Training stopped early after {no_improve_count} failed rollbacks.")
                    return
        





            
def configure_opt(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    # optimize only trainable params (e.g., LoRA)
    trainable_params = (p for p in model.model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def corrupt_main(cfg):
    for corrupt in cfg.corruptions:
        cfg.corrupt = corrupt
        cfg.out_name = corrupt
        torch.cuda.empty_cache()
        main(cfg)


def main(cfg: Box) -> int:

    gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = Model(cfg)
        model.setup()

    load_datasets = call_load_dataset(cfg)
    train_data, val_data, pt_data = load_datasets(cfg, img_size=1024, return_pt = True)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    pt_data = fabric._setup_dataloader(pt_data)
    optimizer, scheduler = configure_opt(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])
    init_iou = 0
    # print('-'*100)
    # print('\033[92mDirect test on the original SAM.\033[0m') 
    # init_iou, _, = validate(fabric, cfg, model, val_data, name=cfg.name, epoch=0)
    # print('-'*100)
    # del _     




    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data, init_iou)

    del model, train_data, val_data






###############################################################################
# 🧠 SAM2 Integration
# ---------------------------------------------------------------------------
# This section handles all SAM2-related components including:
#   • Model prediction and mask generation
#   • Prompt handling (points, boxes, embeddings, etc.)
#   • Loss computation (focal, dice, entropy, IoU)
#   • Device management (ensuring tensors stay on the same GPU)
#   • Training and validation forward passes
#
# NOTE:
#   Ensure all tensors (pred_mask, soft_mask, etc.) are moved to the same device
#   before loss calculation to avoid cuda/cpu mismatches.
###############################################################################






from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
# from peft import LoraConfig, get_peft_model

model_cfg = "./configs/sam2/sam2_hiera_b+.yaml"
checkpoint = "./pretrain/sam2_hiera_base_plus.pt"



def sam2forward(img_tensor, prompts ,predictor):
    
    images = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    with torch.no_grad():
        predictor.set_image(images)
        entropy_maps = []
        pred_masks = []
        for i in range(prompts[0][0].shape[0]):
            mask_tuple, scores, logits = predictor.predict(
                point_coords=prompts[0][0][i].unsqueeze(0),      # single point
                point_labels=prompts[0][1][i].unsqueeze(0),      
                multimask_output=False           # only 1 mask
            )

            logits_full = F.interpolate(torch.tensor(logits).unsqueeze(0), size=(1024, 1024), mode='bilinear', align_corners=False)
            soft_mask_full = torch.sigmoid(logits_full[0][0])

            pred_mask = torch.sigmoid(soft_mask_full)

            pred_masks.append(pred_mask)
            
            entropy_map = entropy_map_calculate(pred_mask.unsqueeze(0))
            entropy_maps.append(entropy_map)
    
    return entropy_maps, pred_masks


def sam2forward_bbox(img_tensor, prompts_boxes ,predictor):
    
    images = img_tensor[0].permute(1, 2, 0).cpu().numpy()
    with torch.no_grad():
        predictor.set_image(images)
        pred_masks = []
        for i in range(prompts_boxes.shape[0]):
            mask_tuple, scores, logits = predictor.predict(
                box=prompts_boxes[i].unsqueeze(0),      # single point
                multimask_output=False           # only 1 mask
            )

            pred_masks.append(mask_tuple[0])


    return  pred_masks
        


# #     return pred_masks, Iou_prediciton
def pass_for_training(img_tensor, prompts, predictor):
    """
    Differentiable SAM2 forward pass for training with point prompts.
    """

    device = img_tensor.device
    image = img_tensor[0].to(device)

    # 1️⃣ Encode image (keep gradients)
    image_dict = predictor.model.image_encoder(image.unsqueeze(0))
    image_embedding = image_dict["vision_features"]
    image_pe = predictor.model.sam_prompt_encoder.get_dense_pe().to(device)

    mask_decoder = predictor.model.sam_mask_decoder

    

    # 3️⃣ Prepare prompts
    point_coords = prompts[0][0].to(device)
    point_labels = prompts[0][1].to(device)

    pred_masks = []
    iou_predictions = []

    # 4️⃣ Loop over each point
    for i in range(point_coords.shape[0]):
        single_point = point_coords[i].unsqueeze(0)
        single_label = point_labels[i].unsqueeze(0)

        sparse_embeddings, dense_embeddings = predictor.model.sam_prompt_encoder(
            points=(single_point, single_label),
            boxes=None,
            masks=None
        )

        # 5️⃣ Decode masks
     
        mask_logits, iou_pred, mask_tokens_out, object_score_logit = predictor.model.sam_mask_decoder(
            image_embeddings=image_embedding,
            image_pe=predictor.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,   # single mask output
            repeat_image=False,
        )

        # 6️⃣ Normalize masks
        mask_logits = F.interpolate(mask_logits, size=(1024, 1024),
                                    mode="bilinear", align_corners=False)
        mask_probs = torch.sigmoid(mask_logits[0, 0])

        pred_masks.append(mask_probs)
        iou_predictions.append(iou_pred)

    pred_masks = torch.stack(pred_masks)
    iou_predictions = torch.stack(iou_predictions)

    return pred_masks, iou_predictions





def train_sam2(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_pts,
):

    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    max_iou = 0.
    match_interval = cfg.match_interval

    predictor = SAM2ImagePredictor(model)

    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        focal_losses = AverageMeter()
        dice_losses = AverageMeter()
        iou_losses = AverageMeter()
        total_losses = AverageMeter()
        match_losses = AverageMeter()
        end = time.time()
        num_iter = len(train_dataloader)

        eval_interval = int(len(train_dataloader) * 0.1) 

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, img_paths= data
            del data

            slice_step = 50
            for i in range(0, len(gt_masks[0]), slice_step):
                
                gt_masks_new = gt_masks[0][i:i+slice_step].unsqueeze(0)
                prompts = get_prompts(cfg, bboxes, gt_masks_new)

                batch_size = images_weak.size(0)

                entropy_maps, preds = sam2forward(images_weak, prompts, predictor)
                pred_stack = torch.stack(preds, dim=0)
                pred_binary = (pred_stack > 0.5).float()
                overlap_count = pred_binary.sum(dim=0) 
                overlap_map = (overlap_count > 1).float()
                invert_overlap_map = 1.0 - overlap_map

                

                soft_masks = []
                bboxes = []
                point_list = []
                point_labels_list = []
                for i, (entr_map, pred) in enumerate(zip(entropy_maps, preds)):
                    point_coords = prompts[0][0][i][:].unsqueeze(0)
                    point_coords_lab = prompts[0][1][i][:].unsqueeze(0)

                    entr_norm = (entr_map - entr_map.min()) / (entr_map.max() - entr_map.min() + 1e-8)
                    
                    pred = (pred>0.5)
                    pred_w_overlap = pred * invert_overlap_map

                    ys, xs = torch.where(pred_w_overlap > 0.5)
                    if len(xs) > 0 and len(ys) > 0:
                        x_min, x_max = xs.min().item(), xs.max().item()
                        y_min, y_max = ys.min().item(), ys.max().item()
                        bboxes.append(torch.tensor([x_min, y_min , x_max, y_max], dtype=torch.float32))

                        point_list.append(point_coords)
                        point_labels_list.append(point_coords_lab)
                       
                if len(point_list) !=0:
                    point_ = torch.cat(point_list).squeeze(1)
                    point_labels_ = torch.cat(point_labels_list)
                    new_prompts = [(point_, point_labels_)]
                
                    bboxes = torch.stack(bboxes)

                    soft_masks = sam2forward_bbox(images_weak, bboxes, predictor)

                    pred_masks , iou_predictions = pass_for_training(images_strong, new_prompts,predictor )
                

                    torch.cuda.empty_cache()

                    num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
                    loss_focal = torch.tensor(0., device=fabric.device)
                    loss_dice = torch.tensor(0., device=fabric.device)
                    loss_iou = torch.tensor(0., device=fabric.device)

                    for i, (pred_mask, soft_mask, iou_prediction) in enumerate(
                            zip(pred_masks, soft_masks, iou_predictions  )
                        ):
                            
                            soft_mask = (torch.tensor(soft_mask) > 0.).float().unsqueeze(0)
                            pred_mask = pred_mask.unsqueeze(0).to(soft_mask.device)
                            iou_prediction = iou_prediction.to(soft_mask.device)
                        
                            
                            # Apply entropy mask to losses
                            loss_focal += focal_loss(pred_mask, soft_mask)  #, entropy_mask=entropy_mask
                            loss_dice += dice_loss(pred_mask, soft_mask)   #, entropy_mask=entropy_mask
                            batch_iou = calc_iou(pred_mask, soft_mask)
                            loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

                
                    del  pred_masks, iou_predictions 
                    # loss_dist = loss_dist / num_masks
                    loss_dice = loss_dice #/ num_masks
                    loss_focal = loss_focal #/ num_masks
                    torch.cuda.empty_cache()


                    loss_total =  20 * loss_focal +  loss_dice  + loss_iou #+ loss_iou  +  +



                    fabric.backward(loss_total)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    del  prompts, soft_masks

                    batch_time.update(time.time() - end)
                    end = time.time()

                    focal_losses.update(loss_focal.item(), batch_size)
                    dice_losses.update(loss_dice.item(), batch_size)
                    iou_losses.update(loss_iou.item(), batch_size)
                    total_losses.update(loss_total.item(), batch_size)
                
                    del loss_dice, loss_iou, loss_focal

            if (iter+1) %int(eval_interval/10)==0:
                fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                             f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                             f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                             f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                             f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                             f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                             f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            if (iter+1)%eval_interval == 0:
                iou, _, = validate_sam2(fabric, cfg, model, val_dataloader, name=cfg.name, epoch=0)
                del iou
            torch.cuda.empty_cache()
            
        # if epoch % cfg.eval_interval == 0:
        #     iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
        #     # if iou > max_iou:
        #     #     state = {"model": model, "optimizer": optimizer}
        #     #     fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
        #     #     max_iou = iou
        #     del iou  

def configure_opt2(cfg: Box, model: Model):

    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)

    # optimize only trainable params (e.g., LoRA)
    trainable_params = (p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(trainable_params, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    return optimizer, scheduler


def main2(cfg: Box) -> int:
    gpu_ids = [str(i) for i in range(torch.cuda.device_count())]
    num_devices = len(gpu_ids)
    fabric = L.Fabric(accelerator="auto",
                      devices=num_devices,
                      strategy="auto",
                      loggers=[TensorBoardLogger(cfg.out_dir)])
    fabric.launch()
    fabric.seed_everything(1337 + fabric.global_rank)

    if fabric.global_rank == 0:
        os.makedirs(os.path.join(cfg.out_dir, "save"), exist_ok=True)
        create_csv(os.path.join(cfg.out_dir, "metrics.csv"), csv_head=cfg.csv_keys)

    with fabric.device:
        model = build_sam2(model_cfg, checkpoint, mode='train')
    encoder = model.image_encoder
    lora_config = LoraConfig(
        r=4,                   # rank
        lora_alpha=16,
        target_modules=["qkv"],  # Hiera merges q,k,v in one linear layer
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # model.image_encoder = encoder


    load_datasets = call_load_dataset(cfg)
    train_data, val_data, pt_data = load_datasets(cfg, img_size=1024, return_pt = True)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    pt_data = fabric._setup_dataloader(pt_data)
    optimizer, scheduler = configure_opt2(cfg, model)
    model, optimizer = fabric.setup(model, optimizer)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(checkpoint)
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])


    # print('-'*100)
    # print('\033[92mDirect test on the original SAM.\033[0m') 
    # _, _, = validate_sam2(fabric, cfg, model, val_data, name=cfg.name, epoch=0)
    # print('-'*100)
    # del _     


    train_sam2(cfg, fabric, model, optimizer, scheduler, train_data, val_data, pt_data)

    del model, train_data, val_data



def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--prompt', help='the type of prompt')
    parser.add_argument('--num_points',type=int, help='the number of points')
    parser.add_argument('--out_dir', help='the dir to save logs and models')
    parser.add_argument('--load_type', help='the dir to save logs and models')      
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_args()

    exec(f'from {args.cfg} import cfg')

    # transfer the args to a dict
    args_dict = vars(args)
    cfg.merge_update(args_dict)
    print(cfg.model.backend)

    if cfg.model.backend == 'sam':
        main(cfg)
    elif cfg.model.backend == 'sam2':
        main2(cfg)
    torch.cuda.empty_cache()
