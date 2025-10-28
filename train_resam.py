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
from utils.eval_utils import AverageMeter, validate, get_prompts, calc_iou
from utils.tools import copy_model, create_csv, reduce_instances
from utils.utils import *
# from utils.finch import FINCH

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
    entropy_map = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
    entropy_map = entropy_map.max(dim=0)[0]

    return entropy_map

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


def train_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_pts,
):
    # collected = sort_entropy_(model, target_pts)
    focal_loss = FocalLoss()
    dice_loss = DiceLoss()
    max_iou = 0.
    match_interval = cfg.match_interval

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

        for iter, data in enumerate(train_dataloader):

            data_time.update(time.time() - end)
            images_weak, images_strong, bboxes, gt_masks, img_paths= data
            del data

        # for rank, (entropy_scalar, img_path, render) in enumerate(collected, start=1):
        #     img_name = os.path.splitext(os.path.basename(img_path))[0]

            # ---- Convert and move to device
            # img_np = render['real']        # numpy HxWx3
            # images_weak = torch.from_numpy(img_np).permute(2,0,1).float() / 255.0
            # images_weak = images_weak.unsqueeze(0).to(fabric.device)

            # prompts = render['prompt']

            

            prompts = get_prompts(cfg, bboxes, gt_masks)

            batch_size = images_weak.size(0)

            entropy_maps, preds = process_forward(images_weak, prompts, model)
            pred_stack = torch.stack(preds, dim=0)
            pred_binary = (pred_stack > 0.99).float() 
            overlap_count = pred_binary.sum(dim=0)
            overlap_map = (overlap_count > 1).float()
            invert_overlap_map = 1.0 - overlap_map


            soft_masks = []
            bboxes = []
            point_list = []
            point_labels_list = []
            flag_train = True
            for i, (entr_map, pred) in enumerate(zip(entropy_maps, preds)):
                entr_norm = (entr_map - entr_map.min()) / (entr_map.max() - entr_map.min() + 1e-8)
                entr_vis = (entr_norm[0].cpu().numpy() * 255).astype(np.uint8)
                pred = (pred[0]>0.99)
                pred_w_overlap = pred * invert_overlap_map[0]

                ys, xs = torch.where(pred_w_overlap > 0.5)
                if len(xs) > 0 and len(ys) > 0:
                    x_min, x_max = xs.min().item(), xs.max().item()
                    y_min, y_max = ys.min().item(), ys.max().item()
                    bboxes.append(torch.tensor([x_min, y_min , x_max, y_max], dtype=torch.float32))
                    point_list.append(prompts[0][0][i])
                    print(prompts[0][1][i])
                    point_labels_list.append(prompts[0][1][i])
                else:

                    flag_train  = False
                    # print("No 1s found in mask")
            point_list = torch.cat(point_list)
            print(point_list.shape)

            point_labels_ = torch.cat(point_labels_list)
            print(point_labels_.shape)
            new_prompts = [(point_list, point_labels_)]
            # print(new_prompts[0].shape)
                
            if True :
                bboxes = torch.stack(bboxes)

                with torch.no_grad():
                    _, soft_masks, _, _ = model(images_weak, bboxes.unsqueeze(0))
                

                
                
                # soft_masks.append(pred_w_overlap)




                _, pred_masks, iou_predictions, _= model(images_strong, new_prompts)
                del _



            
            
                num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
                loss_focal = torch.tensor(0., device=fabric.device)
                loss_dice = torch.tensor(0., device=fabric.device)
                loss_iou = torch.tensor(0., device=fabric.device)


        

                for i, (pred_mask, soft_mask, iou_prediction) in enumerate(
                        zip(pred_masks[0], soft_masks[0], iou_predictions[0]  )
                    ):
                    
        

                        soft_mask = (soft_mask > 0.).float()
                        # Apply entropy mask to losses
                        loss_focal += focal_loss(pred_mask, soft_mask)  #, entropy_mask=entropy_mask
                        loss_dice += dice_loss(pred_mask, soft_mask)   #, entropy_mask=entropy_mask
                        batch_iou = calc_iou(pred_mask.unsqueeze(0), soft_mask.unsqueeze(0))
                        loss_iou += F.mse_loss(iou_prediction.view(-1), batch_iou.view(-1), reduction='sum') / num_masks

            
                del  pred_masks, iou_predictions 
                # loss_dist = loss_dist / num_masks
                loss_dice = loss_dice / num_masks
                loss_focal = loss_focal / num_masks
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
            if (iter+1) %match_interval==0:
                fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                             f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                             f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
                             f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
                             f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
                             f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
                             f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

            if (iter+1)%100 == 0:
                iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
                del iou
            torch.cuda.empty_cache()
            
        # if epoch % cfg.eval_interval == 0:
        #     iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
        #     # if iou > max_iou:
        #     #     state = {"model": model, "optimizer": optimizer}
        #     #     fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
        #     #     max_iou = iou
        #     del iou   




            
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

    optimizer = torch.optim.Adam(model.model.parameters(), lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
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
    # print('-'*100)
    # print('\033[92mDirect test on the original SAM.\033[0m') 
    # _, _, = validate(fabric, cfg, model, val_data, name=cfg.name, epoch=0)
    # print('-'*100)
    # del _     



    # save_uncertanity_mask(cfg, model, pt_data)
    

#     target_pts = offline_prototypes_generation(cfg, model, pt_data)
    
    train_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data, pt_data)

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

    main(cfg)
    torch.cuda.empty_cache()
