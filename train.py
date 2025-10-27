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
            prompts = get_prompts(cfg, bboxes, gt_masks)

            batch_size = img_tensor.size(0)

            entropy_maps, preds = process_forward(images_weak, prompts, model)
            pred_stack = torch.stack(preds, dim=0)
            pred_binary = (pred_stack > 0.99).float() 
            overlap_count = pred_binary.sum(dim=0)
            overlap_map = (overlap_count > 1).float()
            invert_overlap_map = 1.0 - overlap_map


            soft_masks = []
            for i, (entr_map, pred) in enumerate(zip(entropy_maps, preds)):
                entr_norm = (entr_map - entr_map.min()) / (entr_map.max() - entr_map.min() + 1e-8)
                entr_vis = (entr_norm[0].cpu().numpy() * 255).astype(np.uint8)
                pred = (pred[0]>0.99)
                pred_w_overlap = pred * invert_overlap_map[0]

                soft_masks.append(pred_w_overlap)




            _, pred_masks, iou_predictions, _= model(images_strong, prompts)
            del _



        
        
            num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
            loss_focal = torch.tensor(0., device=fabric.device)
            loss_dice = torch.tensor(0., device=fabric.device)
            loss_iou = torch.tensor(0., device=fabric.device)


       

            for i, (pred_mask, soft_mask, iou_prediction) in enumerate(
                    zip(pred_masks[0], soft_masks, iou_predictions[0]  )
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








    # focal_loss = FocalLoss()
    # dice_loss = DiceLoss()
    # max_iou = 0.

    # for epoch in range(1, cfg.num_epochs + 1):
    #     batch_time = AverageMeter()
    #     data_time = AverageMeter()
    #     focal_losses = AverageMeter()
    #     dice_losses = AverageMeter()
    #     iou_losses = AverageMeter()
    #     total_losses = AverageMeter()

    #     end = time.time()
        
    #     for rank, (entropy_scalar, img_path, render) in enumerate(collected, start=1):
    #         img_tensor = torch.from_numpy(render['real']).permute(2,0,1).float() / 255.0
    #         img_tensor = img_tensor.unsqueeze(0).to(fabric.device)

    #         prompt_main = render['prompt']
    #         entropy_maps, pred_masks, iou_predictions = process_forward(img_tensor, prompt_main, model)

    #         prompt_1 = prompt_calibration(cfg, entropy_maps, prompt_main, 1)
    #         entropy_maps_1, preds_1, _ = process_forward(img_tensor, prompt_1, model)
    #         entr_means_pos = [ent.mean() for ent in entropy_maps_1]

    #         # Compute relative entropy = mean entropy / (number of 1s in prediction)
    #         rel_entr_pos = []
    #         for ent, pred in zip(entropy_maps_1, preds_1):
    #             num_ones = (pred >0.0 ).sum()
    #             rel_entr = ent.mean()/ num_ones if num_ones > 0 else float('inf')
    #             rel_entr_pos.append(rel_entr)

    #         prompt_2 = prompt_calibration(cfg, entropy_maps, prompt_main, 0)
    #         entropy_maps_2, preds_2,_ = process_forward(img_tensor, prompt_2, model)
    #         entr_means_neg = [ent.mean() for ent in entropy_maps_2]

    #         rel_entr_neg = []
    #         for ent, pred in zip(entropy_maps_2, preds_2):
    #             num_ones = (pred > 0.0).sum()
    #             rel_entr = ent.mean()/ num_ones if num_ones > 0 else float('inf')
    #             rel_entr_neg.append(rel_entr)

    #         soft_mask_final = []
    #         for i in range(len(preds_1)):
    #             if rel_entr_pos[i] < rel_entr_neg[i]:
    #                 soft_mask_final.append(preds_1[i])
    #             else:
    #                 soft_mask_final.append(preds_2[i])
            
    #         num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
    #         loss_focal = torch.tensor(0., device=fabric.device)
    #         loss_dice = torch.tensor(0., device=fabric.device)
    #         loss_match = torch.tensor(0., device=fabric.device)
    #         loss_iou = torch.tensor(0., device=fabric.device)

           

    #         _, pred_masks, iou_predictions, _ = model(img_tensor, prompt_main)

            
    #         for i, ( pred_mask, soft_mask, prompt, iou_prediction) in enumerate(zip( pred_masks[0], soft_mask_final, prompt_main, iou_predictions)):
    #             # soft_mask = (soft_mask > 0.).float()
      

              

    #             loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
    #             loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
    #             pred_mask = pred_mask.unsqueeze(0)
    #             batch_iou = calc_iou(pred_mask, soft_mask)
    #             iou_prediction = iou_prediction.unsqueeze(1)
    #             loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks

    #         del  pred_masks, iou_predictions

    #         loss_total = 20. * loss_focal + loss_dice #+ loss_iou 

    #         fabric.backward(loss_total)

    #         optimizer.step()
    #         scheduler.step()
    #         optimizer.zero_grad()
    #         torch.cuda.empty_cache()

    #         batch_time.update(time.time() - end)
    #         end = time.time()

    #         focal_losses.update(loss_focal.item(), 1)
    #         dice_losses.update(loss_dice.item(), 1)
    #         iou_losses.update(loss_iou.item(), 1)
    #         total_losses.update(loss_total.item(), 1)
 

    #         if (rank+1) %5==0:
    #             fabric.print(f'Epoch: [{epoch}][{rank + 1}/{len(train_dataloader)}]'
    #                          f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
    #                          f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
    #                          f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
    #                          f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
    #                          f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
    #                          f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

    #         # loss_logger = {
    #         #     "Focal Loss": focal_losses.avg,
    #         #     "Dice Loss": dice_losses.avg,
    #         #     "IoU Loss": iou_losses.avg,
    #         #     "Total Loss": total_losses.avg
    #         # }
    #         # fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)

    #         if (rank+1)%40 == 0:
    #             iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
    #         torch.cuda.empty_cache()
            
        # if epoch % cfg.eval_interval == 0:
        #     iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
        #     if iou > max_iou:
        #         state = {"model": model, "optimizer": optimizer}
        #         fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
        #         max_iou = iou
        #     del iou

            


   
    # max_iou = 0.
    # mem_bank = Store(1, cfg.mem_bank_max_len) 
    # match_interval = cfg.match_interval


   

    #     for iter, data in enumerate(train_dataloader):

    #         data_time.update(time.time() - end)
    #         images_weak, images_strong, bboxes, gt_masks, img_paths= data
    #         del data  
            
        #     batch_size = images_weak.size(0)
        #     num_insts = sum(len(gt_mask) for gt_mask in gt_masks)
        #     if num_insts > cfg.max_nums:
        #         bboxes, gt_masks = reduce_instances(bboxes, gt_masks, cfg.max_nums)
        #     prompts = get_prompts(cfg, bboxes, gt_masks)

        #     #1. caculate pairwise IoUs of masks
        #     mask_ious, init_masks = cal_mask_ious(cfg, model, images_weak, prompts, gt_masks)
       
        #     #2. get new prompts through neg_prompt_calibration
        #     new_prompts = neg_prompt_calibration(cfg, mask_ious, prompts)

        #     #3. start training using new prompt
        #     soft_image_embeds, soft_masks, _, _ = model(images_weak, new_prompts)    # teacher
            
        #     if isinstance(soft_image_embeds, dict):
        #         soft_image_embeds = soft_image_embeds['vision_features']  
        #     # torch.cuda.empty_cache()
        #     _, pred_masks, iou_predictions, _= model(images_strong, prompts)   # student

        #     del _

        #     num_masks = sum(len(pred_mask) for pred_mask in pred_masks)
        #     loss_focal = torch.tensor(0., device=fabric.device)
        #     loss_dice = torch.tensor(0., device=fabric.device)
        #     loss_match = torch.tensor(0., device=fabric.device)
        #     loss_iou = torch.tensor(0., device=fabric.device)
            
        #     for i, (embed, pred_mask, soft_mask,  gt_mask, prompt, iou_prediction) in enumerate(zip(soft_image_embeds, pred_masks, soft_masks, gt_masks, prompts, iou_predictions)):
                    
        #         soft_mask = (soft_mask > 0.).float()
        #         pred_feats = generate_predict_feats(cfg, embed, soft_mask, prompt)
        #         target_pts_ = target_pts['target_pts']  
        #         pred_feats = pred_feats.cpu().tolist()
        #         for pred_feat in pred_feats:
        #             mem_bank.add([[pred_feat]], [0])
        #         if len(mem_bank.retrieve(0)) >= cfg.mem_bank_max_len and (iter + 1) % match_interval == 0:
        #             pred_feats = mem_bank.retrieve(0)  
        #             pred_feats = np.array(pred_feats) 

        #             #FINCH
        #             fin = FINCH(verbose=False)
        #             results = fin.fit(pred_feats)
        #             last_key = list(results.partitions.keys())[-1]
        #             pred_pts = results.partitions[last_key]['cluster_centers']
        #             loss_match += Matching_Loss(pred_pts, target_pts_, device = fabric.device)

        #         del embed
                
        #         if vis:
        #             img_name = os.path.basename(img_paths[i]).split('.')[0]
                    
        #             image_weak = images_weak[0].permute(1,2,0).cpu().numpy()* 255
        #             image_weak = cv2.cvtColor(image_weak, cv2.COLOR_BGR2RGB)

        #             if vis:
        #                 for j in range(len(soft_mask)):
        #                     mask_iou = torch.max(mask_ious[j])
        #                     image_weak_ = image_weak.copy()
        #                     mask_area = torch.sum(gt_mask[j])
                            
        #                     gt_mask_np = gt_mask[j].cpu().numpy() * 255 
        #                     gt_mask_img = cv2.cvtColor(gt_mask_np, cv2.COLOR_GRAY2RGB)
                            
        #                     init_prompt_po = prompts[0][0][j][:cfg.num_points]
        #                     init_prompt_ne = prompts[0][0][j][cfg.num_points:]
                            
        #                     for po in init_prompt_po:
        #                         cv2.circle(image_weak_, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
                                
        #                     init_mask_img = init_masks[j].cpu().detach().numpy() * 255
        #                     init_mask_img = cv2.cvtColor(init_mask_img, cv2.COLOR_GRAY2RGB)
        #                     for po,ne in zip(init_prompt_po,init_prompt_ne):
        #                         cv2.circle(init_mask_img, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
        #                         cv2.circle(init_mask_img, (int(ne[0]), int(ne[1])), 12, (0, 255, 0), -1)
                                
        #                     prompt_po = new_prompts[0][0][j][:cfg.num_points]
        #                     prompt_ne = new_prompts[0][0][j][cfg.num_points:]
        #                     soft_mask_img = soft_mask[j].cpu().detach().numpy() * 255  

        #                     soft_mask_img = cv2.cvtColor(soft_mask_img, cv2.COLOR_GRAY2RGB)

        #                     for po,ne in zip(prompt_po,prompt_ne):
        #                         cv2.circle(soft_mask_img, (int(po[0]), int(po[1])), 12, (0, 0, 255), -1)
        #                         cv2.circle(soft_mask_img, (int(ne[0]), int(ne[1])), 12, (0, 255, 0), -1)

        #                     output_dir = "./save_mask_{}/{}/{}/".format(str(float(cfg.num_points)),cfg.dataset,str(epoch))
        #                     if not os.path.exists(output_dir):
        #                         os.makedirs(output_dir)
        #                     merged_image = concatenate_images_with_padding([image_weak_, gt_mask_img, init_mask_img, soft_mask_img])
        #                     img_name_ = '{}_{}_iou{}.jpg'.format(img_name,str(j),str(mask_iou))
        #                     if mask_iou>float(cfg.iou_thr) and mask_area>3000:
        #                         cv2.imwrite(os.path.join(output_dir,img_name_), merged_image) 
        #         del init_masks, mask_ious

        #         loss_focal += focal_loss(pred_mask, soft_mask, num_masks)
        #         loss_dice += dice_loss(pred_mask, soft_mask, num_masks)
        #         batch_iou = calc_iou(pred_mask, soft_mask)
        #         loss_iou += F.mse_loss(iou_prediction, batch_iou, reduction='sum') / num_masks
                
        #     del soft_image_embeds, pred_masks, iou_predictions, gt_masks 
                
        #     loss_total = 20. * loss_focal + loss_dice + loss_iou + 0.1*loss_match            

        #     fabric.backward(loss_total)

        #     optimizer.step()
        #     scheduler.step()
        #     optimizer.zero_grad()
        #     torch.cuda.empty_cache()

        #     batch_time.update(time.time() - end)
        #     end = time.time()

        #     focal_losses.update(loss_focal.item(), batch_size)
        #     dice_losses.update(loss_dice.item(), batch_size)
        #     iou_losses.update(loss_iou.item(), batch_size)
        #     total_losses.update(loss_total.item(), batch_size)
        #     match_losses.update(loss_match.item(), batch_size)

        #     if (iter+1) %match_interval==0:
        #         fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
        #                      f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
        #                      f' | Data [{data_time.val:.3f}s ({data_time.avg:.3f}s)]'
        #                      f' | Focal Loss [{focal_losses.val:.4f} ({focal_losses.avg:.4f})]'
        #                      f' | Dice Loss [{dice_losses.val:.4f} ({dice_losses.avg:.4f})]'
        #                      f' | IoU Loss [{iou_losses.val:.4f} ({iou_losses.avg:.4f})]'
        #                      f' | Match Loss [{match_losses.val:.4f} ({match_losses.avg:.4f})]'
        #                      f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]')

        #     # loss_logger = {
        #     #     "Focal Loss": focal_losses.avg,
        #     #     "Dice Loss": dice_losses.avg,
        #     #     "IoU Loss": iou_losses.avg,
        #     #     "Total Loss": total_losses.avg
        #     # }
        #     # fabric.log_dict(loss_logger, num_iter * (epoch - 1) + iter)
        #     torch.cuda.empty_cache()
            
        # if epoch % cfg.eval_interval == 0:
        #     iou, _= validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
        #     if iou > max_iou:
        #         state = {"model": model, "optimizer": optimizer}
        #         fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
        #         max_iou = iou
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

#     del model, train_data, val_data


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
