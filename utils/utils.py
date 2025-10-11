import random
from collections import deque
from tqdm import tqdm
from box import Box
import numpy as np
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn as nn
import torch.nn.functional as F

from .finch import FINCH
from .sample_utils import get_point_prompts

class Store:
    def __init__(self, total_num_classes, items_per_class, shuffle=False):
        self.shuffle = shuffle
        self.items_per_class = items_per_class
        self.total_num_classes = total_num_classes
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def add(self, items, class_ids):
        for idx, class_id in enumerate(class_ids):
            self.store[class_id].append(items[idx])

    def retrieve(self, class_id):
        if class_id != -1:
            items = []
            for item in self.store[class_id]:
                items.extend(list(item))
            if self.shuffle:
                random.shuffle(items)
            return items
        else:
            all_items = []
            for i in range(self.total_num_classes):
                items = []
                for item in self.store[i]:
                    items.append(list(item))
                all_items.append(items)
            return all_items

    def reset(self):
        self.store = [deque(maxlen=self.items_per_class) for _ in range(self.total_num_classes)]

    def __str__(self):
        s = self.__class__.__name__ + '('
        for idx, item in enumerate(self.store):
            s += '\n Class ' + str(idx) + ' --> ' + str(len(list(item))) + ' items'
        s = s + ' )'
        return s

    def __repr__(self):
        return self.__str__()

    def __len__(self):
        return sum([len(s) for s in self.store])

def concatenate_images_with_padding(images, padding=10, color=(255, 255, 255)):
    heights = [image.shape[0] for image in images]
    widths = [image.shape[1] for image in images]

    total_width = sum(widths) + padding * (len(images) - 1)
    max_height = max(heights)
    
    if len(images[0].shape) == 3:
        new_image = np.full((max_height, total_width, 3), color, dtype=np.uint8)
    else:
        new_image = np.full((max_height, total_width), color[0], dtype=np.uint8)

    x_offset = 0
    for image in images:
        new_image[0:image.shape[0], x_offset:x_offset + image.shape[1]] = image
        x_offset += image.shape[1] + padding
    
    return new_image

def calculate_iou(mask1, mask2):
    intersection = torch.logical_and(mask1, mask2)
    union = torch.logical_or(mask1, mask2)
    iou = torch.sum(intersection).float() / torch.sum(union).float()
    return iou

def calc_iou_matrix(mask_list1, mask_list2):
    iou_matrix = torch.zeros((len(mask_list1), len(mask_list2)))
    for i, mask1 in enumerate(mask_list1):
        for j, mask2 in enumerate(mask_list2):
            iou_matrix[i, j] = calculate_iou(mask1, mask2)
    return iou_matrix

def cal_mask_ious(
    cfg,
    model,
    images_weak,
    prompts,
    gt_masks,
):
    with torch.no_grad():
         _, soft_masks, _, _ = model(images_weak, prompts)   

    for i, (soft_mask, gt_mask) in enumerate(zip(soft_masks, gt_masks)):  
        soft_mask = (soft_mask > 0).float()
        mask_ious = calc_iou_matrix(soft_mask, soft_mask)
        indices = torch.arange(mask_ious.size(0))
        mask_ious[indices, indices] = 0.0
    return mask_ious, soft_mask


def neg_prompt_calibration(
    cfg,
    mask_ious,
    prompts,
):
    '''
    mask_ious:[mask_nums,mask_nums]
    '''
    point_list = []
    point_labels_list = []
    num_points = cfg.num_points
    for m in range(len(mask_ious)):
            
        pos_point_coords = prompts[0][0][m][:num_points].unsqueeze(0) 
        neg_point = prompts[0][0][m][num_points:].unsqueeze(0)  
        neg_points_list = []
        neg_points_list.extend(neg_point[0])

        indices = torch.nonzero(mask_ious[m] > float(cfg.iou_thr)).squeeze(1)

        if indices.numel() != 0:
            # neg_points_list = []
            for indice in indices:
                neg_points_list.extend(prompts[0][0][indice][:num_points])
            neg_points = random.sample(neg_points_list, num_points)
        else:
            neg_points =neg_points_list
            
        neg_point_coords = torch.tensor([p.tolist() for p in neg_points], device=neg_point.device).unsqueeze(0)

        point_coords = torch.cat((pos_point_coords, neg_point_coords), dim=1) 

        point_list.append(point_coords)
        pos_point_labels = torch.ones(pos_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        neg_point_labels = torch.zeros(neg_point_coords.shape[0:2], dtype=torch.int, device=neg_point.device)
        point_labels = torch.cat((pos_point_labels, neg_point_labels), dim=1)  
        point_labels_list.append(point_labels)

    point_ = torch.cat(point_list).squeeze(1)
    point_labels_ = torch.cat(point_labels_list)
    new_prompts = [(point_, point_labels_)]
    return new_prompts

def get_prompts(cfg: Box, bboxes, gt_masks):
    if cfg.prompt == "box" or cfg.prompt == "coarse":
        prompts = bboxes
    elif cfg.prompt == "point":
        prompts = get_point_prompts(gt_masks, cfg.num_points)
    else:
        raise ValueError("Prompt Type Error!")
    return prompts

def generate_predict_feats(cfg, embed, pseudo_label, gts):
    coords, lbls = gts
    selected_coords = []
    
    num_insts = len(pseudo_label)
    num_points = cfg.num_points
    for coord_grp, lbl_grp in zip(coords, lbls):
        for coord, lbl in zip(coord_grp, lbl_grp):  
            if lbl.item() == 1:  
                selected_coords.append(coord.tolist())

    # Downsample coordinates (SAM's stride is 16)
    coords = [[int(c // 16) for c in pair] for pair in selected_coords]

    embed = embed.permute(1, 2, 0)  # [H, W, C]

    pos_pts = [] 

    for index in range(0, num_insts * num_points, num_points):
        index = random.randint(0, num_points - 1)
        x, y = coords[index]
        pos_pt = embed[x, y]
        pos_pts.append(pos_pt)

    predict_feats = torch.stack(pos_pts, dim=0)

    return predict_feats


def offline_prototypes_generation(cfg, model, loader):
    model.eval()
    pts = []
    max_iters = 128 
    num_points = cfg.num_points

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Generating target prototypes', ncols=100)):
            if i >= max_iters: 
                break
            imgs, boxes, masks, _ = batch
            prompts = get_prompts(cfg, boxes, masks)

            embeds, masks, _, _ = model(imgs, prompts) 
            del _

            if isinstance(embeds, dict):
                embeds = embeds['vision_features'] 

            for embed, prompt, mask in zip(embeds, prompts, masks):
                num_insts = len(mask)
                embed = embed.permute(1, 2, 0)  # [H, W, C]
                coords = []

                points, labels = prompt
                for point_grp, label_grp in zip(points, labels):
                    for point, label in zip(point_grp, label_grp): 
                        if label.item() == 1:  
                            coords.append(point.tolist())

                # 16 is the stride of SAM
                coords = [[int(pt / 16) for pt in pair] for pair in coords]
                for index in range(0, num_insts*num_points, num_points):
                    x, y = coords[index]
                    pt = embed[x,y]
                    pts.append(pt)

    fin = FINCH(verbose=True)
    pts = torch.stack(pts).cpu().numpy()
    res = fin.fit(pts)

    last_key = list(res.partitions.keys())[-1]
    pt_stats = {'target_pts': res.partitions[last_key]['cluster_centers']}
    return pt_stats






from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image, ImageDraw

def save_uncertanity_mask(cfg, model, loader):
    model.eval()
    pts = []
    num_points = cfg.num_points


    save_dir = "temp_test"
    # make save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with torch.no_grad():
        for i, batch in enumerate(tqdm(loader, desc='Generating target prototypes', ncols=100)):
            
            imgs, boxes, masks, _ = batch
            prompts = get_prompts(cfg, boxes, masks)

            embeds, masks_pred, _, _ = model(imgs, prompts) 
            del _

            p = masks_pred[0].clamp(1e-6, 1 - 1e-6)

            print(prompts[0][0].shape)
      

            entropy_map = - (p * torch.log(p) + (1 - p) * torch.log(1 - p))
            entropy_map = entropy_map.max(dim=0)[0]   # take pixel-wise max across instances → [B, H, W]
            # print(entropy_map.shape)
            # -----------------
            # Save images
            # -----------------
            for b in range(imgs.shape[0]):
                # Convert image tensor (C,H,W) → (H,W,C)
                img_np = (imgs[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)

                # ===== Ground Truth Mask =====
                gt_mask = masks[b].cpu().numpy()   # [N, H, W]
                if gt_mask.ndim == 3:
                    gt_mask = np.max(gt_mask, axis=0) * 255   # merge all instances
                gt_mask = gt_mask.astype(np.uint8)

                # ===== Prediction Mask =====
                pred_mask = masks_pred[b].cpu().numpy()   # [N, H, W]
                if pred_mask.ndim == 3:
                    pred_mask = np.max(pred_mask, axis=0)   
                pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
                pred_mask = pred_mask.astype(np.uint8)

                # ===== Entropy Heatmap =====
                entropy_np = entropy_map.cpu().numpy()
                entropy_norm = (entropy_np - entropy_np.min()) / (entropy_np.max() - entropy_np.min() + 1e-6)
                cmap = cm.get_cmap("viridis")
                entropy_color = (cmap(entropy_norm)[:, :, :3] * 255).astype(np.uint8)
                
                # -----------------
                # Overlay prompts on masks
                # -----------------
                gt_img = Image.fromarray(gt_mask).convert("RGB")
                pred_img = Image.fromarray(pred_mask).convert("RGB")

                draw_gt = ImageDraw.Draw(gt_img)
                draw_pred = ImageDraw.Draw(pred_img)

                # points[b] → torch.Size([num_points, 2, 2])
                pts = prompts[0][0].cpu().numpy()  # [num_points, 2, 2]
                for pnt in pts:
                    x, y = pnt[0]   # first [x,y]
                    label = int(pnt[1][0]) if pnt.shape[0] > 1 else 1  # assume label in second entry
                    color = "green" if label == 1 else "red"
                    r = 4
                    draw_gt.ellipse((x-r, y-r, x+r, y+r), fill=color)
                    draw_pred.ellipse((x-r, y-r, x+r, y+r), fill=color)

                # Save outputs
                Image.fromarray(img_np).save(os.path.join(save_dir, f"{i+1}.jpg"))         # original
                gt_img.save(os.path.join(save_dir, f"{i+1}_gt.jpg"))                       # GT + points
                pred_img.save(os.path.join(save_dir, f"{i+1}_pred.jpg"))                   # Pred + points
                Image.fromarray(entropy_color).save(os.path.join(save_dir, f"{i+1}_en.jpg")) # entropy



            # if i > 2:   # debugging stop
            #     break


            # print(masks[0].shape)

            # print(masks_pred[0].shape)
            # print(embeds.shape)

            # if i > 2:

            #     break


            # if isinstance(embeds, dict):
            #     embeds = embeds['vision_features'] 

            # for embed, prompt, mask in zip(embeds, prompts, masks):
            #     num_insts = len(mask)
            #     embed = embed.permute(1, 2, 0)  # [H, W, C]
            #     coords = []

            #     points, labels = prompt
            #     for point_grp, label_grp in zip(points, labels):
            #         for point, label in zip(point_grp, label_grp): 
            #             if label.item() == 1:  
            #                 coords.append(point.tolist())

            #     # 16 is the stride of SAM
            #     coords = [[int(pt / 16) for pt in pair] for pair in coords]
            #     for index in range(0, num_insts*num_points, num_points):
            #         x, y = coords[index]
            #         pt = embed[x,y]
            #         pts.append(pt)
