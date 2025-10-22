import os
import time
import argparse
import random
import math

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
from matplotlib import cm
from sklearn.mixture import GaussianMixture

import lightning as L
from lightning.fabric.loggers import TensorBoardLogger
from lightning.fabric.fabric import _FabricOptimizer

from box import Box
from datasets import call_load_dataset
from utils.model import Model
from utils.eval_utils import AverageMeter, validate, get_prompts, calc_iou
from utils.tools import copy_model, create_csv, reduce_instances
from utils.utils import *

class UncertaintyCalibratedSAM(nn.Module):
    """Uncertainty-aware calibration module for SAM"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Temperature scaling for calibration
        self.temperature = nn.Parameter(torch.ones(1))
        
        # Evidential uncertainty head (small conv layer)
        self.evidence_head = nn.Conv2d(256, 2, kernel_size=1)  # predicts alpha, beta for Dirichlet
        
        # Feature statistics for Mahalanobis distance
        self.register_buffer('feature_mean', torch.zeros(256))
        self.register_buffer('feature_cov', torch.eye(256))
        self.feature_stats_initialized = False
        
    def update_feature_stats(self, features):
        """Update feature statistics for Mahalanobis distance calculation"""
        if not self.feature_stats_initialized:
            # Flatten features: [B, C, H, W] -> [B*H*W, C]
            flat_features = features.permute(0, 2, 3, 1).reshape(-1, features.size(1))
            
            # Compute mean and covariance
            self.feature_mean = flat_features.mean(dim=0)
            self.feature_cov = torch.cov(flat_features.T)
            
            # Add small diagonal for numerical stability
            self.feature_cov += 1e-6 * torch.eye(features.size(1), device=features.device)
            self.feature_stats_initialized = True
    
    def compute_mahalanobis_distance(self, features):
        """Compute Mahalanobis distance for OOD detection"""
        if not self.feature_stats_initialized:
            return torch.zeros(features.size(0), features.size(2), features.size(3), device=features.device)
        
        # Flatten features: [B, C, H, W] -> [B*H*W, C]
        B, C, H, W = features.shape
        flat_features = features.permute(0, 2, 3, 1).reshape(-1, C)
        
        # Compute Mahalanobis distance
        diff = flat_features - self.feature_mean.unsqueeze(0)
        inv_cov = torch.linalg.inv(self.feature_cov)
        mahal_dist = torch.sqrt(torch.sum(diff @ inv_cov * diff, dim=1))
        
        # Reshape back to spatial dimensions
        mahal_dist = mahal_dist.reshape(B, H, W)
        return mahal_dist
    
    def compute_evidential_uncertainty(self, features):
        """Compute evidential uncertainty using Dirichlet distribution"""
        evidence = self.evidence_head(features)
        alpha = F.softplus(evidence[:, 0:1]) + 1  # concentration parameter
        beta = F.softplus(evidence[:, 1:2]) + 1
        
        # Total evidence
        total_evidence = alpha + beta
        
        # Uncertainty measures
        aleatoric_uncertainty = (alpha * beta) / ((total_evidence ** 2) * (total_evidence + 1))
        epistemic_uncertainty = 1 / (total_evidence + 1)
        
        return aleatoric_uncertainty, epistemic_uncertainty, alpha, beta
    
    def apply_temperature_scaling(self, logits):
        """Apply learnable temperature scaling for calibration"""
        return logits / self.temperature

class ConfidenceDrivenPromptGenerator:
    """Generate prompts based on uncertainty cues"""
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.uncertainty_threshold = 0.5
        
    def generate_uncertainty_prompts(self, images, model, bboxes, gt_masks, current_masks=None):
        """Generate prompts from uncertainty regions"""
        with torch.no_grad():
            # Get initial predictions for uncertainty estimation
            if current_masks is None:
                # Use generic prompts to get initial uncertainty map
                generic_prompts = get_prompts(self.cfg, bboxes, gt_masks)#self._generate_generic_prompts(images)
                _, initial_masks, _, _ = model(images, generic_prompts)
            else:
                initial_masks = current_masks
            
            # Compute uncertainty map
            uncertainty_maps = self._compute_uncertainty_map(initial_masks)

            print("FFFFFFFFFFFFFFFFFF", torch.stack(uncertainty_maps, dim=0).shape, torch.stack(gt_masks, dim=0).shape)
            
            # Sample points from high uncertainty regions
            prompts = []
            for i, uncertainty_map in enumerate(uncertainty_maps):
                # Sample positive points from high uncertainty regions
                pos_points = self._sample_uncertainty_points(uncertainty_map, self.cfg.num_points, positive=True)
                
                # Sample negative points from low uncertainty regions
                neg_points = self._sample_uncertainty_points(uncertainty_map, self.cfg.num_points, positive=False)
                
                # Combine points
                point_coords = torch.cat([pos_points, neg_points], dim=1)
                point_labels = torch.cat([
                    torch.ones(pos_points.shape[:2], dtype=torch.int, device=pos_points.device),
                    torch.zeros(neg_points.shape[:2], dtype=torch.int, device=neg_points.device)
                ], dim=1)
                
                prompts.append((point_coords, point_labels))
            
            return prompts, uncertainty_maps
    
    def _generate_generic_prompts(self, images):
        """Generate generic prompts for initial uncertainty estimation"""
        B, _, H, W = images.shape
        prompts = []
        
        for _ in range(B):
            # Sample random points across the image
            points = torch.randint(0, min(H, W), (1, self.cfg.num_points * 2, 2), device=images.device)
            labels = torch.ones(1, self.cfg.num_points * 2, dtype=torch.int, device=images.device)
            prompts.append((points.squeeze(0), labels.squeeze(0)))
        
        return prompts
    
    def _compute_uncertainty_map(self, masks):
        """Compute uncertainty map from predicted masks"""
        uncertainty_maps = []
        
        for mask_batch in masks:
            # Convert to probabilities and clamp
            p = torch.clamp(mask_batch, 1e-6, 1 - 1e-6)
            
            # Compute entropy as uncertainty measure
            entropy = -(p * torch.log(p) + (1 - p) * torch.log(1 - p))
            
            # Aggregate across instances
            if entropy.ndim == 3:
                entropy = entropy.max(dim=0)[0]
            
            uncertainty_maps.append(entropy)
        
        return uncertainty_maps
    
    def _sample_uncertainty_points(self, uncertainty_map, num_points, positive=True):
        """Sample points from uncertainty map"""
        H, W = uncertainty_map.shape
        
        if positive:
            # Sample from high uncertainty regions
            threshold = torch.quantile(uncertainty_map.flatten(), 0.8)
            mask = uncertainty_map > threshold
        else:
            # Sample from low uncertainty regions
            threshold = torch.quantile(uncertainty_map.flatten(), 0.2)
            mask = uncertainty_map < threshold
        
        # Get coordinates of selected regions
        coords = torch.nonzero(mask, as_tuple=False).float()
        
        if len(coords) == 0:
            # Fallback to random sampling
            coords = torch.randint(0, min(H, W), (num_points, 2), device=uncertainty_map.device).float()
        else:
            # Sample from available coordinates
            if len(coords) < num_points:
                # Repeat coordinates if not enough
                indices = torch.randint(0, len(coords), (num_points,), device=uncertainty_map.device)
            else:
                indices = torch.randperm(len(coords), device=uncertainty_map.device)[:num_points]
            
            coords = coords[indices]
        
        return coords.unsqueeze(0)

class CalibrationAwareLoss(nn.Module):
    """Calibration-aware loss combining segmentation and calibration objectives"""
    
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.lambda_cal = cfg.get('lambda_cal', 1.0)
        self.lambda_uncert = cfg.get('lambda_uncert', 0.1)
        
    def forward(self, pred_masks, pseudo_labels, uncertainty_info, mahal_distances):
        """
        Compute calibration-aware loss
        
        Args:
            pred_masks: Predicted masks from SAM
            pseudo_labels: Soft pseudo-labels (probabilistic targets)
            uncertainty_info: Dict containing uncertainty measures
            mahal_distances: Mahalanobis distances for OOD detection
        """
        # Segmentation loss (weighted by inverse uncertainty)
        seg_loss = self._compute_segmentation_loss(pred_masks, pseudo_labels, uncertainty_info)
        
        # Calibration loss (ACE-based)
        cal_loss = self._compute_calibration_loss(pred_masks, pseudo_labels)
        
        # Uncertainty regularization
        uncert_loss = self._compute_uncertainty_loss(mahal_distances, uncertainty_info)
        
        # Total loss
        total_loss = seg_loss + self.lambda_cal * cal_loss + self.lambda_uncert * uncert_loss
        
        return total_loss, {
            'seg_loss': seg_loss,
            'cal_loss': cal_loss,
            'uncert_loss': uncert_loss,
            'total_loss': total_loss
        }
    
    def _compute_segmentation_loss(self, pred_masks, pseudo_labels, uncertainty_info):
        """Compute segmentation loss weighted by uncertainty"""
        losses = []
        
        for pred_mask, pseudo_label in zip(pred_masks, pseudo_labels):
            # Convert to probabilities
            pred_prob = torch.sigmoid(pred_mask)
            
            # Compute uncertainty weights (inverse of uncertainty)
            uncertainty = uncertainty_info.get('total_uncertainty', torch.ones_like(pred_prob))
            weights = 1.0 / (uncertainty + 1e-6)
            
            # Binary cross-entropy loss'
            print(pred_prob.shape, pseudo_label.shape)
            bce_loss = F.binary_cross_entropy(pred_prob, pseudo_label, reduction='none')
            
            # Weight by uncertainty
            weighted_loss = bce_loss * weights
            losses.append(weighted_loss.mean())
        
        return torch.stack(losses).mean()
    
    def _compute_calibration_loss(self, pred_masks, pseudo_labels):
        """Compute calibration loss using ACE (Average Calibration Error)"""
        losses = []
        
        for pred_mask, pseudo_label in zip(pred_masks, pseudo_labels):
            pred_prob = torch.sigmoid(pred_mask)
            
            # Group predictions into confidence bins
            num_bins = 10
            bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=pred_prob.device)
            
            bin_losses = []
            for i in range(num_bins):
                # Find pixels in this confidence bin
                bin_mask = (pred_prob >= bin_boundaries[i]) & (pred_prob < bin_boundaries[i + 1])
                
                if bin_mask.sum() > 0:
                    bin_pred = pred_prob[bin_mask]
                    bin_label = pseudo_label[bin_mask]
                    
                    # Compute calibration error for this bin
                    bin_conf = bin_pred.mean()
                    bin_acc = bin_label.mean()
                    bin_error = torch.abs(bin_conf - bin_acc)
                    
                    bin_losses.append(bin_error)
            
            if bin_losses:
                losses.append(torch.stack(bin_losses).mean())
            else:
                losses.append(torch.tensor(0.0, device=pred_prob.device))
        
        return torch.stack(losses).mean()
    
    def _compute_uncertainty_loss(self, mahal_distances, uncertainty_info):
        """Compute uncertainty regularization loss"""
        losses = []
        
        for mahal_dist in mahal_distances:
            # Penalize high Mahalanobis distances (OOD regions)
            ood_penalty = torch.relu(mahal_dist - 2.0)  # Threshold at 2 standard deviations
            
            # Add evidential uncertainty regularization
            if 'aleatoric_uncertainty' in uncertainty_info:
                aleatoric = uncertainty_info['aleatoric_uncertainty']
                epistemic = uncertainty_info['epistemic_uncertainty']
                
                # Encourage appropriate uncertainty levels
                evidential_loss = torch.mean(aleatoric + epistemic)
            else:
                evidential_loss = torch.tensor(0.0, device=mahal_dist.device)
            
            losses.append(ood_penalty.mean() + evidential_loss)
        
        return torch.stack(losses).mean()

def train_uncertainty_calibrated_sam(
    cfg: Box,
    fabric: L.Fabric,
    model: Model,
    optimizer: _FabricOptimizer,
    scheduler: _FabricOptimizer,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    target_pts,
):
    """Main training function with uncertainty calibration"""
    
    # Initialize uncertainty calibration module
    uncertainty_module = UncertaintyCalibratedSAM(cfg)
    uncertainty_module = fabric.setup(uncertainty_module)
    
    # Initialize prompt generator
    prompt_generator = ConfidenceDrivenPromptGenerator(cfg)
    
    # Initialize calibration-aware loss
    calibration_loss = CalibrationAwareLoss(cfg)
    
    model.train()
    uncertainty_module.train()
    
    max_iou = 0.0
    
    for epoch in range(1, cfg.num_epochs + 1):
        batch_time = AverageMeter()
        data_time = AverageMeter()
        seg_losses = AverageMeter()
        cal_losses = AverageMeter()
        uncert_losses = AverageMeter()
        total_losses = AverageMeter()
        
        end = time.time()
        
        for iter, data in enumerate(train_dataloader):
            data_time.update(time.time() - end)
            # images, bboxes, gt_masks, img_paths = data
            images_soft, images, bboxes, gt_masks, img_paths = data
            
            batch_size = images.size(0)
            
            # Generate confidence-driven prompts
            prompts, uncertainty_maps = prompt_generator.generate_uncertainty_prompts(
                images, model, bboxes, 
                gt_masks
            )
            
            # Forward pass through SAM
            image_embeddings, pred_masks, iou_predictions, _ = model(images, prompts)
            
            # Update feature statistics for Mahalanobis distance
            if isinstance(image_embeddings, dict):
                features = image_embeddings['vision_features']
            else:
                features = image_embeddings
            
            uncertainty_module.update_feature_stats(features)
            
            # Compute uncertainty measures
            mahal_distances = uncertainty_module.compute_mahalanobis_distance(features)
            aleatoric_uncert, epistemic_uncert, alpha, beta = uncertainty_module.compute_evidential_uncertainty(features)
            
            # Apply temperature scaling to predictions
            scaled_logits = uncertainty_module.apply_temperature_scaling(
                torch.logit(torch.clamp(pred_masks[0], 1e-6, 1-1e-6))
            )
            calibrated_masks = torch.sigmoid(scaled_logits)

            print("CCCCCCCCCCCCCCCC", calibrated_masks.shape)
            
            # Create soft pseudo-labels from current predictions
            pseudo_labels = []
            for pred_mask in calibrated_masks:
                # Use current prediction as pseudo-label with smoothing
                smoothed_label = 0.7 * pred_mask + 0.3 * 0.5  # Label smoothing
                pseudo_labels.append(smoothed_label)

            print("PPPPPPPPPPPPPPPP", torch.stack(pseudo_labels, dim=0).shape)

            try:
                save_root = os.path.join(
                    ".", "debug_outputs",
                    f"epoch_{epoch:03d}_iter_{iter:05d}"
                )
                os.makedirs(save_root, exist_ok=True)

                def _to_u8_img(x: torch.Tensor) -> np.ndarray:
                    # Accepts mask-like tensors; returns uint8 [H,W]
                    x = x.detach().float().cpu()
                    # squeeze singleton dims
                    while x.ndim > 2:
                        x = x[0]
                    x = x.clamp(0, 1)
                    x = (x * 255.0).round().to(torch.uint8).numpy()
                    return x

                # Save GT masks
                # gt_masks can be list[Tensor] or Tensor
                if isinstance(gt_masks, (list, tuple)):
                    for bi, gtm in enumerate(gt_masks):
                        try:
                            img = _to_u8_img(gtm)
                            cv2.imwrite(os.path.join(save_root, f"gt_mask_b{bi}.png"), img)
                        except Exception as e:
                            print(f"[save][gt] skip b{bi}: {e}")
                elif torch.is_tensor(gt_masks):
                    B = gt_masks.shape[0]
                    for bi in range(B):
                        try:
                            img = _to_u8_img(gt_masks[bi])
                            cv2.imwrite(os.path.join(save_root, f"gt_mask_b{bi}.png"), img)
                        except Exception as e:
                            print(f"[save][gt] skip b{bi}: {e}")

                # Save pseudo labels (built from calibrated_masks)
                stacked_pseudo = torch.stack(pseudo_labels, dim=0)  # shape depends on calibrated_masks
                P = stacked_pseudo.shape[0]
                for pi in range(P):
                    try:
                        img = _to_u8_img(stacked_pseudo[pi])
                        cv2.imwrite(os.path.join(save_root, f"pseudo_b{pi}.png"), img)
                    except Exception as e:
                        print(f"[save][pseudo] skip p{pi}: {e}")

                # Save predicted masks
                # Raw predicted probabilities (from pred_masks[0]) and calibrated probabilities
                # pred_masks[0] is clamped earlier; treat as probabilities
                raw_pred = torch.clamp(pred_masks[0].detach(), 1e-6, 1 - 1e-6)
                RP = raw_pred.shape[0] if raw_pred.ndim > 2 else 1
                for ri in range(RP):
                    try:
                        img = _to_u8_img(raw_pred if RP == 1 else raw_pred[ri])
                        cv2.imwrite(os.path.join(save_root, f"pred_raw_b{ri}.png"), img)
                    except Exception as e:
                        print(f"[save][pred_raw] skip r{ri}: {e}")

                cal_pred = calibrated_masks.detach()
                CP = cal_pred.shape[0] if cal_pred.ndim > 2 else 1
                for ci in range(CP):
                    try:
                        img = _to_u8_img(cal_pred if CP == 1 else cal_pred[ci])
                        cv2.imwrite(os.path.join(save_root, f"pred_calibrated_b{ci}.png"), img)
                    except Exception as e:
                        print(f"[save][pred_cal] skip c{ci}: {e}")
            except Exception as e:
                print(f"[save] debug export failed: {e}")
            
            # Prepare uncertainty info
            uncertainty_info = {
                'total_uncertainty': uncertainty_maps[0] if uncertainty_maps else torch.ones_like(pred_masks[0]),
                'aleatoric_uncertainty': aleatoric_uncert,
                'epistemic_uncertainty': epistemic_uncert,
                'alpha': alpha,
                'beta': beta
            }
            
            # Compute calibration-aware loss
            _,loss_dict = calibration_loss(
                calibrated_masks,
                torch.stack(pseudo_labels, dim=0),
                uncertainty_info,
                [mahal_distances]
            )
            
            total_loss = loss_dict['total_loss']
            
            # Backward pass
            fabric.backward(total_loss)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Update meters
            seg_losses.update(loss_dict['seg_loss'].item(), batch_size)
            cal_losses.update(loss_dict['cal_loss'].item(), batch_size)
            uncert_losses.update(loss_dict['uncert_loss'].item(), batch_size)
            total_losses.update(total_loss.item(), batch_size)
            
            # Logging
            if (iter + 1) % 10 == 0:
                fabric.print(f'Epoch: [{epoch}][{iter + 1}/{len(train_dataloader)}]'
                           f' | Time [{batch_time.val:.3f}s ({batch_time.avg:.3f}s)]'
                           f' | Seg Loss [{seg_losses.val:.4f} ({seg_losses.avg:.4f})]'
                           f' | Cal Loss [{cal_losses.val:.4f} ({cal_losses.avg:.4f})]'
                           f' | Uncert Loss [{uncert_losses.val:.4f} ({uncert_losses.avg:.4f})]'
                           f' | Total Loss [{total_losses.val:.4f} ({total_losses.avg:.4f})]'
                           f' | Temp [{uncertainty_module.temperature.item():.4f}]')
            
            torch.cuda.empty_cache()
        
        # Validation
        if epoch % cfg.eval_interval == 0:
            iou, f1 = validate(fabric, cfg, model, val_dataloader, cfg.name, epoch)
            if iou > max_iou:
                state = {
                    "model": model,
                    "uncertainty_module": uncertainty_module,
                    "optimizer": optimizer
                }
                fabric.save(os.path.join(cfg.out_dir, "save", "best-ckpt.pth"), state)
                max_iou = iou

def configure_opt(cfg: Box, model: Model, uncertainty_module: UncertaintyCalibratedSAM):
    """Configure optimizer for both model and uncertainty module"""
    
    def lr_lambda(step):
        if step < cfg.opt.warmup_steps:
            return step / cfg.opt.warmup_steps
        elif step < cfg.opt.steps[0]:
            return 1.0
        elif step < cfg.opt.steps[1]:
            return 1 / cfg.opt.decay_factor
        else:
            return 1 / (cfg.opt.decay_factor**2)
    
    # Combine parameters from both model and uncertainty module
    all_params = list(model.model.parameters()) + list(uncertainty_module.parameters())
    optimizer = torch.optim.Adam(all_params, lr=cfg.opt.learning_rate, weight_decay=cfg.opt.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    return optimizer, scheduler

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
        
        # Initialize uncertainty calibration module
        uncertainty_module = UncertaintyCalibratedSAM(cfg)

    load_datasets = call_load_dataset(cfg)
    train_data, val_data, pt_data = load_datasets(cfg, img_size=1024, return_pt=True)
    train_data = fabric._setup_dataloader(train_data)
    val_data = fabric._setup_dataloader(val_data)
    pt_data = fabric._setup_dataloader(pt_data)
    
    optimizer, scheduler = configure_opt(cfg, model, uncertainty_module)
    model, optimizer = fabric.setup(model, optimizer)
    uncertainty_module = fabric.setup(uncertainty_module)

    if cfg.resume and cfg.model.ckpt is not None:
        full_checkpoint = fabric.load(cfg.model.ckpt)
        model.load_state_dict(full_checkpoint["model"])
        if "uncertainty_module" in full_checkpoint:
            uncertainty_module.load_state_dict(full_checkpoint["uncertainty_module"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])

    # Start uncertainty-calibrated training
    train_uncertainty_calibrated_sam(cfg, fabric, model, optimizer, scheduler, train_data, val_data, pt_data)

def parse_args():
    parser = argparse.ArgumentParser(description='Train uncertainty-calibrated SAM')
    parser.add_argument('--cfg', help='train config file path')
    parser.add_argument('--prompt', help='the type of prompt')
    parser.add_argument('--num_points', type=int, help='the number of points')
    parser.add_argument('--out_dir', help='the dir to save logs and models')
    parser.add_argument('--load_type', help='the dir to save logs and models')
    parser.add_argument('--lambda_cal', type=float, default=1.0, help='calibration loss weight')
    parser.add_argument('--lambda_uncert', type=float, default=0.1, help='uncertainty loss weight')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    print(torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.set_float32_matmul_precision('high')
    args = parse_args()

    exec(f'from {args.cfg} import cfg')

    # Transfer the args to a dict
    args_dict = vars(args)
    cfg.merge_update(args_dict)

    main(cfg)
    torch.cuda.empty_cache()

















