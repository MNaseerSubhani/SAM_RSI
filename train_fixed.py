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
    model.eval()

    save_dir = "entropy_sorted"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Collect entropy and renders in one pass
    collected = []  # list of (entropy_scalar, img_path, img_np, gt_img, pred_img, entropy_color, base)
    with torch.no_grad():
        for i, batch in enumerate(tqdm(target_pts, desc='Computing entropy and collecting renders', ncols=100)):
            imgs, boxes, masks, img_paths = batch
            prompts = get_prompts(cfg, boxes, masks)

            embeds, masks_pred, _, _ = model(imgs, prompts)
            del _

            batch_size = imgs.shape[0]
            for b in range(batch_size):
                # Get image path and base name
                img_path = img_paths[b] if isinstance(img_paths, (list, tuple)) else img_paths
                base = os.path.splitext(os.path.basename(img_path))[0]
                
                # Compute entropy
                p_b = masks_pred[b].clamp(1e-6, 1 - 1e-6)
                entropy_map_b = - (p_b * torch.log(p_b) + (1 - p_b) * torch.log(1 - p_b))
                if entropy_map_b.ndim == 3:
                    entropy_map_b = entropy_map_b.max(dim=0)[0]
                entropy_scalar = float(entropy_map_b.mean().cpu().item())

                # Render images
                img_np = (imgs[b].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
                
                gt_mask = masks[b].cpu().numpy()
                if gt_mask.ndim == 3:
                    gt_mask = np.max(gt_mask, axis=0)
                gt_img = (gt_mask.astype(np.uint8) * 255)
                
                pred_mask = masks_pred[b].detach().cpu().numpy()
                if pred_mask.ndim == 3:
                    pred_mask = np.max(pred_mask, axis=0)
                pred_img = ((pred_mask > 0.5).astype(np.uint8) * 255)
                
                entropy_np = entropy_map_b.cpu().numpy()
                entropy_norm = (entropy_np - entropy_np.min()) / (entropy_np.max() - entropy_np.min() + 1e-6)
                cmap = cm.get_cmap("viridis")
                entropy_color = (cmap(entropy_norm)[:, :, :3] * 255).astype(np.uint8)
                
                collected.append((entropy_scalar, img_path, img_np, gt_img, pred_img, entropy_color, base))

    # Sort by entropy (lowest first)
    collected.sort(key=lambda x: x[0])
    print(f"Found {len(collected)} images, sorted by entropy")

    # Save sorted list
    list_path = os.path.join(save_dir, "sorted_list.txt")
    with open(list_path, 'w') as f：
        for rank, (entropy_scalar, img_path, _, _, _, _, _) in enumerate(collected, start=1):
            f.write(f"{rank:05d}\t{entropy_scalar:.6f}\t{img_path}\n")

    # Save images in sorted entropy order (low to high)
    print(f"Saving {len(collected)} images in entropy order (low → high)...")
    for rank, (entropy_scalar, img_path, img_np, gt_img, pred_img, entropy_color, base) in enumerate(collected, start=1):
        # Save all 4 images with rank prefix (from lowest to highest entropy)
        Image.fromarray(img_np).save(os.path.join(save_dir, f"{rank:05d}_{base}.jpg"))
        Image.fromarray(gt_img).save(os.path.join(save_dir, f"{rank:05d}_{base}_gt.jpg"))
        Image.fromarray(pred_img).save(os.path.join(save_dir, f"{rank:05d}_{base}_pred.jpg"))
        Image.fromarray(entropy_color).save(os.path.join(save_dir, f"{rank:05d}_{base}_en.jpg"))

