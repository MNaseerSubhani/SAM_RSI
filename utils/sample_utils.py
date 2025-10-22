import numpy as np
import torch
from sklearn.cluster import KMeans


def uniform_sampling(masks, N=1):
    n_points = []
    for i, mask in enumerate(masks):
        if not isinstance(mask, np.ndarray):
            mask = mask.cpu().numpy()
        indices = np.argwhere(mask == 1) # [y, x]
        sampled_indices = np.random.choice(len(indices), N, replace=True)
        sampled_points = np.flip(indices[sampled_indices], axis=1)
        n_points.append(sampled_points.tolist())

    return n_points


# import numpy as np

# def uniform_sampling(masks, N=1, pad_ratio=0.05):
#     """
#     Uniformly sample N points per mask while keeping a padding margin
#     from image borders (simulating human-like labeling).

#     Args:
#         masks: list of (H, W) binary masks (numpy arrays or torch tensors)
#         N: number of points to sample per mask
#         pad_ratio: percentage of H and W to exclude from edges
#     Returns:
#         n_points: list of [[x, y], ...] points for each mask
#     """
#     n_points = []

#     for mask in masks:
#         if not isinstance(mask, np.ndarray):
#             mask = mask.cpu().numpy()

#         h, w = mask.shape
#         pad_y, pad_x = int(h * pad_ratio), int(w * pad_ratio)

#         # get mask coordinates
#         indices = np.argwhere(mask == 1)  # [y, x]
#         if len(indices) == 0:
#             n_points.append([])  # empty mask
#             continue

#         # filter out edge-close points
#         inside = (
#             (indices[:, 0] >= pad_y) & (indices[:, 0] < h - pad_y) &
#             (indices[:, 1] >= pad_x) & (indices[:, 1] < w - pad_x)
#         )
#         indices = indices[inside]

#         # fallback if too few points after padding
#         if len(indices) == 0:
#             indices = np.argwhere(mask == 1)

#         sampled_indices = np.random.choice(len(indices), min(N, len(indices)), replace=True)
#         sampled_points = np.flip(indices[sampled_indices], axis=1)  # to [x, y]
#         n_points.append(sampled_points.tolist())

#     return n_points



def get_multi_distance_points(input_point, mask, points_nubmer):
    new_points = np.zeros((points_nubmer + 1, 2))
    new_points[0] = [input_point[1], input_point[0]]
    for i in range(points_nubmer):
        new_points[i + 1] = get_next_distance_point(new_points[:i + 1, :], mask)

    new_points = swap_xy(new_points)
    return new_points


def get_next_distance_point(input_points, mask):
    max_distance_point = [0, 0]
    max_distance = 0
    input_points = np.array(input_points)

    indices = np.argwhere(mask == True)
    for x, y in indices:
        # print(x,y,input_points)
        distance = np.sum(np.sqrt((x - input_points[:, 0]) ** 2 + (y - input_points[:, 1]) ** 2))
        if max_distance < distance:
            max_distance_point = [x, y]
            max_distance = distance
    return max_distance_point


def swap_xy(points):
    new_points = np.zeros((len(points),2))
    new_points[:,0] = points[:,1]
    new_points[:,1] = points[:,0]
    return new_points


def k_means_sampling(mask, k):
    points = np.argwhere(mask == 1) # [y, x]
    points = np.flip(points, axis=1)

    kmeans = KMeans(n_clusters=k)
    kmeans.fit(points)
    points = kmeans.cluster_centers_
    return points


def get_point_prompt_max_dist(masks, num_points):
    n_points = []
    for mask in masks:
        mask_np = mask.cpu().numpy()

        indices = np.argwhere(mask_np > 0)
        random_index = np.random.choice(len(indices), 1)[0]

        first_point = [indices[random_index][1], indices[random_index][0]]
        new_points = get_multi_distance_points(first_point, mask_np, num_points - 1)
        n_points.append(new_points)

    return n_points


def get_point_prompt_kmeans(masks, num_points):
    n_points = []
    for mask in masks:
        mask_np = mask.cpu().numpy()
        points = k_means_sampling(mask_np, num_points)
        n_points.append(points.astype(int))
    return n_points


def get_point_prompts(gt_masks, num_points):
    prompts = []
    # print('prompt',len(gt_masks))
    for mask in gt_masks:
        # print('prompt',len(mask))
        po_points = uniform_sampling(mask, num_points)
        # print('ori_po',po_points)
        na_points = uniform_sampling((~mask.to(bool)).to(float), num_points)
        # print('na_points',na_points)
        # print('na_points',na_points)
        po_point_coords = torch.tensor(po_points, device=mask.device)
        na_point_coords = torch.tensor(na_points, device=mask.device)
        # print('po_point_coords',po_point_coords.shape)
        # print('na_point_coords',na_point_coords.shape)
        point_coords = torch.cat((po_point_coords, na_point_coords), dim=1)
        po_point_labels = torch.ones(po_point_coords.shape[:2], dtype=torch.int, device=po_point_coords.device)
        na_point_labels = torch.zeros(na_point_coords.shape[:2], dtype=torch.int, device=na_point_coords.device)
        point_labels = torch.cat((po_point_labels, na_point_labels), dim=1)
        in_points = (point_coords, point_labels)
        prompts.append(in_points)
    return prompts