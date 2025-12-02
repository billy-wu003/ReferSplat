import os
import torch
import matplotlib.pyplot as plt
from random import randint
from utils.loss_utils import l1_loss, ssim,bce_loss,dice_loss,multi_pos_cross_entropy
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
import torch.nn.functional as F
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams


def compute_visibility_labels(gaussians, camera, gt_mask, visibility_filter):
    x, y, w = gaussians.project_to_pixels(camera)
    x_idx = x.long()
    y_idx = y.long()
    in_bounds = (x_idx >= 0) & (x_idx < camera.image_width) & (y_idx >= 0) & (y_idx < camera.image_height)
    in_front = w > 0
    valid_mask = visibility_filter & in_bounds & in_front
    labels = torch.zeros_like(x, device="cuda", dtype=torch.float32)
    if valid_mask.any():
        labels[valid_mask] = gt_mask[0, y_idx[valid_mask], x_idx[valid_mask]].float()
    return valid_mask, labels


def compute_smooth_loss(gaussians, semantic_logits, visibility_mask, k, delta_n):
    visible_idx = visibility_mask.nonzero(as_tuple=False).squeeze(-1)
    if visible_idx.numel() <= 1:
        return torch.zeros(1, device="cuda")

    xyz_visible = gaussians.get_xyz[visible_idx].detach()
    semantic_visible = semantic_logits[visible_idx]
    colors_dc = gaussians._features_dc[visible_idx, :3, 0].detach()

    # k-NN on visible Gaussians
    k = min(k + 1, xyz_visible.shape[0])
    distances = torch.cdist(xyz_visible, xyz_visible)
    knn_idx = torch.topk(distances, k=k, dim=1, largest=False).indices[:, 1:]

    sigma_i = semantic_visible.unsqueeze(1).expand(-1, knn_idx.shape[1])
    sigma_j = semantic_visible[knn_idx]

    color_diff = torch.norm(colors_dc.unsqueeze(1) - colors_dc[knn_idx], dim=-1)
    weight = torch.exp(-color_diff / max(delta_n, 1e-6))

    return (weight * torch.abs(sigma_i - sigma_j)).mean()
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
    

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from,epoch):
    
    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    if opt.include_feature:
        if not checkpoint:
            raise ValueError("checkpoint missing!!!!!")
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        if len(model_params) == 12 and opt.include_feature:
            first_iter = 0
        gaussians.restore(model_params, opt)
        
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color,dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, 100000), desc="Training progress")
    first_iter += 1
    iteration = first_iter
    ratio=0.1
    total_loss=[]
    iteration=1
    for epoch in range(epoch_num):
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        while len(viewpoint_stack)!=0:
            viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
            text_feature=gaussians.get_text(viewpoint_cam.sentence).to("cuda")
            for i in range(len(viewpoint_cam.sentence)):
                iter_start.record()
                render_pkg = render(viewpoint_cam, gaussians, pipe, background, opt,sentence=viewpoint_cam.sentence[i],ratio=ratio)
                language_feature,mean_tensor=render_pkg["language_feature_image"],render_pkg["mean_tensor"]
                if opt.include_feature:
                    features=gaussians.mlp1(text_feature)
                    features=torch.mean(features, dim=1)
                    mean_tensor=F.normalize(mean_tensor,dim=1)
                    features=F.normalize(features,dim=1)

                    cosine_similarities=(torch.matmul(mean_tensor,features.T)/0.1).to("cuda")
                    
                    sentence_tensor = torch.zeros(len(viewpoint_cam.sentence))
                    
                    sentence_tensor[i] = 1
                    current_category = viewpoint_cam.category[i]
                    category_indices = [idx for idx, cat in enumerate(viewpoint_cam.category) if cat == current_category]
                    sentence_tensor[category_indices] = 1
                    sentence_tensor = sentence_tensor.unsqueeze(0).to("cuda")
                    com_loss = multi_pos_cross_entropy(cosine_similarities, sentence_tensor)
                    gt_mask = viewpoint_cam.gt_mask[viewpoint_cam.category[i]].to("cuda")
                    visibility_mask, view_labels = compute_visibility_labels(gaussians, viewpoint_cam, gt_mask, render_pkg["visibility_filter"])
                    gaussians.update_vote_statistics(visibility_mask, view_labels, gamma_p=opt.vote_gamma_p, gamma_m=opt.vote_gamma_m)
                    hard_labels, valid_mask = gaussians.get_hard_labels(opt.vote_mmin)
                    soft_labels = gaussians.get_soft_labels(opt.vote_eps)
                    semantic_seed = gaussians.mlp2(gaussians._language_feature)
                    semantic_logits = gaussians.semantic_head(semantic_seed).squeeze(-1)
                    smooth_loss = compute_smooth_loss(
                        gaussians,
                        semantic_logits,
                        visibility_mask,
                        opt.smooth_k,
                        opt.smooth_delta_n,
                    )
                    if valid_mask.any():
                        loss_hard = F.binary_cross_entropy_with_logits(semantic_logits[valid_mask], hard_labels[valid_mask])
                        loss_soft = F.binary_cross_entropy_with_logits(semantic_logits[valid_mask], soft_labels[valid_mask])
                    else:
                        loss_hard = torch.zeros(1, device="cuda")
                        loss_soft = torch.zeros(1, device="cuda")
                    loss = (
                        bce_loss(language_feature, gt_mask)
                        + 0.1 * com_loss
                        + opt.lambda_3d * loss_hard
                        + opt.lambda_3d_soft * loss_soft
                        + opt.lambda_smooth * smooth_loss
                    )
                    loss.backward()
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)
                iter_end.record()
                iteration+=1
                if iteration%2000==0 and ratio>0.005:
                    ratio=ratio*0.6
                    if ratio<0.005:
                        ratio=0.005
                with torch.no_grad():
                    ema_loss_for_log = 0.4*loss.item()+0.6*ema_loss_for_log
                    if iteration % 10 == 0:
                        progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                        progress_bar.update(10)
                        total_loss.append(ema_loss_for_log)
        
        torch.save((gaussians.capture(opt.include_feature), iteration), scene.model_path + "/chkpnt_cbasetea251" + str(epoch) + ".pth")
    progress_bar.close()
    
if __name__ == "__main__":
    # Set up command line argument parser
    torch.set_default_dtype(torch.float32)
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=55555)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = 'output/teatime/chkpnt30000.pth')
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    print(args)
    args.model_path = args.model_path
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    epoch_num=5
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from,epoch_num)

    print("\nTraining complete.")