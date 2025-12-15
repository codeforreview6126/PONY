import torch
import torch.nn.functional as F

def gradcam_pointnet2(model, xyz_rgb, extra_features=None, target_layer="sa1"):
    model.eval()
    feats = {}
    grads = {}

    def save_feats(module, input, output):
        # output = (new_xyz, new_points, fps_idx)
        feats['value'] = output[1].detach()  # new_points

    def save_grads(module, grad_input, grad_output):
        # grad_output[1] corresponds to new_points gradient
        # sometimes grad_output is a tuple with one element
        try:
            grads['value'] = grad_output[1].detach()  # gradient w.r.t new_points
        except IndexError:
            grads['value'] = grad_output[0].detach()

    hook_layer = getattr(model, target_layer)
    fwd_hook = hook_layer.register_forward_hook(save_feats)
    bwd_hook = hook_layer.register_full_backward_hook(save_grads)

    # Forward pass
    pred, _ = model(xyz_rgb, extra_features)

    # Backward pass
    model.zero_grad()
    pred.sum().backward()

    fwd_hook.remove()
    bwd_hook.remove()

    # Check that grads were captured
    if 'value' not in grads:
        raise RuntimeError(f"Gradients not captured for layer {target_layer}")

    # GradCAM computation
    weights = grads['value'].mean(dim=2)  # (B, C)
    cam = (weights.unsqueeze(2) * feats['value']).sum(dim=1)  # (B, N)    
    cam = F.relu(cam)
    
    # Map back to original coordinates
    fps_idx = model.fps_indices[target_layer]
    coords = xyz_rgb[:, :3, :].permute(0, 2, 1)
    coords_sampled = torch.stack([coords[b, fps_idx[b]] for b in range(coords.size(0))], dim=0)
    heatmap_out = torch.cat([coords_sampled, cam.unsqueeze(-1)], dim=-1)

    return heatmap_out, pred