import torch
import torch.nn.functional as F
import torch.nn as nn
from pytorch3d.ops.knn import knn_points
from torch import ones_like
from dc_utils import quad_from_grid_DC


def gradient(y, x, grad_outputs=None):
    """Compute the gradient of y with respect to x."""
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def get_poly_adj_edge(pred_v, quad_f):
    """
    Returns the edge vectors and adjacent as tuple.
    """
    quad_verts = pred_v[quad_f]
    rolled_vert = quad_verts.roll(1, 1)
    e_vec = rolled_vert - quad_verts
    e_vec_unit = e_vec / (e_vec.norm(dim=-1, keepdim=True) + 1e-5)
    return e_vec_unit, e_vec_unit.roll(1, 1)


def get_quad_angles(pred_v, crossing_vox, vox_ind, valid_cell_id, n_verts):
    """Calculate the angles of quads from a voxel grid."""
    quads = quad_from_grid_DC(
        crossing_vox, vox_ind, valid_cell_id, n_verts, vox_size=65
    )
    quad_e1, quad_e2 = get_poly_adj_edge(pred_v, quads)
    quad_angle = torch.arccos(torch.einsum("bij,bij->bi", quad_e2, quad_e1))
    # Take compliment of angle
    quad_angle = ones_like(quad_angle) * torch.pi - quad_angle
    return quad_angle


def trig_angle_loss(pred_v, pred_f):
    """Compute a loss to encourage triangles to be equilateral."""
    if pred_v.ndim > 2:
        raise ValueError("pred_v should be a 2D tensor")
    vec_1, vec_2 = get_poly_adj_edge(pred_v, pred_f)
    adj_angles = torch.arccos(torch.einsum("bij,bij->bi", vec_2, vec_1))
    adj_angles = ones_like(adj_angles) * torch.pi - adj_angles
    angle_disp = (
        (torch.cos(adj_angles) - torch.cos(ones_like(adj_angles) * torch.pi / 3))
        .abs()
        .mean()
    )
    return angle_disp


class soft_L1(nn.Module):
    """A soft L1 loss that is zero for small errors."""

    def __init__(self):
        super(soft_L1, self).__init__()

    def forward(self, input, target, eps=0.0):
        ret = torch.abs(input - target) - eps
        ret = torch.clamp(ret, min=0.0, max=100.0)
        return ret


def bi_dir_cd(pred, gt):
    """Compute bidirectional Chamfer distance."""
    x_nn = knn_points(pred, gt, norm=2, K=1)
    y_nn = knn_points(gt, pred, norm=2, K=1)

    cham_x = x_nn.dists[..., 0].squeeze().sqrt().mean()
    cham_y = y_nn.dists[..., 0].squeeze().sqrt().mean()
    bi_cd = (cham_x + cham_y) / 2
    return bi_cd
