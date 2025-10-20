# eval_mesh.py
import os
import os.path as osp

import numpy as np
import pymeshlab
import torch
import torch.nn.functional as F
import trimesh
from scipy.spatial import cKDTree as KDTree
from utils import (
    f_from_m,
    get_closest_f_dist,
    get_closest_f_ids,
    get_face_normals,
    numpied,
    scale_m_to_unit_cube,
    scale_to_unit_sphere,
    torched,
    trimesh_from_vf,
    v_from_m,
    get_usdf_py3d,
    get_close_grid_idx,
    scale_m_to_unit_sphere,
)


def sample(mesh, n_points=100000):
    """Sample points evenly from a mesh surface."""
    # attempt to sample points evenly 3 times if the 1rst time fails
    np.random.seed(0)
    samples = trimesh.sample.sample_surface_even(mesh, n_points)[0]
    i = 10000
    for n in range(3):
        if len(samples) < n_points:
            samples = trimesh.sample.sample_surface_even(mesh, n_points + i * (n + 1))[
                0
            ]
        if len(samples) > n_points:
            idx = np.random.choice(len(samples), n_points, replace=False)
            samples = samples[idx]
    if len(samples) < n_points:
        rest_samples = trimesh.sample.sample_surface(mesh, n_points - len(samples))[0]
        samples = np.concatenate([samples, rest_samples])
    return samples


def chamfer_distance(sA, sB):
    """Calculate the Chamfer distance between two point clouds."""
    treeA = KDTree(sA)
    treeB = KDTree(sB)
    distB2A = treeA.query(sB)[0]
    distA2B = treeB.query(sA)[0]
    cd = np.mean(distA2B) + np.mean(distB2A)
    return cd / 2


def eval_p2f_dist(pred_m, gt_m):
    """Evaluate the point-to-face distance between a predicted and ground truth mesh."""
    pred_m_sc = trimesh_from_vf(
        scale_to_unit_sphere(v_from_m(pred_m)), f_from_m(pred_m)
    )
    pred_pts = trimesh.sample.sample_surface_even(pred_m_sc, 100000)[0]
    dist = get_closest_f_dist(
        torched(pred_pts),
        torched(scale_to_unit_sphere(v_from_m(gt_m)))[
            torched(gt_m.faces, dtype=torch.int64)
        ],
    )
    return dist.mean().item()


def eval_normals(pred_m, gt_m):
    """Evaluate the normal consistency between a predicted and ground truth mesh."""
    pred_m_sc = scale_m_to_unit_cube(pred_m)
    gt_m_sc = scale_m_to_unit_cube(gt_m)

    gt_v, gt_f = v_from_m(gt_m_sc), f_from_m(gt_m_sc)
    pred_v, pred_f = v_from_m(pred_m_sc), f_from_m(pred_m_sc)
    pred_v_th = torched(np.array(pred_v)).float()

    pred_trig_th = torched(np.array(pred_v[pred_f])).float()
    gt_nn_f_ids = get_closest_f_ids(torched(gt_v), pred_trig_th)
    gt_nn_f_ids = numpied(gt_nn_f_ids)
    pred_closest_norm = numpied(
        get_face_normals(pred_v_th, torched(pred_f).long())[gt_nn_f_ids]
    )
    gt_v_norm = np.array(gt_m_sc.vertex_normals)
    sim_1 = F.cosine_similarity(torched(pred_closest_norm), torched(gt_v_norm))
    sim_2 = F.cosine_similarity(torched(pred_closest_norm), torched(-gt_v_norm))
    pi_sim = torch.max(sim_1, sim_2)
    dis_sim = 1.0 - pi_sim
    return 100 * dis_sim.mean().item()


def eval_nmf(mesh, gt_m=None):
    """Evaluate the non-manifoldness of a mesh."""
    mesh = trimesh.Trimesh(
        v_from_m(mesh),
        np.unique(np.sort(np.array(mesh.faces), axis=1), axis=0),
        process=False,
    )
    ordered_edges = [np.sort(x) for x in mesh.edges]
    uni, count = np.unique(ordered_edges, return_counts=True, axis=0)
    edge_manifoldness = len(np.argwhere(count == 2)) / float(len(count))
    return 1 - edge_manifoldness


def scipy_CD(sA, sB):
    """Calculate Chamfer distance using scipy's KDTree."""
    treeA = KDTree(sA)
    treeB = KDTree(sB)
    distB2A = treeA.query(sB)[0]
    distA2B = treeB.query(sA)[0]
    cd = np.mean(distA2B) + np.mean(distB2A)
    return cd / 2


def eval_edgeCD(pred_m, gt_m, samp_per_e=10):
    """Evaluate the Chamfer distance between edges of predicted and ground truth meshes."""
    pred_m_sc = scale_m_to_unit_sphere(pred_m)
    gt_m_sc = scale_m_to_unit_sphere(gt_m)
    pred_e_samp = sample_along_edge(
        v_from_m(pred_m_sc), pred_m_sc.edges_unique, samp_per_e
    )
    gt_e_samp = sample_along_edge(v_from_m(gt_m_sc), gt_m_sc.edges_unique, samp_per_e)
    return scipy_CD(pred_e_samp, gt_e_samp)


def sample_along_edge(verts, edges_unique, samp_per_e=5):
    """Sample points along the edges of a mesh."""
    lines = verts[edges_unique]
    num_fracs = lines.shape[0] * samp_per_e
    line_frac = np.random.uniform(0.1, 0.9, size=num_fracs).reshape(-1, samp_per_e, 1)
    complim_frac = 1.0 - line_frac
    pts = line_frac * lines[:, 0:1, :] + complim_frac * lines[:, 1:2, :]
    return pts.reshape(-1, 3)


def eval_CD(pred_m, gt_m):
    """Evaluate the Chamfer distance between a predicted and ground truth mesh."""
    pred_m_sc = scale_m_to_unit_sphere(pred_m)
    gt_m_sc = scale_m_to_unit_sphere(gt_m)

    pred_samp = sample(pred_m_sc)
    gt_samp = sample(gt_m_sc)

    return chamfer_distance(pred_samp, gt_samp)


def eval_intersec(pred_m, gt_m):
    """Evaluate the percentage of self-intersecting faces in a mesh."""
    for i in range(10):
        cur_name = "temp%d.ply" % i
        if not osp.isfile(cur_name):
            break
    _ = pred_m.export(cur_name)
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(cur_name)
    ms.compute_selection_by_self_intersections_per_face()
    int_f_id = ms.current_mesh().face_selection_array()
    v_f = ms.current_mesh().face_matrix()
    num_intersec = v_f[int_f_id].shape[0]
    num_face = v_f.shape[0]
    percentage = (num_intersec / num_face) * 100.0
    os.remove(cur_name)
    return percentage


def eval_small_trig(pred_m, gt_m, thresh=150.0):
    """Evaluate the percentage of small triangles in a mesh."""
    m_tri = v_from_m(pred_m)[f_from_m(pred_m)]
    m_tri_roll = np.roll(m_tri, 1, axis=1)
    tri_e_vec = m_tri_roll - m_tri
    tri_e_vec_u = tri_e_vec / (np.linalg.norm(tri_e_vec, axis=-1, keepdims=True) + 1e-7)
    tri_e_idx = [[1, 0], [2, 1], [0, 2]]
    all_angle = []
    for e_idx in tri_e_idx:
        all_angle.append(trimesh.geometry.vector_angle(tri_e_vec_u[:, e_idx, :]))

    all_angle = np.array(all_angle).T
    total_e = np.product(np.array(all_angle).shape)
    angle_bw = np.rad2deg(np.pi) - np.rad2deg(np.array(all_angle))
    bad_ratio = (angle_bw > 120.0).nonzero()[0].shape[0] / total_e
    return bad_ratio * 100.0


def eval_all(pred_m, gt_m):
    """Evaluate a comprehensive set of metrics for a predicted mesh against a ground truth mesh."""
    metrics = ["CD", "NMF", "Normal", "Intersec", "Small Trig", "P2F", "EdgeCD"]
    metric_funcs = [
        eval_CD,
        eval_nmf,
        eval_normals,
        eval_intersec,
        eval_small_trig,
        eval_p2f_dist,
        eval_edgeCD,
    ]
    out_dict = {}
    for metric, func in zip(metrics, metric_funcs):
        out_dict[metric] = func(pred_m, gt_m)
    return out_dict


def eval_mini(pred_m, gt_m):
    """Evaluate a smaller set of metrics for a predicted mesh against a ground truth mesh."""
    metrics = ["CD", "Normal", "Intersec", "Small Trig", "EdgeCD"]
    metric_funcs = [eval_CD, eval_normals, eval_intersec, eval_small_trig, eval_edgeCD]
    out_dict = {}
    for metric, func in zip(metrics, metric_funcs):
        out_dict[metric] = func(pred_m, gt_m)
    return out_dict


def eval_dummy():
    """Return a dictionary of dummy evaluation values."""
    values = -1
    keys = ["CD", "NMF", "Normal", "Intersec", "Small Trig"]
    out_dict = dict.fromkeys(keys, values)
    return out_dict


def eval_cl_SDF(pred_m_glob, gt_sdf, grid_coord, grid_coord_us):
    """Evaluate the SDF consistency of a predicted mesh."""
    pred_v_glob_c = torched(v_from_m(pred_m_glob))
    pred_v_sus = scale_to_unit_sphere(v_from_m(pred_m_glob))
    pred_f = torched(f_from_m(pred_m_glob)).long()
    assert grid_coord.ndim == 4
    vox_size = grid_coord.shape[0] - 1
    assert pred_v_glob_c.min() >= 0 and pred_v_glob_c.max() <= vox_size
    close_idx = get_close_grid_idx(grid_coord, pred_v_glob_c)
    grid_xyz_vox_close = grid_coord_us[
        close_idx[:, 0], close_idx[:, 1], close_idx[:, 2]
    ]
    p2f = get_usdf_py3d(
        grid_xyz_vox_close.reshape(-1, 3),
        torched(pred_v_sus).view(-1, 3),
        pred_f.view(-1, 3),
    )
    gt_sdf_close = gt_sdf[close_idx[:, 0], close_idx[:, 1], close_idx[:, 2]]
    return (
        ((p2f.squeeze().abs() - gt_sdf_close.squeeze().abs()) ** 2).mean().sqrt().item()
    )
