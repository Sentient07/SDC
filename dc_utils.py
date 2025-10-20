# dc_utils.py

import time
import torch
import numpy as np
from torch.cuda import LongTensor as cLT
from torch import LongTensor as LT
from torch import logical_and as LAnd
from torch import zeros_like, ones_like, zeros, ones
from utils import custom_sample_trig, numpied, zero_pad_sdf
from torch.cuda import LongTensor
from torch import logical_or
from utils import manual_vox_ind


def get_cell_v_idx(vox_ind):
    """Gets the vertex indices for each cell in a voxel grid."""
    disp_ind = (
        torch.Tensor(
            [
                [0, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
                [0, 1, 1],
                [1, 0, 0],
                [1, 0, 1],
                [1, 1, 0],
                [1, 1, 1],
            ]
        )
        .to(vox_ind.device)
        .long()
    )
    return vox_ind.reshape(-1, 3).unsqueeze(1) + disp_ind


def get_valid_face_id(crossing_vox):
    """Filters face IDs to find valid ones based on their position."""
    cross_nz = crossing_vox.nonzero().squeeze()
    vox_size = crossing_vox.shape[0]
    cr_range = vox_size - 1

    x_cond = (
        (cross_nz[:, 1] > 0)
        * (cross_nz[:, 2] > 0)
        * (cross_nz[:, 1] < cr_range)
        * (cross_nz[:, 2] < cr_range)
    )
    y_cond = (
        (cross_nz[:, 0] > 0)
        * (cross_nz[:, 2] > 0)
        * (cross_nz[:, 0] < cr_range)
        * (cross_nz[:, 2] < cr_range)
    )
    z_cond = (
        (cross_nz[:, 1] > 0)
        * (cross_nz[:, 0] > 0)
        * (cross_nz[:, 1] < cr_range)
        * (cross_nz[:, 0] < cr_range)
    )
    x_idx_mask = LAnd(cross_nz[:, 3] == 0, x_cond)
    y_idx_mask = LAnd(cross_nz[:, 3] == 1, y_cond)
    z_idx_mask = LAnd(cross_nz[:, 3] == 2, z_cond)

    cross_nz_x_ind = cross_nz[x_idx_mask][:, :3]
    cross_nz_y_ind = cross_nz[y_idx_mask][:, :3]
    cross_nz_z_ind = cross_nz[z_idx_mask][:, :3]

    return cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind


def fast_face_construct(crossing_vox, vert_vox_idx):
    """Constructs mesh faces from voxel crossings using long indices."""
    cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind = get_valid_face_id(crossing_vox)
    x_offset_trig_1 = LT([[0, -1, -1], [0, 0, -1], [0, 0, 0]]).to(crossing_vox.device)
    x_offset_trig_2 = LT([[0, -1, -1], [0, 0, 0], [0, -1, 0]]).to(crossing_vox.device)

    y_offset_trig_1 = LT([[-1, 0, -1], [0, 0, 0], [0, 0, -1]]).to(crossing_vox.device)
    y_offset_trig_2 = LT([[-1, 0, -1], [-1, 0, 0], [0, 0, 0]]).to(crossing_vox.device)

    z_offset_trig_1 = LT([[-1, -1, 0], [0, 0, 0], [-1, 0, 0]]).to(crossing_vox.device)
    z_offset_trig_2 = LT([[-1, -1, 0], [0, -1, 0], [0, 0, 0]]).to(crossing_vox.device)

    offset_1 = [x_offset_trig_1, y_offset_trig_1, z_offset_trig_1]
    offset_2 = [x_offset_trig_2, y_offset_trig_2, z_offset_trig_2]
    cross_dir = [cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind]
    all_trig = []
    for t1_idx, t2_idx, cross_ in zip(offset_1, offset_2, cross_dir):
        adj_ind_1 = (cross_.unsqueeze(1) + t1_idx).view(-1, 3)
        adj_ind_2 = (cross_.unsqueeze(1) + t2_idx).view(-1, 3)

        trig_1 = vert_vox_idx[
            adj_ind_1[:, 0], adj_ind_1[:, 1], adj_ind_1[:, 2]
        ].reshape(-1, 3)
        trig_2 = vert_vox_idx[
            adj_ind_2[:, 0], adj_ind_2[:, 1], adj_ind_2[:, 2]
        ].reshape(-1, 3)
        all_trig.extend([trig_1, trig_2])

    return torch.cat(all_trig)


def fast_face_construct_proba(crossing_vox, vert_vox_idx, vox_proba):
    """Constructs mesh faces with associated probabilities."""
    cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind = get_valid_face_id(crossing_vox)
    x_offset_trig_1 = LT([[0, -1, -1], [0, 0, -1], [0, 0, 0]]).to(crossing_vox.device)
    x_offset_trig_2 = LT([[0, -1, -1], [0, 0, 0], [0, -1, 0]]).to(crossing_vox.device)

    y_offset_trig_1 = LT([[-1, 0, -1], [0, 0, 0], [0, 0, -1]]).to(crossing_vox.device)
    y_offset_trig_2 = LT([[-1, 0, -1], [-1, 0, 0], [0, 0, 0]]).to(crossing_vox.device)

    z_offset_trig_1 = LT([[-1, -1, 0], [0, 0, 0], [-1, 0, 0]]).to(crossing_vox.device)
    z_offset_trig_2 = LT([[-1, -1, 0], [0, -1, 0], [0, 0, 0]]).to(crossing_vox.device)

    offset_1 = [x_offset_trig_1, y_offset_trig_1, z_offset_trig_1]
    offset_2 = [x_offset_trig_2, y_offset_trig_2, z_offset_trig_2]
    cross_dir = [cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind]
    all_trig = []
    all_proba = []
    for c_dir, (t1_idx, t2_idx, cross_) in enumerate(
        zip(offset_1, offset_2, cross_dir)
    ):
        adj_ind_1 = (cross_.unsqueeze(1) + t1_idx).view(-1, 3)
        adj_ind_2 = (cross_.unsqueeze(1) + t2_idx).view(-1, 3)

        trig_1 = vert_vox_idx[
            adj_ind_1[:, 0], adj_ind_1[:, 1], adj_ind_1[:, 2]
        ].reshape(-1, 3)
        trig_2 = vert_vox_idx[
            adj_ind_2[:, 0], adj_ind_2[:, 1], adj_ind_2[:, 2]
        ].reshape(-1, 3)
        trig_alt_cat = (
            zeros((trig_1.shape[0] + trig_2.shape[0], 3)).to(trig_1.device).long()
        )
        trig_alt_cat[::2, :] = trig_1
        trig_alt_cat[1::2, :] = trig_2
        cur_proba = vox_proba[cross_[:, 0], cross_[:, 1], cross_[:, 2], c_dir]
        all_proba.append(cur_proba)
        all_proba.append(cur_proba)
        all_trig.extend([trig_alt_cat])

    return torch.cat(all_trig), torch.cat(all_proba)


def fast_quad_construct(crossing_vox, vert_vox_idx):
    """Constructs quads from voxel crossings."""
    cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind = get_valid_face_id(crossing_vox)

    x_offset_q = LongTensor([[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]]).cuda()

    y_offset_q = LongTensor([[-1, 0, -1], [0, 0, -1], [0, 0, 0], [-1, 0, 0]]).cuda()

    z_offset_q = LongTensor([[-1, -1, 0], [0, -1, 0], [0, 0, 0], [-1, 0, 0]]).cuda()

    offsets = [x_offset_q, y_offset_q, z_offset_q]
    cross_dir = [cross_nz_x_ind, cross_nz_y_ind, cross_nz_z_ind]
    all_q = []
    for q_idx, cross_ in zip(offsets, cross_dir):
        adj_q_ind = (cross_.unsqueeze(1) + q_idx).view(-1, 3)
        if adj_q_ind.max() > vert_vox_idx.shape[0] - 1:
            raise ValueError("Index out of bounds")

        q_1 = vert_vox_idx[adj_q_ind[:, 0], adj_q_ind[:, 1], adj_q_ind[:, 2]].reshape(
            -1, 4
        )
        all_q.extend([q_1])

    return torch.cat(all_q)


def f_cross_from_grid_DC(sdf_vox):
    """Gets face crossings from a grid using dual contouring."""
    grid_size = sdf_vox.shape[1]
    vox_ind = manual_vox_ind(grid_size).cuda().long()
    cell_v_idx_th = get_cell_v_idx(vox_ind)
    cross_e, cross_cell = get_e_cell_intersec(sdf_vox, vox_ind)
    valid_cell_id = cross_cell.squeeze().sum(dim=-1) > 0
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    vert_vox_id = ones((grid_size, grid_size, grid_size)).to(sdf_vox.device).long() * -1
    vert_vox_id[cell_bot_id[:, 0], cell_bot_id[:, 1], cell_bot_id[:, 2]] = torch.arange(
        cell_bot_id.shape[0]
    ).to(sdf_vox.device)
    trigs = fast_face_construct(cross_e, vert_vox_id)
    return trigs, valid_cell_id


def trig_from_grid_DC(
    pred_v_clamp,
    crossing_edge_vox,
    vox_ind,
    valid_int_mask,
    vox_size,
    return_quads=False,
):
    """Extracts triangles from a grid using dual contouring."""
    v_ind = vox_ind[valid_int_mask]
    vert_vox_id = (
        ones((vox_size, vox_size, vox_size)).to(pred_v_clamp.device).long() * -1
    )
    vert_vox_id[v_ind[:, 0], v_ind[:, 1], v_ind[:, 2]] = torch.arange(
        pred_v_clamp.shape[0]
    ).to(pred_v_clamp.device)
    quads = None
    trigs = fast_face_construct(crossing_edge_vox, vert_vox_id)

    if return_quads:
        quads = fast_quad_construct(crossing_edge_vox, vert_vox_id)
        return trigs, quads

    return trigs


def quad_from_grid_DC(crossing_vox, vox_ind, valid_cell_id, n_verts, vox_size=64):
    """Extracts quads from a grid using dual contouring."""
    v_ind = vox_ind[valid_cell_id]
    vert_vox_id = (
        ones((vox_size, vox_size, vox_size)).to(crossing_vox.device).long() * -1
    )
    vert_vox_id[v_ind[:, 0], v_ind[:, 1], v_ind[:, 2]] = torch.arange(n_verts).to(
        crossing_vox.device
    )
    quads = fast_quad_construct(crossing_vox, vert_vox_id)
    return quads


def get_indiv_intersec(indicator_sdf_fdc):
    """Gets individual intersections along each axis of an SDF grid."""
    x_neigh_prod = indicator_sdf_fdc[:-1, :, :] * indicator_sdf_fdc[1:, :, :]
    y_neigh_prod = indicator_sdf_fdc[:, :-1, :] * indicator_sdf_fdc[:, 1:, :]
    z_neigh_prod = indicator_sdf_fdc[:, :, :-1] * indicator_sdf_fdc[:, :, 1:]
    return [x_neigh_prod, y_neigh_prod, z_neigh_prod]


def get_indiv_intersec_bat(grid_sdf):
    """Gets individual intersections for a batch of SDF grids."""
    x_neigh_prod = grid_sdf[:, :-1, :, :] * grid_sdf[:, 1:, :, :]
    y_neigh_prod = grid_sdf[:, :, :-1, :] * grid_sdf[:, :, 1:, :]
    z_neigh_prod = grid_sdf[:, :, :, :-1] * grid_sdf[:, :, :, 1:]
    return [x_neigh_prod, y_neigh_prod, z_neigh_prod]


def get_cell_e_idx(vox_ind):
    """Constructs indices that denote the edges for each cell."""
    x_ind = (
        torch.Tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1]])
        .to(vox_ind.device)
        .long()
    )
    y_ind = torch.roll(x_ind, 1, 1)
    y_ind[[1, 2], :] = y_ind[[2, 1], :]
    z_ind = torch.roll(x_ind, 2, 1)
    x_edge_ind_fl = (vox_ind.reshape(-1, 3).unsqueeze(1) + x_ind).reshape(-1, 3)
    y_edge_ind_fl = (vox_ind.reshape(-1, 3).unsqueeze(1) + y_ind).reshape(-1, 3)
    z_edge_ind_fl = (vox_ind.reshape(-1, 3).unsqueeze(1) + z_ind).reshape(-1, 3)
    return [x_edge_ind_fl, y_edge_ind_fl, z_edge_ind_fl]


def cell_from_vox(voxel, vox_ind):
    """Constructs cell edges from a voxel grid."""
    all_cell_ind = get_cell_e_idx(vox_ind)
    all_cell = []
    for i, dir_ind in enumerate(all_cell_ind):
        cell_dir = voxel[dir_ind[:, 0], dir_ind[:, 1], dir_ind[:, 2], i].reshape(
            vox_ind.shape[0], 4, -1
        )
        all_cell.append(cell_dir)
    return torch.cat(all_cell, dim=1)


def cell_from_vox_bat(voxel, vox_ind):
    """Constructs cell edges for a batch of voxel grids."""
    all_cell_ind = get_cell_e_idx(vox_ind)
    all_cell = []
    n_batch = voxel.shape[0]
    for i, dir_ind in enumerate(all_cell_ind):
        bat_ind = (
            torch.arange(n_batch)
            .to(voxel.device)
            .long()
            .repeat_interleave(dir_ind[:, 0].shape[0])
        )
        dir_ind_rep = dir_ind.unsqueeze(0).repeat(n_batch, 1, 1).reshape(-1, 3)
        cell_dir = voxel[
            bat_ind, dir_ind_rep[:, 0], dir_ind_rep[:, 1], dir_ind_rep[:, 2], i
        ].reshape(n_batch, vox_ind.shape[0], 4, -1)
        all_cell.append(cell_dir)
    return torch.cat(all_cell, dim=2)


def get_e_cell_intersec(grid_sdf, vox_ind):
    """Gets edge intersections indexed by cell."""
    all_neigh_product = get_indiv_intersec(grid_sdf)
    vox_size = grid_sdf.shape[0]
    vox_edge_cross = zeros((vox_size, vox_size, vox_size) + (3,)).to(grid_sdf.device)
    for ind, neigh_prod in enumerate(all_neigh_product):
        cross_ = (
            torch.where(neigh_prod > 0.0, zeros_like(neigh_prod), ones_like(neigh_prod))
            .long()
            .nonzero()
        )
        vox_edge_cross[cross_[:, 0], cross_[:, 1], cross_[:, 2], ind] = 1
    return vox_edge_cross, cell_from_vox(vox_edge_cross, vox_ind)


def get_e_cell_intersec_bat(grid_sdf, vox_ind):
    """Gets edge intersections for a batch of cells."""
    all_neigh_product = get_indiv_intersec_bat(grid_sdf)
    vox_size = grid_sdf.shape[1]
    vox_edge_cross = zeros((grid_sdf.shape[0], vox_size, vox_size, vox_size) + (3,)).to(
        grid_sdf.device
    )
    for ind, neigh_prod in enumerate(all_neigh_product):
        cross_ = (
            torch.where(neigh_prod > 0.0, zeros_like(neigh_prod), ones_like(neigh_prod))
            .long()
            .nonzero()
        )
        vox_edge_cross[cross_[:, 0], cross_[:, 1], cross_[:, 2], cross_[:, 3], ind] = 1
    return vox_edge_cross, cell_from_vox_bat(vox_edge_cross, vox_ind)


def get_intersec(grid_sdf, vox_ind):
    """Gets all intersections from an SDF grid."""
    cross_e, cross_cell = get_e_cell_intersec(grid_sdf, vox_ind)
    valid_cell_id = cross_cell.squeeze().sum(dim=-1) > 0
    return cross_e, valid_cell_id


def pred_v_grid_bat(grid_sdf, decoder, valid_cell_id):
    """Predicts vertices for a batch of grids."""
    n_batch = grid_sdf.shape[0]
    pred_pos_full = decoder(grid_sdf.unsqueeze(1)).squeeze()
    pred_pos_flat = pred_pos_full.reshape(n_batch, -1, 3)[
        valid_cell_id[:, 0], valid_cell_id[:, 1]
    ]
    return pred_pos_flat, pred_pos_full


def pred_v_grid(grid_sdf, decoder, valid_cell_id):
    """Predicts vertices for a single grid."""
    pred_pos_full = decoder(grid_sdf.unsqueeze(0).unsqueeze(1))
    pred_pos = pred_pos_full.squeeze()[:-1, :-1, :-1].reshape(-1, 3)
    pred_pos_sel = pred_pos[valid_cell_id]
    return pred_pos_sel, pred_pos_full


def pad_and_predict_vert(grid_sdf, decoder, valid_cell_id, cell_v_idx_th, pad=3):
    """Pads an SDF grid and predicts vertex positions."""
    vox_size = grid_sdf.shape[0] - 1
    padded_sdf_grid = zero_pad_sdf(grid_sdf, vox_size, grid_pad=pad)
    pred_pos_full = decoder(padded_sdf_grid.unsqueeze(0).unsqueeze(1))
    pred_pos = pred_pos_full.squeeze()[:-1, :-1, :-1].reshape(-1, 3)
    pred_pos_sel = pred_pos[valid_cell_id].squeeze()
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    pred_pos_glob = pred_pos_sel + cell_bot_id
    return pred_pos_glob, pred_pos_full.squeeze()


def get_dc_f_grid(sdf_vox, vox_ind=None, cell_v_idx_th=None):
    """Gets dual contouring faces from an SDF grid."""
    if vox_ind is None:
        vox_ind = manual_vox_ind(sdf_vox.shape[0]).cuda().long()
    if cell_v_idx_th is None:
        cell_v_idx_th = get_cell_v_idx(vox_ind)
    n_vox = sdf_vox.shape[1]
    vert_vox_id = ones((n_vox, n_vox, n_vox)).to(sdf_vox.device).long() * -1
    cross_e, valid_cell_id = get_intersec(sdf_vox, vox_ind)
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    vert_vox_id[cell_bot_id[:, 0], cell_bot_id[:, 1], cell_bot_id[:, 2]] = torch.arange(
        cell_bot_id.shape[0]
    ).to(sdf_vox.device)
    return fast_face_construct(cross_e, vert_vox_id)


def get_dc_f_proba_grid(sdf_vox, proba_vox, vox_ind=None, cell_v_idx_th=None):
    """Gets dual contouring faces with probabilities."""
    if vox_ind is None:
        vox_ind = manual_vox_ind(sdf_vox.shape[0]).cuda().long()
    if cell_v_idx_th is None:
        cell_v_idx_th = get_cell_v_idx(vox_ind)
    n_vox = sdf_vox.shape[1]
    vert_vox_id = ones((n_vox, n_vox, n_vox)).to(sdf_vox.device).long() * -1
    cross_e, valid_cell_id = get_intersec(sdf_vox, vox_ind)
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    vert_vox_id[cell_bot_id[:, 0], cell_bot_id[:, 1], cell_bot_id[:, 2]] = torch.arange(
        cell_bot_id.shape[0]
    ).to(sdf_vox.device)
    return fast_face_construct_proba(cross_e, vert_vox_id, proba_vox)


def get_vert_grid(
    grid_sdf, decoder, valid_cell_id, cell_v_idx_th, dc_f, clamp=False, extra_sample=0
):
    """Gets vertices from a grid."""
    pred_pos_sel, pred_pos_full = pred_v_grid(grid_sdf, decoder, valid_cell_id)
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    pred_glob = pred_pos_sel + cell_bot_id
    gen_samp_glob, samp_id = custom_sample_trig(
        pred_glob[dc_f], extra_sample=extra_sample
    )
    if clamp:
        cell_max = cell_bot_id + ones((1, 3)).to(grid_sdf.device).to(cell_bot_id.dtype)
        pred_glob = torch.clamp(pred_glob, cell_bot_id, cell_max)
    return pred_pos_full, pred_glob, gen_samp_glob, samp_id


def pred_v_grid_short(grid_sdf, decoder, valid_cell_id, pad_tup, grid_size):
    """Predicts vertices from a cropped grid."""
    pred_pos_full = zeros((grid_size, grid_size, grid_size, 3)).to(grid_sdf.device)
    pred_pos_ = decoder(grid_sdf.unsqueeze(0).unsqueeze(1))[0]
    pred_pos_full[
        pad_tup[0] : pad_tup[1], pad_tup[2] : pad_tup[3], pad_tup[4] : pad_tup[5]
    ] = pred_pos_
    pred_pos = pred_pos_full[:-1, :-1, :-1].reshape(-1, 3)
    pred_pos_sel = pred_pos[valid_cell_id]
    return pred_pos_sel, pred_pos_full


def get_vert_grid_short(
    grid_sdf,
    decoder,
    valid_cell_id,
    cell_v_idx_th,
    dc_f,
    pad_tup,
    grid_size,
    clamp=False,
    extra_sample=0,
):
    """Gets vertices from a cropped grid."""
    pred_pos_sel, pred_pos_full = pred_v_grid_short(
        grid_sdf, decoder, valid_cell_id, pad_tup, grid_size
    )
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    pred_glob = pred_pos_sel + cell_bot_id
    gen_samp_glob, samp_id = custom_sample_trig(
        pred_glob[dc_f], extra_sample=extra_sample
    )
    if clamp:
        cell_max = cell_bot_id + ones((1, 3)).to(grid_sdf.device).to(cell_bot_id.dtype)
        pred_glob = torch.clamp(pred_glob, cell_bot_id, cell_max)
    return pred_pos_full, pred_glob, gen_samp_glob, samp_id


def get_vert_grid_short_GC(
    grid_sdf,
    decoder,
    valid_cell_id,
    cell_v_idx_th,
    dc_f,
    pad_tup,
    grid_size,
    clamp=False,
    extra_sample=0,
):
    """Gets vertices from a cropped grid with global coordinates."""
    pred_pos_sel, pred_pos_full = pred_v_grid_short(
        grid_sdf, decoder, valid_cell_id, pad_tup, grid_size
    )
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id]
    pred_glob = pred_pos_sel * (2 / (grid_size - 1)) + cell_bot_id
    gen_samp_glob, samp_id = custom_sample_trig(
        pred_glob[dc_f], extra_sample=extra_sample
    )
    return pred_pos_full, pred_glob, gen_samp_glob, samp_id


def get_vert_grid_bat(grid_sdf, decoder, valid_cell_id, cell_v_idx_th, dc_f, f_len):
    """Gets vertices for a batch of grids."""
    pred_pos_sel, pred_pos_full = pred_v_grid_bat(grid_sdf, decoder, valid_cell_id)
    cell_bot_id = cell_v_idx_th[:, 0, :][valid_cell_id[:, 1]]
    pred_glob = pred_pos_sel + cell_bot_id
    f_len_th = torch.cuda.LongTensor(f_len)
    extra_samp = (((f_len_th * 10) - f_len_th.max() * 10) * -1).long()
    batch_gen_samp = []
    batch_gen_idx = []
    n_batch = grid_sdf.shape[0]
    for i in range(n_batch):
        cur_verts = pred_glob[(valid_cell_id[:, 0] == i).nonzero().squeeze()]
        cur_faces = dc_f[i, : f_len[i]]
        gen_samp_glob, samp_id = custom_sample_trig(
            cur_verts[cur_faces], extra_sample=extra_samp[i].item()
        )
        batch_gen_idx.append(samp_id)
        batch_gen_samp.append(gen_samp_glob)

    pred_glob_bat = pred_glob
    gen_samp_glob_cat = torch.stack(batch_gen_samp)
    gen_samp_idx_cat = torch.stack(batch_gen_idx)

    return pred_pos_full, pred_glob_bat, gen_samp_glob_cat, gen_samp_idx_cat


def get_undc_edges_from_out(int_grid, face_id_grid, confidence=0.3):
    """Gets edges from the output of a UN-DC network."""
    int_grid_bin = torch.where(
        int_grid > confidence, ones_like(int_grid), zeros_like(int_grid)
    )
    x_offset_trig_1 = (
        torch.LongTensor([[0, -1, -1], [0, 0, -1], [0, 0, 0]])
        .long()
        .to(int_grid.device)
    )
    x_offset_trig_2 = (
        torch.LongTensor([[0, -1, -1], [0, 0, 0], [0, -1, 0]])
        .long()
        .to(int_grid.device)
    )

    x_offset_quad = (
        torch.LongTensor([[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]])
        .long()
        .to(int_grid.device)
    )

    y_offset_trig_1 = (
        torch.LongTensor([[-1, 0, -1], [0, 0, 0], [0, 0, -1]])
        .long()
        .to(int_grid.device)
    )
    y_offset_trig_2 = (
        torch.LongTensor([[-1, 0, -1], [-1, 0, 0], [0, 0, 0]])
        .long()
        .to(int_grid.device)
    )

    y_offset_quad = (
        torch.LongTensor([[-1, 0, -1], [0, 0, -1], [0, 0, 0], [-1, 0, 0]])
        .long()
        .to(int_grid.device)
    )

    z_offset_trig_1 = (
        torch.LongTensor([[-1, -1, 0], [0, 0, 0], [-1, 0, 0]])
        .long()
        .to(int_grid.device)
    )
    z_offset_trig_2 = (
        torch.LongTensor([[-1, -1, 0], [0, -1, 0], [0, 0, 0]])
        .long()
        .to(int_grid.device)
    )

    z_offset_quad = (
        torch.LongTensor([[-1, -1, 0], [-1, 0, 0], [0, 0, 0], [0, -1, 0]])
        .long()
        .to(int_grid.device)
    )

    int_grid_bin_x = int_grid_bin[:-1, 1:-1, 1:-1, :]
    int_grid_bin_y = int_grid_bin[1:-1, :-1, 1:-1, :]
    int_grid_bin_z = int_grid_bin[1:-1, 1:-1, :-1, :]

    int_grid_nz_ind_x = torch.nonzero(int_grid_bin_x)
    int_grid_nz_ind_y = torch.nonzero(int_grid_bin_y)
    int_grid_nz_ind_z = torch.nonzero(int_grid_bin_z)

    nonzero_x = int_grid_nz_ind_x[:, 3] == 0
    x_dir_ind = int_grid_nz_ind_x[nonzero_x][:, :3]
    nonzero_y = int_grid_nz_ind_y[:, 3] == 1
    y_dir_ind = int_grid_nz_ind_y[nonzero_y][:, :3]
    nonzero_z = int_grid_nz_ind_z[:, 3] == 2
    z_dir_ind = int_grid_nz_ind_z[nonzero_z][:, :3]

    all_trigs, alt_proba, proba_comb = [], [], []
    all_quad = []
    proba_dict = {}
    trig_1, trig_2 = [], []
    if len(x_dir_ind) > 0:
        quad_ind = (x_dir_ind.unsqueeze(1) + x_offset_quad).view(-1, 3)
        quad_offset = (
            LongTensor([0, 1, 1]).cuda().unsqueeze(0).expand(quad_ind.shape[0], -1)
        )

        trig_1_ind = (x_dir_ind.unsqueeze(1) + x_offset_trig_1).view(-1, 3)
        trig_2_ind = (x_dir_ind.unsqueeze(1) + x_offset_trig_2).view(-1, 3)
        trig_offset = LongTensor([0, 1, 1]).cuda().unsqueeze(0)
        trig_1_ind += trig_offset.expand(trig_1_ind.shape[0], -1)
        trig_2_ind += trig_offset.expand(trig_1_ind.shape[0], -1)

        x_trigs_1 = face_id_grid[
            trig_1_ind[:, 0], trig_1_ind[:, 1], trig_1_ind[:, 2]
        ].view(-1, 3)
        x_trigs_2 = face_id_grid[
            trig_2_ind[:, 0], trig_2_ind[:, 1], trig_2_ind[:, 2]
        ].view(-1, 3)
        x_trigs_proba = int_grid[:-1, 1:-1, 1:-1][
            x_dir_ind[:, 0], x_dir_ind[:, 1], x_dir_ind[:, 2], 0
        ]
        x_proba_combined = torch.repeat_interleave(x_trigs_proba, 2)
        all_trigs.append(
            torch.cat([x_trigs_1, x_trigs_2], dim=-1).view(x_trigs_1.shape[0], 2, -1)
        )

        trig_1.append(x_trigs_1)
        trig_2.append(x_trigs_2)
        for k, v in zip(numpied(x_trigs_1).tolist(), numpied(x_trigs_proba).tolist()):
            proba_dict[tuple(sorted(tuple(k)))] = v
        for k, v in zip(numpied(x_trigs_2).tolist(), numpied(x_trigs_proba).tolist()):
            proba_dict[tuple(sorted(tuple(k)))] = v

        q_ind_f = quad_ind + quad_offset
        cur_quad = face_id_grid[q_ind_f[:, 0], q_ind_f[:, 1], q_ind_f[:, 2]].view(-1, 4)
        all_quad.append(cur_quad)
        alt_proba.append(x_trigs_proba)
        proba_comb.append(x_proba_combined)

    if len(y_dir_ind) > 0:
        quad_ind = (y_dir_ind.unsqueeze(1) + y_offset_quad).view(-1, 3)
        quad_offset = (
            LongTensor([1, 0, 1]).cuda().unsqueeze(0).expand(quad_ind.shape[0], -1)
        )
        trig_1_ind = (y_dir_ind.unsqueeze(1) + y_offset_trig_1).view(-1, 3)
        trig_2_ind = (y_dir_ind.unsqueeze(1) + y_offset_trig_2).view(-1, 3)
        trig_offset = LongTensor([1, 0, 1]).cuda().unsqueeze(0)
        trig_1_ind += trig_offset.expand(trig_1_ind.shape[0], -1)
        trig_2_ind += trig_offset.expand(trig_1_ind.shape[0], -1)

        y_trigs_1 = face_id_grid[
            trig_1_ind[:, 0], trig_1_ind[:, 1], trig_1_ind[:, 2]
        ].view(-1, 3)
        y_trigs_2 = face_id_grid[
            trig_2_ind[:, 0], trig_2_ind[:, 1], trig_2_ind[:, 2]
        ].view(-1, 3)
        y_trigs_proba = int_grid[1:-1, :-1, 1:-1][
            y_dir_ind[:, 0], y_dir_ind[:, 1], y_dir_ind[:, 2], 1
        ]
        y_proba_combined = torch.repeat_interleave(y_trigs_proba, 2)
        trig_1.append(y_trigs_1)
        trig_2.append(y_trigs_2)
        for k, v in zip(numpied(y_trigs_1).tolist(), numpied(y_trigs_proba).tolist()):
            proba_dict[tuple(sorted(tuple(k)))] = v
        for k, v in zip(numpied(y_trigs_2).tolist(), numpied(y_trigs_proba).tolist()):
            proba_dict[tuple(sorted(tuple(k)))] = v
        q_ind_f = quad_ind + quad_offset
        cur_quad = face_id_grid[q_ind_f[:, 0], q_ind_f[:, 1], q_ind_f[:, 2]].view(-1, 4)
        all_quad.append(cur_quad)
        all_trigs.append(
            torch.cat([y_trigs_1, y_trigs_2], dim=-1).view(y_trigs_1.shape[0], 2, -1)
        )
        alt_proba.append(y_trigs_proba)
        proba_comb.append(y_proba_combined)

    if len(z_dir_ind) > 0:
        quad_ind = (z_dir_ind.unsqueeze(1) + z_offset_quad).view(-1, 3)
        quad_offset = (
            LongTensor([1, 1, 0]).cuda().unsqueeze(0).expand(quad_ind.shape[0], -1)
        )
        trig_offset = LongTensor([1, 1, 0]).cuda().unsqueeze(0)
        trig_1_ind = (z_dir_ind.unsqueeze(1) + z_offset_trig_1).view(-1, 3)
        trig_2_ind = (z_dir_ind.unsqueeze(1) + z_offset_trig_2).view(-1, 3)
        trig_1_ind += trig_offset.expand(trig_1_ind.shape[0], -1)
        trig_2_ind += trig_offset.expand(trig_1_ind.shape[0], -1)
        z_trigs_1 = face_id_grid[
            trig_1_ind[:, 0], trig_1_ind[:, 1], trig_1_ind[:, 2]
        ].view(-1, 3)
        z_trigs_2 = face_id_grid[
            trig_2_ind[:, 0], trig_2_ind[:, 1], trig_2_ind[:, 2]
        ].view(-1, 3)
        z_trigs_proba = int_grid[1:-1, 1:-1, :-1][
            z_dir_ind[:, 0], z_dir_ind[:, 1], z_dir_ind[:, 2], 2
        ]

        z_proba_combined = torch.repeat_interleave(z_trigs_proba, 2)
        trig_1.append(z_trigs_1)
        trig_2.append(z_trigs_2)
        for k, v in zip(numpied(z_trigs_1).tolist(), numpied(z_trigs_proba).tolist()):
            proba_dict[tuple(sorted(tuple(k)))] = v
        for k, v in zip(numpied(z_trigs_2).tolist(), numpied(z_trigs_proba).tolist()):
            proba_dict[tuple(sorted(tuple(k)))] = v
        q_ind_f = quad_ind + quad_offset
        cur_quad = face_id_grid[q_ind_f[:, 0], q_ind_f[:, 1], q_ind_f[:, 2]].view(-1, 4)
        all_quad.append(cur_quad)
        all_trigs.append(
            torch.cat([z_trigs_1, z_trigs_2], dim=-1).view(z_trigs_1.shape[0], 2, -1)
        )
        alt_proba.append(z_trigs_proba)
        proba_comb.append(z_proba_combined)

    if len(all_trigs) == 0:
        all_trigs = (
            torch.LongTensor([[[0, 0, 0], [0, 0, 0], [0, 0, 0]]]).to(int_grid.device),
        )
        alt_proba = (torch.FloatTensor([0.0]).to(int_grid.device),)

    trig_1 = torch.cat(trig_1)
    trig_2 = torch.cat(trig_2)
    all_trigs = torch.cat(all_trigs).reshape(-1, 3)
    all_quad = torch.cat(all_quad)
    all_proba = torch.cat(proba_comb)
    return all_proba, all_trigs


def get_all_vert_undc(int_grid, float_grid, vox_ind, conf):
    """Gets all vertices from the output of a UN-DC network."""
    dimx, dimy, dimz, _ = int_grid.shape
    vertices_grid = (torch.ones([dimx, dimy, dimz]) * -1).to(int_grid.device).long()
    x_ind, y_ind, z_ind = get_cell_e_idx(vox_ind)
    x_ind_off, y_ind_off, z_ind_off = (
        x_ind.reshape(-1, 4, 3),
        y_ind.reshape(-1, 4, 3),
        z_ind.reshape(-1, 4, 3),
    )
    xy_bool = logical_or(
        int_grid[x_ind_off[:, :, 0], x_ind_off[:, :, 1], x_ind_off[:, :, 2], 0] >= conf,
        int_grid[y_ind_off[:, :, 0], y_ind_off[:, :, 1], y_ind_off[:, :, 2], 1] >= conf,
    )
    xyz_bool = (
        logical_or(
            xy_bool,
            int_grid[z_ind_off[:, :, 0], z_ind_off[:, :, 1], z_ind_off[:, :, 2], 2]
            >= conf,
        )
        .any(dim=1)
        .nonzero()
        .squeeze()
    )
    xyz_bool_vox = torch.from_numpy(
        np.unravel_index(numpied(xyz_bool), (dimx - 1, dimy - 1, dimz - 1))
    ).T
    vertices_grid[xyz_bool_vox[:, 0], xyz_bool_vox[:, 1], xyz_bool_vox[:, 2]] = (
        torch.arange(xyz_bool_vox.shape[0]).to(int_grid.device).to(vertices_grid.dtype)
    )

    pos_offset = xyz_bool_vox.to(int_grid.device)
    pos = (
        float_grid[xyz_bool_vox[:, 0], xyz_bool_vox[:, 1], xyz_bool_vox[:, 2]]
        + pos_offset
    )

    return pos, vertices_grid


def get_undc_face_proba(int_grid, float_grid, all_vox_int, confidence=0.5):
    """Gets faces and probabilities from the output of a UN-DC network."""
    all_vert_th, vertices_grid_th = get_all_vert_undc(
        int_grid, float_grid, all_vox_int, confidence
    )

    all_proba_th, all_trigs = get_undc_edges_from_out(
        int_grid, vertices_grid_th, confidence
    )
    return all_vert_th, all_proba_th, all_trigs


def get_undc_face_proba_close(
    pred_bool_th, pred_float_th, voxel_xyz_int, all_vox_int, confidence=0.5
):
    """Gets faces and probabilities for a cropped region."""
    vox_size = pred_bool_th.shape[0] - 1
    pred_bool_th_ = torch.zeros(
        [vox_size + 1, vox_size + 1, vox_size + 1, 3],
        dtype=torch.float32,
        device=pred_bool_th.device,
    )

    pred_bool_th_[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]] = (
        pred_bool_th[voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]]
    )
    return get_undc_face_proba(
        pred_bool_th_, pred_float_th, voxel_xyz_int, all_vox_int, confidence=confidence
    )


def dual_contouring_NDC(int_grid, float_grid):
    """Performs dual contouring on a grid."""
    all_vertices = []
    all_triangles = []
    int_grid = np.squeeze(int_grid)
    dimx, dimy, dimz = int_grid.shape
    vertices_grid = np.full([dimx, dimy, dimz], -1, np.int32)

    for i in range(0, dimx - 1):
        for j in range(0, dimy - 1):
            for k in range(0, dimz - 1):

                v0 = int_grid[i, j, k]
                v1 = int_grid[i + 1, j, k]
                v2 = int_grid[i + 1, j + 1, k]
                v3 = int_grid[i, j + 1, k]
                v4 = int_grid[i, j, k + 1]
                v5 = int_grid[i + 1, j, k + 1]
                v6 = int_grid[i + 1, j + 1, k + 1]
                v7 = int_grid[i, j + 1, k + 1]

                if (
                    v1 != v0
                    or v2 != v0
                    or v3 != v0
                    or v4 != v0
                    or v5 != v0
                    or v6 != v0
                    or v7 != v0
                ):
                    vertices_grid[i, j, k] = len(all_vertices)
                    pos = float_grid[i, j, k] + np.array([i, j, k], np.float32)
                    all_vertices.append(pos)

    all_vertices = np.array(all_vertices, np.float32)

    for i in range(0, dimx - 1):
        for j in range(1, dimy - 1):
            for k in range(1, dimz - 1):
                v0 = int_grid[i, j, k]
                v1 = int_grid[i + 1, j, k]
                if v0 != v1:
                    all_triangles.append(
                        [
                            vertices_grid[i, j - 1, k - 1],
                            vertices_grid[i, j, k - 1],
                            vertices_grid[i, j, k],
                        ]
                    )
                    all_triangles.append(
                        [
                            vertices_grid[i, j - 1, k - 1],
                            vertices_grid[i, j, k],
                            vertices_grid[i, j - 1, k],
                        ]
                    )

    for i in range(1, dimx - 1):
        for j in range(0, dimy - 1):
            for k in range(1, dimz - 1):
                v0 = int_grid[i, j, k]
                v1 = int_grid[i, j + 1, k]
                if v0 != v1:
                    all_triangles.append(
                        [
                            vertices_grid[i - 1, j, k - 1],
                            vertices_grid[i, j, k],
                            vertices_grid[i, j, k - 1],
                        ]
                    )
                    all_triangles.append(
                        [
                            vertices_grid[i - 1, j, k - 1],
                            vertices_grid[i - 1, j, k],
                            vertices_grid[i, j, k],
                        ]
                    )

    for i in range(1, dimx - 1):
        for j in range(1, dimy - 1):
            for k in range(0, dimz - 1):
                v0 = int_grid[i, j, k]
                v1 = int_grid[i, j, k + 1]
                if v0 != v1:
                    all_triangles.append(
                        [
                            vertices_grid[i - 1, j - 1, k],
                            vertices_grid[i, j, k],
                            vertices_grid[i - 1, j, k],
                        ]
                    )
                    all_triangles.append(
                        [
                            vertices_grid[i - 1, j - 1, k],
                            vertices_grid[i, j - 1, k],
                            vertices_grid[i, j, k],
                        ]
                    )

    all_triangles = np.array(all_triangles, np.int32)
    return all_vertices, all_triangles
