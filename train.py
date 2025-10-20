import argparse
import json
import os
import os.path as osp
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pytorch3d.loss import mesh_laplacian_smoothing
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.ops.knn import knn_points
from pytorch3d.structures import Meshes
from torch import ones_like
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tqdm._utils import _term_move_up

from dataset import ABC_grid_hdf5, my_collate
from dc_utils import get_cell_e_idx, get_cell_v_idx, get_vert_grid, get_vert_grid_short
from eval_mesh import eval_all
from mesh_loss import bi_dir_cd, trig_angle_loss
from my_argparse import get_argparse_2
from Networks import decoders, encoders, sdfNets
from utils import (
    angle_bw_th,
    dotdict,
    face_normals,
    gen_grid_vert,
    load_m,
    manual_vox_ind,
    numpied,
    put_to_device,
    safe_make_dirs,
    sample_along_edge_th,
    scatter_mean_to_vox,
    torched,
    trimesh_from_vf,
)


def save_all(epoch=None):
    suffix = "final" if epoch is None else str(epoch)
    save_path = osp.join(save_dir, "model_%s.th" % suffix)
    torch.save(
        {"decoder_sd": decoder.state_dict(), "optim_dec_sd": optim_dec.state_dict()},
        save_path,
    )


def advance_scheduler():
    if s_epoch > 0:
        for ep in range(s_epoch):
            scheduler_dec.step()


def load_decoder(path):
    loaded_th = torch.load(path)
    try:
        decoder.load_state_dict(loaded_th)
    except:
        decoder.load_state_dict(loaded_th["decoder_sd"])
        optim_dec.load_state_dict(loaded_th["optim_dec_sd"])


def predict_dc_vert(grid_sdf, valid_cell_id, dc_f, pad_tup=None, is_short=True):
    if is_short:
        assert pad_tup is not None
        pred_v_full, pred_v_glob, gen_samp_glob, samp_id = get_vert_grid_short(
            grid_sdf, decoder, valid_cell_id, cell_v_idx_th, dc_f, pad_tup, grid_size
        )
    else:
        pred_v_full, pred_v_glob, gen_samp_glob, samp_id = get_vert_grid(
            grid_sdf, decoder, valid_cell_id, cell_v_idx_th, dc_f
        )

    return pred_v_full, pred_v_glob, gen_samp_glob, samp_id.squeeze()


def apply_gt_loss(gt_pts, pred_pos, gt_mask):
    gt_pts = gt_pts.squeeze().to(pred_pos.device)
    cd_pt = torch.sum(((pred_pos - gt_pts) ** 2) * gt_mask) / torch.clamp(
        torch.sum(gt_mask), min=1
    )
    return cd_pt


def apply_face_cd(gt_pt_glob, gen_samp_glob):
    if cfg_dd.masked_cd:
        occ_mask_ = (
            torch.cdist(gt_pt_glob.floor().unique(dim=0), gen_samp_glob.floor()).min(
                dim=0
            )[0]
            == 0.0
        )
        gen_samp_glob = gen_samp_glob[occ_mask_]
    if not gen_samp_glob.ndim == 3:
        gen_samp_glob = gen_samp_glob.unsqueeze(0)
    if not gt_pt_glob.ndim == 3:
        gt_pt_glob = gt_pt_glob.unsqueeze(0)

    cd_pt = bi_dir_cd(gen_samp_glob, gt_pt_glob)
    return cd_pt


def apply_p2f(gt_pt_glob, gen_samp_glob, pred_v_glob, dc_f, mesh_v, mesh_f):
    from utils import get_closest_f_dist

    p2f = get_closest_f_dist(gt_pt_glob.squeeze(), pred_v_glob[dc_f])
    p2f_bi = get_closest_f_dist(gen_samp_glob.squeeze(), mesh_v[mesh_f])
    if p2f.isnan().any():
        raise ValueError("p2f has nan values")
    cd_pt = p2f.mean() + p2f_bi.mean()
    return cd_pt


def apply_p2f_offsur(sdf_pts, sdf_vals, pred_v_glob, dc_f):
    from utils import get_closest_f_dist

    p2f = get_closest_f_dist(sdf_pts.squeeze(), pred_v_glob[dc_f])
    if p2f.isnan().any():
        raise ValueError("p2f has nan values in off surface loss")
    return (p2f.squeeze() - sdf_vals.squeeze()).mean()


def apply_norm_cd(surf_pt_glob, gt_pt_norm, gen_samp_glob, gen_f_norm):
    if not gen_samp_glob.ndim == 3:
        gen_samp_glob = gen_samp_glob.unsqueeze(0)
    if not surf_pt_glob.ndim == 3:
        surf_pt_glob = surf_pt_glob.unsqueeze(0)
    if not gt_pt_norm.ndim == 3:
        gt_pt_norm = gt_pt_norm.unsqueeze(0)
    if not gen_f_norm.ndim == 3:
        gen_f_norm = gen_f_norm.unsqueeze(0)

    cd_pt, cd_norm = chamfer_distance(
        gen_samp_glob, surf_pt_glob, x_normals=gen_f_norm, y_normals=gt_pt_norm
    )
    return cd_pt, cd_norm


def apply_second_order_cd(
    gt_pt_glob, gen_samp_glob, gt_pt_norm, gt_e1, gt_e2, gt_rho_1, gt_rho_2
):
    if cfg_dd.masked_cd:
        occ_mask_ = (
            torch.cdist(gt_pt_glob.floor().unique(dim=0), gen_samp_glob.floor()).min(
                dim=0
            )[0]
            == 0.0
        )
        gen_samp_glob = gen_samp_glob[occ_mask_]
    dist, nn_idx, foot_pts = knn_points(
        gen_samp_glob.unsqueeze(0), gt_pt_glob.unsqueeze(0), norm=2, K=1, return_nn=True
    )
    dist, foot_pts = dist.squeeze().sqrt(), foot_pts.squeeze()
    fp_vec = gen_samp_glob - foot_pts
    fp_norm = gt_pt_norm[nn_idx.squeeze()]
    # Rho is radius so abs
    gt_rho_1 = torch.nan_to_num(gt_rho_1, 1e6)
    gt_rho_2 = torch.nan_to_num(gt_rho_2, 1e6)

    fp_rho_1 = gt_rho_1[nn_idx.squeeze()].abs()
    fp_rho_2 = gt_rho_2[nn_idx.squeeze()].abs()
    fp_e1 = gt_e1[nn_idx.squeeze()]
    fp_e2 = gt_e2[nn_idx.squeeze()]
    sign = (
        angle_bw_th(fp_vec, fp_norm).clamp(-1, 1).acos().rad2deg() > 90.0
    ).float() * (-1.0)
    masked_dist = -1.0 * dist  # This makes sure that denominator is non-zero
    # masked_dist = dist
    delta_1 = masked_dist / (masked_dist - fp_rho_1 - 1e-8)
    delta_2 = masked_dist / (masked_dist - fp_rho_2 - 1e-8)

    # Else the loss will explore with small denominator
    delta_1 = torch.where(delta_1.abs() > 1.0, masked_dist, delta_1)
    delta_2 = torch.where(delta_2.abs() > 1.0, masked_dist, delta_2)
    # delta_1 = delta_1*delta_1_ma
    # delta_2 = delta_2*delta_2_ma
    xy_dist = (
        delta_1 * (torch.einsum("ij,ij->i", fp_e1, fp_vec)) ** 2
        + delta_2 * (torch.einsum("ij,ij->i", fp_e2, fp_vec)) ** 2
    )
    z_dist = torch.einsum("ij,ij->i", fp_norm, fp_vec) ** 2
    approx_dist = z_dist + xy_dist

    return approx_dist.mean()


def apply_e_cd(gt_pt_glob, pred_v_glob, unique_e_idx):
    t1 = time.time()
    gen_e_samp = sample_along_edge_th(pred_v_glob, unique_e_idx, 4).squeeze()
    if gt_pt_glob.shape[0] > 50000:
        rand_idx = np.random.choice(gt_pt_glob.shape[0], 50000, replace=False)
        gt_pt_glob = gt_pt_glob[rand_idx]
    cd_pt = bi_dir_cd(gen_e_samp.unsqueeze(0), gt_pt_glob.squeeze().unsqueeze(0))
    t2 = time.time()
    return cd_pt


def joint_local_aug(surf_pt_glob, gen_offset):
    surf_pt_offset = surf_pt_glob.floor()
    surf_pt_uc = surf_pt_glob - surf_pt_offset
    choice = np.random.randint(0, 4)
    offset_max = vox_size - 1
    if choice == 0:
        surf_pt_offset = torch.hstack(
            (offset_max - surf_pt_offset[:, 0:1], surf_pt_offset[:, 1:])
        )
        gen_offset = torch.hstack((offset_max - gen_offset[:, 0:1], gen_offset[:, 1:]))
        surf_pt_flip = torch.hstack((1.0 - surf_pt_uc[:, 0:1], surf_pt_uc[:, 1:]))
    elif choice == 1:
        surf_pt_offset = torch.hstack(
            (
                surf_pt_offset[:, 0:1],
                offset_max - surf_pt_offset[:, 1:2],
                surf_pt_offset[:, 2:],
            )
        )
        gen_offset = torch.hstack(
            (gen_offset[:, 0:1], offset_max - gen_offset[:, 1:2], gen_offset[:, 2:])
        )
        surf_pt_flip = torch.hstack(
            (surf_pt_uc[:, 0:1], 1.0 - surf_pt_uc[:, 1:2], surf_pt_uc[:, 2:])
        )
    elif choice == 2:
        surf_pt_offset = torch.hstack(
            (surf_pt_offset[:, 0:2], offset_max - surf_pt_offset[:, 2:])
        )
        gen_offset = torch.hstack((gen_offset[:, 0:2], offset_max - gen_offset[:, 2:]))
        surf_pt_flip = torch.hstack((surf_pt_uc[:, 0:2], 1.0 - surf_pt_uc[:, 2:]))
    else:
        surf_pt_offset = surf_pt_offset
        gen_offset = gen_offset
        surf_pt_flip = surf_pt_uc

    surf_pt_flip = surf_pt_flip + surf_pt_offset
    return surf_pt_flip, gen_offset


def joint_glob_aug(surf_pt_glob, gen_offset):
    surf_pt_offset = surf_pt_glob.floor()
    surf_pt_uc = surf_pt_glob - surf_pt_offset
    permutation_list = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    permutation = permutation_list[np.random.randint(len(permutation_list))]
    surf_pt_offset = surf_pt_offset[:, permutation]
    gen_offset = gen_offset[:, permutation]
    surf_pt_uc = surf_pt_uc[:, permutation]
    return surf_pt_uc + surf_pt_offset, gen_offset


def training_step():
    for idx, (train_batch) in enumerate(train_loader):
        net_loss, cd_norm = 0.0, 0.0
        loss_dict = {}
        decoder.train()
        optim_dec.zero_grad()
        surf_pt_glob = torched(train_batch["surf_pt"]).squeeze()

        grid_sdf, val_cell_id, dc_f, pad_tup = put_to_device(
            [train_batch[i] for i in ["sdf_inp", "val_cell_id", "dc_f", "pad_tup"]]
        )
        dc_f = dc_f.squeeze().long()
        pad_tup = pad_tup.long()
        gen_offset = val_cell_id.reshape(vox_size, vox_size, vox_size).nonzero()

        if cfg_dd.get("local_aug", False):
            surf_pt_glob, gen_offset = joint_local_aug(surf_pt_glob, gen_offset)

        if cfg_dd.get("glob_aug", False):
            raise NotImplementedError
            surf_pt_glob, gen_offset = joint_glob_aug(surf_pt_glob, gen_offset)

        val_cell_id = (
            (
                gen_offset
                * torch.cuda.FloatTensor([vox_size**2, vox_size, 1]).unsqueeze(0)
            )
            .sum(dim=1)
            .long()
        )

        pred_v_full, pred_v_glob, gen_samp_glob, samp_id = predict_dc_vert(
            grid_sdf, val_cell_id, dc_f, pad_tup, is_short=True
        )

        if np.random.randint(2) == 0:
            grid_sdf = grid_sdf * -1

        if cfg_dd.use_gt:
            gt_pts, gt_mask = put_to_device(
                train_batch[i] for i in ["dc_v_vox", "dc_vox_ma"]
            )
            cd_pt = apply_gt_loss(gt_pts, pred_v_full, gt_mask)

        elif cfg_dd.vert_cd:
            cd_pt = bi_dir_cd(pred_v_glob.unsqueeze(0), gt_pts.unsqueeze(0))

        elif cfg_dd.cd_norm_coeff > 0.0:
            gt_pt_norm = torched(train_batch["gt_surf_norm"]).squeeze()
            gen_m_norm = face_normals(pred_v_glob, dc_f)
            gen_f_norm = gen_m_norm[samp_id]
            cd_pt, cd_norm = apply_norm_cd(
                surf_pt_glob.unsqueeze(0),
                gt_pt_norm.unsqueeze(0),
                gen_samp_glob,
                gen_f_norm,
            )

        elif cfg_dd.second_order:
            curvature_keys = ["gt_surf_norm", "gt_e1", "gt_e2", "gt_rho_1", "gt_rho_2"]
            gt_pt_norm, gt_e1, gt_e2, gt_rho_1, gt_rho_2 = put_to_device(
                [train_batch[i] for i in curvature_keys]
            )
            cd_pt = apply_second_order_cd(
                surf_pt_glob,
                gen_samp_glob,
                gt_pt_norm,
                gt_e1,
                gt_e2,
                gt_rho_1,
                gt_rho_2,
            )
        else:
            cd_pt = apply_face_cd(surf_pt_glob, gen_samp_glob)

        loss_dict["cd"] = cd_pt * cfg_dd.cd_coeff
        loss_dict["cd_norm"] = cd_norm * cfg_dd.cd_norm_coeff

        writer.add_scalar("CD Loss", cd_pt.item(), ep * len(train_loader) + idx)

        if cfg_dd.lap_coeff > 0.0:
            gen_m = Meshes(verts=[pred_v_glob], faces=[dc_f])
            lap_loss = mesh_laplacian_smoothing(gen_m, method="cot")
            loss_dict["lap_loss"] = lap_loss * cfg_dd.lap_coeff
            writer.add_scalar("lap_loss", lap_loss.item(), ep * len(train_loader) + idx)

        if cfg_dd.p2f_coeff > 0.0:
            mesh_v, mesh_f = (
                torched(train_batch["gt_v"]).squeeze(),
                torched(train_batch["gt_f"]).squeeze().long(),
            )
            p2f = apply_p2f(
                surf_pt_glob, gen_samp_glob, pred_v_glob, dc_f, mesh_v, mesh_f
            )
            loss_dict["p2f"] = p2f * cfg_dd.p2f_coeff
            writer.add_scalar("p2f_loss", p2f.item(), ep * len(train_loader) + idx)

        if cfg_dd.get("p2f_offsur_coeff", 0.0) > 0.0:
            sdf_pts = torched(train_batch["sdf_pts"]).squeeze()
            sdf_vals = torched(train_batch["sdf_vals"]).squeeze()
            p2f_off = apply_p2f_offsur(sdf_pts, sdf_vals, pred_v_glob, dc_f)
            loss_dict["p2f_offsur"] = p2f_off * cfg_dd.p2f_offsur_coeff
            writer.add_scalar(
                "p2f_offsur_loss", p2f_off.item(), ep * len(train_loader) + idx
            )

        if cfg_dd.get("angle_coeff", 0.0) > 0.0:
            ang_func = trig_angle_loss(pred_v_glob, dc_f)
            loss_dict["ang_l"] = ang_func * cfg_dd.angle_coeff

        if cfg_dd.get("ecd_coeff", 0.0) > 0.0:
            surf_pt_glob = torched(train_batch["edge_pt"]).squeeze()
            unique_e_idx = torched(train_batch["dc_e"], dtype=torch.long).squeeze()
            cd_e = apply_e_cd(surf_pt_glob, pred_v_glob, unique_e_idx)
            loss_dict["cd_e"] = cd_e * cfg_dd.ecd_coeff
            writer.add_scalar("Edge CD", cd_e.item(), ep * len(train_loader) + idx)

        cd_loss = cd_pt * cfg_dd.cd_coeff
        cd_loss += cd_norm * cfg_dd.cd_norm_coeff

        if cfg_dd.mp_coeff > 0.0:
            cell_bot_id = cell_v_idx_th[:, 0, :][val_cell_id]
            mp_con = mse_l(
                (pred_v_glob - cell_bot_id),
                (ones_like(pred_v_glob - cell_bot_id) * 0.5),
            )
            loss_dict["mp_con"] = mp_con * cfg_dd.mp_coeff

        for k, v in loss_dict.items():
            net_loss += v
        net_loss.backward()
        optim_dec.step()

        writer.add_scalar("cd_loss", cd_loss.item(), ep * len(train_loader) + idx)

        all_mean_l["cd"].append(cd_loss.item())

    save_all()
    return net_loss


def validation_step(quant=False):
    all_scores = defaultdict(dict)
    for idx, (val_batch) in enumerate(val_loader):
        decoder.eval()
        name = val_batch["name"]
        gt_sdf, dc_f, val_cell_id = put_to_device(
            [val_batch[i] for i in ["sdf_inp", "dc_f", "val_cell_id"]]
        )
        grid_sdf = gt_sdf.reshape(grid_size + 6, grid_size + 6, grid_size + 6).cuda()
        val_cell_id = val_cell_id.long().nonzero().squeeze()

        pred_v_vox = decoder(grid_sdf.unsqueeze(0).unsqueeze(1)).squeeze()[
            :-1, :-1, :-1
        ]
        cell_bot_id = cell_v_idx_th[:, 0, :][val_cell_id]
        pred_v = pred_v_vox.reshape(-1, 3)[val_cell_id] + cell_bot_id
        m_path = osp.join(save_dir, "%s_%s_%d.ply" % (name, exp_name, ep))
        dc_m = trimesh_from_vf(numpied(pred_v), numpied(dc_f))
        _ = dc_m.export(m_path)
        try:
            mesh_name = os.listdir(osp.join(cfg_dd.mesh_dir, name))[0]
            gt_file = osp.join(cfg_dd.mesh_dir, name, mesh_name)
        except FileNotFoundError:
            gt_file = osp.join(cfg_dd.mesh_dir, "%s.obj" % name)

        if osp.isfile(gt_file):
            gt_m = load_m(gt_file)
            try:
                cur_score = eval_all(dc_m, gt_m)
                all_scores["CD"][name] = cur_score["CD"]
                all_scores["Normal"][name] = cur_score["Normal"]
                all_scores["Intersec"][name] = cur_score["Intersec"]
                cd_eval = cur_score["CD"]
                writer.add_scalar("cd_eval", cd_eval, ep * len(val_loader) + idx)
            except:
                continue
    json.dump(all_scores, open(osp.join(save_dir, "all_scores_%d.json" % ep), "w"))
    print(
        "Epoch %d, CD: %.4f, Normal: %.4f, Intersec: %.4f"
        % (
            ep,
            np.mean(list(all_scores["CD"].values())),
            np.mean(list(all_scores["Normal"].values())),
            np.mean(list(all_scores["Intersec"].values())),
        )
    )

    if not args.wo_model_save:
        save_all(epoch=ep)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = get_argparse_2(parser).parse_args()
    exp_name = args.exp_name
    writer = SummaryWriter(osp.join(os.getcwd(), "runs", exp_name))

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dec_dim = config["dec_dim"]

    name_id_dict = defaultdict(int)

    n_epoch = config["n_epoch"]
    s_epoch = config.get("start_epoch", 0)
    vox_size = config["vox_size"]
    grid_size = vox_size + 1
    save_dir = osp.join(config["results_dir"], exp_name)
    safe_make_dirs(save_dir)

    # saving config
    config_save_f = osp.join(save_dir, "%s_config.json" % exp_name)
    json.dump(args.__dict__, open(config_save_f, "w"), indent=2)

    num_workers = config["num_workers"]
    batch_size = config["batch_size"]
    epoch_thresh = config["epoch_thresh"]
    cd_coeff = config["cd_coeff"]
    sc_us = config.get("sc_us", False)
    add_noise = config.get("add_noise", False)

    cfg_dd = dotdict(config)

    ## Voxelization
    vox_ind = manual_vox_ind(grid_size).cuda().long()
    all_cell_ind = get_cell_e_idx(vox_ind)
    cell_v_idx_th = get_cell_v_idx(vox_ind)
    grid_xyz = gen_grid_vert(start=cfg_dd.v_start, end=cfg_dd.v_end, vox_size=grid_size)
    grid_xyz_th = torched(grid_xyz, device="cuda")
    grid_xyz_vox = grid_xyz_th.reshape(grid_size, grid_size, grid_size, -1)
    grid_pt_ba = grid_xyz_th.reshape(-1, 3).unsqueeze(0).cuda()
    grid_dim = (grid_xyz_vox[1, 1, 1] - grid_xyz_vox[0, 0, 0]).mean()

    # Model
    encoder_cls = getattr(encoders, config["enc_name"])
    sdfnet_cls = getattr(sdfNets, config["sdf_name"])
    decoder_cls = getattr(decoders, config["dec_name"])

    print("Using Sigmoid? %s " % str(cfg_dd.use_sig))
    dec_in_dim = cfg_dd.dec_dim
    decoder = (
        decoder_cls(
            input_dim=dec_in_dim, use_sig=cfg_dd.use_sig, sig_scale=cfg_dd.sig_scale
        )
        .float()
        .cuda()
    )

    border = "=" * 50
    clear_border = _term_move_up() + "\r" + " " * len(border) + "\r"

    # Loss
    all_mean_l = defaultdict(list)
    l2_loss = torch.nn.L1Loss(reduce="mean")
    mse_l = torch.nn.MSELoss()
    pred_f_fix = None
    val_dir = cfg_dd.get("val_dir", cfg_dd.gt_dir)
    train_set = ABC_grid_hdf5(
        cfg_dd.gt_dir,
        cfg_dd.mesh_dir,
        vox_size,
        cfg_dd.grid_pad,
        train=True,
        n_surf_pt=cfg_dd.n_surf_pt,
        get_curvature=cfg_dd.use_curvature,
        get_vf=cfg_dd.use_vf,
        use_eCD=cfg_dd.ecd_coeff > 0.0,
        get_offsur=cfg_dd.p2f_offsur_coeff > 0.0,
        split_file=cfg_dd.get("tr_split_file", None),
        sc_us=sc_us,
        add_noise=add_noise,
    )

    valset = ABC_grid_hdf5(
        val_dir,
        cfg_dd.mesh_dir,
        vox_size,
        cfg_dd.grid_pad,
        train=False,
        split_file=cfg_dd.get("te_split_file", None),
        sc_us=sc_us,
    )
    train_loader = DataLoader(
        train_set,
        pin_memory=True,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=num_workers > 0,
        shuffle=True,
        collate_fn=my_collate,
    )
    val_loader = DataLoader(
        valset,
        num_workers=num_workers,
        batch_size=1,
        persistent_workers=num_workers > 0,
        shuffle=False,
        collate_fn=my_collate,
    )

    optim_dec = torch.optim.Adam(
        [{"params": decoder.parameters()}], lr=float(cfg_dd.lr)
    )
    scheduler_dec = MultiStepLR(optim_dec, milestones=cfg_dd.optim_step, gamma=0.5)

    if cfg_dd.dec_weight is not None:
        load_decoder(cfg_dd.dec_weight)

    advance_scheduler()

    with torch.autograd.set_detect_anomaly(True):
        for ep in tqdm(range(s_epoch, n_epoch), initial=s_epoch, total=n_epoch):
            if ep % cfg_dd.val_every == 0:
                validation_step()
            net_loss = training_step()

            scheduler_dec.step()

    writer.close()

    validation_step(quant=True)

    plt.title("Loss Prog Optim")
    for k, v in all_mean_l.items():
        plt.plot(range(len(v)), v, label=k)
        plt.legend(loc="best")
        plt.savefig(osp.join(save_dir, "LossProg_%s.png" % k))
        plt.clf()
    plt.close()
