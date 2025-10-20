import os
import os.path as osp
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
import torch
import trimesh
from torch.utils.data import Dataset
from scipy.spatial import cKDTree as KDTree
from pytorch3d.ops import packed_to_padded

from aug_utils import read_data, read_data_w_noise
from utils import (
    f_from_m,
    gen_grid_vert,
    sample_along_edge,
    scale_to_unit_cube,
    v_from_m,
)


class ABC_grid_hdf5(Dataset):
    """Dataset for loading ABC grid data from HDF5 files."""

    def __init__(
        self,
        data_dir,
        mesh_dir,
        vox_size,
        grid_pad,
        train,
        n_surf_pt=50000,
        get_curvature=False,
        get_vf=False,
        use_eCD=False,
        get_offsur=False,
        n_offsur_pt=50000,
        split_file=None,
        sc_us=False,
        add_noise=False,
    ):
        self.data_dir = data_dir
        self.mesh_dir = mesh_dir
        self.vox_size = vox_size
        self.grid_pad = grid_pad
        self.train = train
        self.n_surf_pt = n_surf_pt
        self.get_curvature = get_curvature
        self.get_vf = get_vf
        self.get_offsur = get_offsur
        self.use_eCD = use_eCD
        self.grid_size = vox_size
        self.n_offsur_pt = n_offsur_pt
        self.sc_us = sc_us
        self.add_noise = add_noise

        if split_file:
            self.hdf5_names = [i.rstrip() for i in open(split_file, "r").readlines()]
        else:
            self.hdf5_names = sorted(
                [Path(i).stem for i in Path(self.data_dir).glob("*.hdf5")]
            )

        self.npz_suffix = "_20k" if get_curvature else ""
        mode = "Train" if self.train else "Val"
        print("Mode %s: %d Shapes" % (mode, len(self.hdf5_names)))
        temp_hdf5_names = []
        for name in self.hdf5_names:
            h5py.File(self.data_dir + "/" + name + ".hdf5", "r")
            temp_hdf5_names.append(name)

        self.hdf5_names = temp_hdf5_names

        self.mesh_dict = {}
        if self.get_vf or self.use_eCD:
            print("Loading Meshes")
            for name in self.hdf5_names:
                mesh_name = os.listdir(osp.join(self.mesh_dir, name))[0]
                self.mesh_dict[name] = trimesh.load(
                    osp.join(self.mesh_dir, name, mesh_name)
                )

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return len(self.hdf5_names)

    def __getitem__(self, index):
        """Return a single sample from the dataset."""
        hdf5_file = osp.join(self.data_dir, self.hdf5_names[index] + ".hdf5")
        grid_size = self.grid_size

        if self.add_noise and self.train and np.random.randint(4) > 0:
            gt_input_ = read_data_w_noise(hdf5_file, grid_size)
        else:
            gt_input_ = read_data(hdf5_file, grid_size)

        np_data = np.load(
            osp.join(self.data_dir, self.hdf5_names[index] + self.npz_suffix + ".npz")
        )
        dc_f = np_data["dc_f"].astype(np.int32)
        val_cell_id = np_data["val_cell_id"].astype(np.int32)

        if self.use_eCD:
            edge_pt = np.zeros_like(surf_pt)
            cur_m = self.mesh_dict[self.hdf5_names[index]]
            unique_e = cur_m.edges_unique
            if unique_e.shape[0] > 30000:
                unique_e = unique_e[
                    np.random.choice(unique_e.shape[0], 30000, replace=False)
                ]
            edge_pt = sample_along_edge(
                scale_to_unit_cube(v_from_m(cur_m)), unique_e, samp_per_e=3
            )
            if self.sc_us:
                edge_pt = (edge_pt + 1.0) * grid_size / 2.0
            else:
                edge_pt = (edge_pt + 0.5) * grid_size
            edge_pt = np.clip(edge_pt, 0, edge_pt)

        if self.sc_us:
            surf_pt = np_data["samp_sc_us"].astype(np.float32)
            surf_pt_us = np.copy(surf_pt)
            surf_idx = np.random.choice(
                surf_pt.shape[0], min(self.n_surf_pt, surf_pt.shape[0]), replace=False
            )
            surf_pt = surf_pt[surf_idx]
            surf_pt_glob = (surf_pt + 1.0) * grid_size / 2.0
        else:
            surf_pt = np_data["samp"].astype(np.float32)
            surf_pt_us = np.copy(surf_pt)
            surf_idx = np.random.choice(
                surf_pt.shape[0], min(self.n_surf_pt, surf_pt.shape[0]), replace=False
            )
            surf_pt = surf_pt[surf_idx]
            surf_pt_glob = (surf_pt + 0.5) * grid_size
        if np.random.randint(2) == 1 and self.train:
            gt_input_ = gt_input_ * -1

        sdf_inp_f = self.pad_inp_test(gt_input_)
        sdf_inp, pad_tup, dilated_tup, dilated_pad_tup = self.pad_inp_train(
            gt_input_, val_cell_id
        )
        sdf_inp_wocl = sdf_inp.copy()
        sdf_inp = np.clip(sdf_inp, -3.5, 3.5)

        ret_dict = {
            "surf_pt_glob": surf_pt_glob,
            "sdf_inp": sdf_inp,
            "surf_pt_us": surf_pt_us,
            "dc_f": dc_f,
            "val_cell_id": val_cell_id,
            "name": self.hdf5_names[index],
            "pad_tup": np.asarray(pad_tup).astype(np.int32),
            "dilated_tup": np.asarray(dilated_tup).astype(np.int32),
            "dilated_pad_tup": np.asarray(dilated_pad_tup).astype(np.int32),
            "sdf_inp_wocl": sdf_inp_wocl,
            "sdf_inp_f": sdf_inp_f,
            "dc_i": np_data["dc_i"].astype(np.int32),
        }

        if self.get_vf:
            gt_m = self.mesh_dict[self.hdf5_names[index]]
            gt_v, gt_f = np.array(gt_m.vertices), np.array(gt_m.faces)
            gt_v = (0.5 + scale_to_unit_cube(gt_v)) * grid_size
            ret_dict["gt_v"] = gt_v
            ret_dict["gt_f"] = gt_f
            gt_surf_norm = np_data["norm"][surf_idx]
            ret_dict["gt_surf_norm"] = gt_surf_norm
            dc_e = np_data["dc_e"].astype(np.int32)
            ret_dict["dc_e"] = dc_e
            ret_dict["edge_pt"] = edge_pt

        if self.get_curvature:
            ret_dict["gt_e1"] = np_data["e1"][surf_idx]
            ret_dict["gt_e2"] = np_data["e2"][surf_idx]
            ret_dict["gt_rho_1"] = np_data["rho_1"][surf_idx]
            ret_dict["gt_rho_2"] = np_data["rho_2"][surf_idx]

        if self.get_offsur:
            sdf_pts = np_data["sdf_pts"]
            offsur_idx = np.random.choice(
                sdf_pts.shape[0], min(self.n_offsur_pt, sdf_pts.shape[0]), replace=False
            )
            ret_dict["sdf_pts"] = sdf_pts[offsur_idx]
            ret_dict["sdf_vals"] = np_data["sdf_vals"][offsur_idx]
        return ret_dict

    def pad_inp_test(self, gt_input_):
        """Pad the input for testing."""
        gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)
        xmin = 0
        xmax = self.grid_size + 1
        ymin = 0
        ymax = self.grid_size + 1
        zmin = 0
        zmax = self.grid_size + 1

        xmin = xmin - self.grid_pad
        xmax = xmax + self.grid_pad
        ymin = ymin - self.grid_pad
        ymax = ymax + self.grid_pad
        zmin = zmin - self.grid_pad
        zmax = zmax + self.grid_pad

        xmin_pad = 0
        xmax_pad = xmax - xmin
        ymin_pad = 0
        ymax_pad = ymax - ymin
        zmin_pad = 0
        zmax_pad = zmax - zmin

        if gt_input_[0, 0, 0, 0] > 0:
            gt_input = np.full([1, xmax_pad, ymax_pad, zmax_pad], 10, np.float32)
        else:
            gt_input = np.full([1, xmax_pad, ymax_pad, zmax_pad], -10, np.float32)

        if xmin < 0:
            xmin_pad -= xmin
            xmin = 0
        if xmax > self.grid_size + 1:
            xmax_pad += self.grid_size + 1 - xmax
            xmax = self.grid_size + 1
        if ymin < 0:
            ymin_pad -= ymin
            ymin = 0
        if ymax > self.grid_size + 1:
            ymax_pad += self.grid_size + 1 - ymax
            ymax = self.grid_size + 1
        if zmin < 0:
            zmin_pad -= zmin
            zmin = 0
        if zmax > self.grid_size + 1:
            zmax_pad += self.grid_size + 1 - zmax
            zmax = self.grid_size + 1

        gt_input[:, xmin_pad:xmax_pad, ymin_pad:ymax_pad, zmin_pad:zmax_pad] = (
            gt_input_[:, xmin:xmax, ymin:ymax, zmin:zmax]
        )

        return gt_input[0]

    def pad_inp_train(self, gt_input_, val_cell_id):
        """Pad the input for training."""
        gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)

        gt_mask = np.zeros_like(gt_input_.squeeze())
        val_cell = np.zeros((self.grid_size, self.grid_size, self.grid_size))

        val_cell_unrav = np.unravel_index(
            torch.from_numpy(val_cell_id).nonzero().squeeze().numpy(),
            (self.grid_size, self.grid_size, self.grid_size),
        )
        val_cell[val_cell_unrav[0], val_cell_unrav[1], val_cell_unrav[2]] = 1
        gt_mask[:-1, :-1, :-1] = val_cell

        # x
        ray = np.max(gt_mask, (1, 2))
        xmin = -1
        xmax = -1
        for i in range(self.grid_size + 1):
            if ray[i] > 0:
                if xmin == -1:
                    xmin = i
                xmax = i
        # y
        ray = np.max(gt_mask, (0, 2))
        ymin = -1
        ymax = -1
        for i in range(self.grid_size + 1):
            if ray[i] > 0:
                if ymin == -1:
                    ymin = i
                ymax = i
        # z
        ray = np.max(gt_mask, (0, 1))
        zmin = -1
        zmax = -1
        for i in range(self.grid_size + 1):
            if ray[i] > 0:
                if zmin == -1:
                    zmin = i
                zmax = i
        xmax += 1
        ymax += 1
        zmax += 1

        x_min_o, x_max_o, y_min_o, y_max_o, z_min_o, z_max_o = [
            deepcopy(i) for i in [xmin, xmax, ymin, ymax, zmin, zmax]
        ]

        xmin = xmin - self.grid_pad
        xmax = xmax + self.grid_pad
        ymin = ymin - self.grid_pad
        ymax = ymax + self.grid_pad
        zmin = zmin - self.grid_pad
        zmax = zmax + self.grid_pad

        xmin_pad = 0
        xmax_pad = xmax - xmin
        ymin_pad = 0
        ymax_pad = ymax - ymin
        zmin_pad = 0
        zmax_pad = zmax - zmin

        if gt_input_[0, 0, 0, 0] > 0:
            gt_input = np.full([1, xmax_pad, ymax_pad, zmax_pad], 10, np.float32)
        else:
            gt_input = np.full([1, xmax_pad, ymax_pad, zmax_pad], -10, np.float32)

        if xmin < 0:
            xmin_pad -= xmin
            xmin = 0
        if xmax > self.grid_size + 1:
            xmax_pad += self.grid_size + 1 - xmax
            xmax = self.grid_size + 1
        if ymin < 0:
            ymin_pad -= ymin
            ymin = 0
        if ymax > self.grid_size + 1:
            ymax_pad += self.grid_size + 1 - ymax
            ymax = self.grid_size + 1
        if zmin < 0:
            zmin_pad -= zmin
            zmin = 0
        if zmax > self.grid_size + 1:
            zmax_pad += self.grid_size + 1 - zmax
            zmax = self.grid_size + 1

        gt_input[:, xmin_pad:xmax_pad, ymin_pad:ymax_pad, zmin_pad:zmax_pad] = (
            gt_input_[:, xmin:xmax, ymin:ymax, zmin:zmax]
        )

        x_min_p, x_max_p, y_min_p, y_max_p, z_min_p, z_max_p = [
            deepcopy(i) for i in [xmin, xmax, ymin, ymax, zmin, zmax]
        ]
        x_min_dp, x_max_dp, y_min_dp, y_max_dp, z_min_dp, z_max_dp = [
            deepcopy(i)
            for i in [xmin_pad, xmax_pad, ymin_pad, ymax_pad, zmin_pad, zmax_pad]
        ]
        dil_inp = (x_min_p, x_max_p, y_min_p, y_max_p, z_min_p, z_max_p)
        dil_pad_inp = (x_min_dp, x_max_dp, y_min_dp, y_max_dp, z_min_dp, z_max_dp)
        return (
            gt_input[0],
            (x_min_o, x_max_o, y_min_o, y_max_o, z_min_o, z_max_o),
            dil_inp,
            dil_pad_inp,
        )

    def apply_aug(self, surf_pt):
        """Apply augmentation to the surface points."""
        raise NotImplementedError(
            "This function is not used and contains undefined variables."
        )


def my_collate(batch):
    """Collate function for the DataLoader."""
    if len(batch) == 1:
        return {
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v
            for k, v in batch[0].items()
        }

    out_dict = defaultdict(list)
    for b in batch:
        for k, v in b.items():
            out_dict[k].append(v)
        out_dict["f_len"].append(len(b["dc_f"]))
    face_cat = np.concatenate(out_dict["dc_f"])
    n_faces_cumsum = torch.from_numpy(np.cumsum([0] + out_dict["f_len"]))
    faces_padded = packed_to_padded(
        torch.from_numpy(face_cat).float(), n_faces_cumsum[:-1], max(out_dict["f_len"])
    )
    out_dict["dc_f"] = faces_padded

    for k, v in out_dict.items():
        if isinstance(v[0], np.ndarray) or isinstance(v[0], list):
            out_dict[k] = torch.from_numpy(np.stack(v)).squeeze()
        else:
            out_dict[k] = v
    return out_dict


class PointCloudDataset(Dataset):
    """Dataset for loading point cloud data from HDF5 files."""

    def __init__(
        self,
        split_f,
        data_dir,
        vox_size,
        v_start=-0.5,
        v_end=0.5,
        n_surf_pt=30000,
        n_sdf_pt=8192 * 2,
        n_inp_pt=4096,
        is_val=False,
        KNN_num=8,
        pooling_radius=2,
        use_trunc=False,
        get_surfpts=False,
        n_enc_pc=4096,
    ):

        super().__init__()
        self.data_dir = data_dir
        self.get_surfpts = get_surfpts
        split_files = [i.rstrip() for i in open(split_f, "r").readlines()]
        self.hdf5_file_tuples = [
            (i, h5py.File(osp.join(self.data_dir, i + ".hdf5"), "r"))
            for i in split_files
        ]

        # self.surf_files = [i for i in sorted(glob(osp.join(surf_dir, '*.npy')))]
        # self.sdf_files = [osp.join(sdf_dir, Path(i).stem + '.npz') for i in self.surf_files]
        assert len(self.hdf5_file_tuples) != 0, "No objects in the data directory"
        self.num_instances = len(self.hdf5_file_tuples)

        self.vox_size = vox_size
        self.grid_size = vox_size + 1
        self.use_trunc = use_trunc

        self.KNN_num = KNN_num
        self.pooling_radius = pooling_radius
        self.v_start = v_start
        self.v_end = v_end
        self.n_sdf_pt = n_sdf_pt
        self.n_surf_pt = n_surf_pt
        self.is_val = is_val
        self.n_inp_pt = n_inp_pt
        self.n_enc_pc = n_enc_pc

        # NOTE: Model and SDF can be different because we can train with 64^3
        self.grid_xyz = gen_grid_vert(start=v_start, end=v_end, vox_size=self.grid_size)
        self.grid_xyz_vox = self.grid_xyz.reshape(
            self.grid_size, self.grid_size, self.grid_size, 3
        )
        self.grid_xyz_fl = self.grid_xyz.reshape(-1, 3)

    def get_nn_info(self, pc_xyz):
        """Get nearest neighbor information for the point cloud."""
        kd_tree = KDTree(pc_xyz)
        _, pc_KNN_idx = kd_tree.query(pc_xyz, k=self.KNN_num)
        pc_KNN_idx = np.reshape(pc_KNN_idx, [-1])
        pc_KNN_xyz = pc_xyz[pc_KNN_idx]
        pc_KNN_xyz = np.reshape(
            pc_KNN_xyz, [len(pc_xyz), self.KNN_num, 3]
        ) - np.reshape(pc_xyz, [len(pc_xyz), 1, 3])
        pc_KNN_xyz = np.reshape(pc_KNN_xyz, [len(pc_xyz) * self.KNN_num, 3])
        return pc_KNN_xyz, pc_KNN_idx

    def to_int_coord(self, xyz):
        """Convert world coordinates to integer grid coordinates."""
        min_val = -1 * self.v_start
        max_val = self.v_end - self.v_start
        size = self.vox_size
        int_coord = ((xyz + min_val) / max_val) * size
        return int_coord

    def get_voxel_info(self, pc_xyz):
        """Get voxel information for the point cloud."""
        kd_tree = KDTree(pc_xyz)
        pc_xyz_int = np.floor(pc_xyz).astype(np.int32)
        pc_xyz_int = np.clip(pc_xyz_int, 0, self.vox_size)

        tmp_grid = np.zeros(
            [self.vox_size + 1, self.vox_size + 1, self.vox_size + 1], np.uint8
        )
        tmp_grid[pc_xyz_int[:, 0], pc_xyz_int[:, 1], pc_xyz_int[:, 2]] = 1
        for ite in range(self.pooling_radius):
            tmp_mask = np.copy(tmp_grid[1:-1, 1:-1, 1:-1])
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        tmp_grid[
                            i : self.vox_size - 1 + i,
                            j : self.vox_size - 1 + j,
                            k : self.vox_size - 1 + k,
                        ] = (
                            tmp_mask
                            | tmp_grid[
                                i : self.vox_size - 1 + i,
                                j : self.vox_size - 1 + j,
                                k : self.vox_size - 1 + k,
                            ]
                        )
        voxel_x, voxel_y, voxel_z = np.nonzero(tmp_grid)
        voxel_xyz = np.concatenate(
            [
                np.reshape(voxel_x, [-1, 1]),
                np.reshape(voxel_y, [-1, 1]),
                np.reshape(voxel_z, [-1, 1]),
            ],
            1,
        )
        voxel_xyz = voxel_xyz.astype(np.float32) + 0.5
        voxel_xyz_int = np.floor(voxel_xyz).astype(np.int64)
        _, voxel_KNN_idx = kd_tree.query(voxel_xyz, k=self.KNN_num)
        voxel_KNN_idx = np.reshape(voxel_KNN_idx, [-1])
        voxel_KNN_xyz = pc_xyz[voxel_KNN_idx]
        voxel_KNN_xyz = np.reshape(
            voxel_KNN_xyz, [len(voxel_xyz), self.KNN_num, 3]
        ) - np.reshape(voxel_xyz, [len(voxel_xyz), 1, 3])
        voxel_KNN_xyz = np.reshape(voxel_KNN_xyz, [len(voxel_xyz) * self.KNN_num, 3])

        return voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz

    def __len__(self):
        """Return the total number of samples in the dataset."""
        return self.num_instances

    def __getitem__(self, index):
        """Return a single sample from the dataset."""
        cur_name = self.hdf5_file_tuples[index][0]
        pts_sdf_vox = self.hdf5_file_tuples[index][1]["%d_sdf" % self.vox_size][:]
        pts_sdf_fl = pts_sdf_vox.reshape(-1)

        if not self.use_trunc:
            pos_idx = (pts_sdf_fl > 0.0).nonzero()[0]
            neg_idx = (pts_sdf_fl < 0.0).nonzero()[0]
            n_pos_pt = int(0.6 * self.n_sdf_pt)
            rand_pos_idx = pos_idx[
                np.random.choice(np.arange(pos_idx.shape[0]), n_pos_pt, replace=True)
            ]
            if not neg_idx.shape[0] == 0:
                rand_neg_idx = neg_idx[
                    np.random.choice(
                        np.arange(neg_idx.shape[0]),
                        self.n_sdf_pt - n_pos_pt,
                        replace=True,
                    )
                ]
            else:
                rand_neg_idx = pos_idx[
                    np.random.choice(
                        np.arange(pos_idx.shape[0]),
                        self.n_sdf_pt - n_pos_pt,
                        replace=True,
                    )
                ]

            rand_sdf_idx = np.r_[rand_pos_idx, rand_neg_idx]
            np.random.shuffle(rand_sdf_idx)
            sdf_val = pts_sdf_fl[rand_sdf_idx]
            sdf_pts = self.grid_xyz_fl[rand_sdf_idx]

        pts_sur = self.hdf5_file_tuples[index][1]["pointcloud"][:].astype(np.float32)

        rand_sur_idx = np.random.choice(
            np.arange(pts_sur.shape[0]), self.n_surf_pt, replace=False
        )
        surf_pts = pts_sur[rand_sur_idx][:, :3]

        surf_pts_glob = self.to_int_coord(surf_pts)[: self.n_inp_pt]
        surf_pts_enc_in = self.to_int_coord(surf_pts)[: self.n_enc_pc]
        pc_KNN_xyz, pc_KNN_idx = self.get_nn_info(surf_pts_enc_in)
        voxel_xyz_int, voxel_KNN_idx, voxel_KNN_xyz = self.get_voxel_info(
            surf_pts_enc_in
        )

        if self.is_val:
            sdf_pts = self.grid_xyz_fl
            LOD_input = self.hdf5_file_tuples[index][1]["pointcloud"][:].astype(
                np.float32
            )
            sdf_pts = (LOD_input + 0.5) * self.vox_size  # denormalize
            sdf_val = pts_sdf_vox

        if self.use_trunc:
            sdf_pts = self.grid_xyz_vox[
                voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2], :
            ]
            sdf_val = pts_sdf_vox[
                voxel_xyz_int[:, 0], voxel_xyz_int[:, 1], voxel_xyz_int[:, 2]
            ]

        ret_tuple = (
            sdf_pts,
            sdf_val,
            pc_KNN_xyz,
            pc_KNN_idx,
            voxel_xyz_int,
            voxel_KNN_idx,
            voxel_KNN_xyz,
        )
        if self.get_surfpts:
            ret_tuple = ret_tuple + (surf_pts_glob,)
        ret_tuple = ret_tuple + (cur_name,)
        return ret_tuple
