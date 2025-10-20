# Thanks to the authors of NDC

import numpy as np
import h5py


def augment_data_ndc(
    LOD_gt_float,
    LOD_input,
    grid_size,
    aug_permutation=True,
    aug_reversal=True,
    aug_inversion=True,
):
    grid_size_1 = grid_size + 1
    grid_sdf_aug = LOD_input.copy()
    newdict = {}
    LOD_gt_int = None

    newdict["float_center_x_"] = LOD_gt_float[:-1, :-1, :-1, 0]
    newdict["float_center_y_"] = LOD_gt_float[:-1, :-1, :-1, 1]
    newdict["float_center_z_"] = LOD_gt_float[:-1, :-1, :-1, 2]

    newdict["input_sdf"] = LOD_input[:, :, :]

    # augment data
    permutation_list = [
        [0, 1, 2],
        [0, 2, 1],
        [1, 0, 2],
        [1, 2, 0],
        [2, 0, 1],
        [2, 1, 0],
    ]
    reversal_list = [
        [0, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        [0, 1, 1],
        [1, 0, 0],
        [1, 0, 1],
        [1, 1, 0],
        [1, 1, 1],
    ]
    if aug_permutation:
        permutation = permutation_list[np.random.randint(len(permutation_list))]
    else:
        permutation = permutation_list[0]
    if aug_reversal:
        reversal = reversal_list[np.random.randint(len(reversal_list))]
    else:
        reversal = reversal_list[0]
    if aug_inversion:
        inversion_flag = np.random.randint(2)
    else:
        inversion_flag = 0

    if reversal[0]:
        for k in newdict:  # reverse
            newdict[k] = newdict[k][::-1, :, :]
            grid_sdf_aug = grid_sdf_aug[::-1, :, :]

        k = "float_center_x_"
        mask = newdict[k] >= 0
        newdict[k] = newdict[k] * (1 - mask) + (1 - newdict[k]) * mask
    if reversal[1]:
        for k in newdict:  # reverse
            newdict[k] = newdict[k][:, ::-1, :]
            grid_sdf_aug = grid_sdf_aug[:, ::-1, :]

        k = "float_center_y_"
        mask = newdict[k] >= 0
        newdict[k] = newdict[k] * (1 - mask) + (1 - newdict[k]) * mask
    if reversal[2]:
        for k in newdict:  # reverse
            newdict[k] = newdict[k][:, :, ::-1]
            grid_sdf_aug = grid_sdf_aug[:, :, ::-1]

        k = "float_center_z_"
        mask = newdict[k] >= 0
        newdict[k] = newdict[k] * (1 - mask) + (1 - newdict[k]) * mask

    if permutation == [0, 1, 2]:
        pass
    else:
        for k in newdict:  # transpose
            newdict[k] = np.transpose(newdict[k], permutation)

        grid_sdf_aug = np.transpose(grid_sdf_aug, permutation)

        olddict = newdict
        newdict = {}
        newdict["input_sdf"] = olddict["input_sdf"]

        if permutation == [0, 2, 1]:
            newdict["float_center_x_"] = olddict["float_center_x_"]
            newdict["float_center_y_"] = olddict["float_center_z_"]
            newdict["float_center_z_"] = olddict["float_center_y_"]
        elif permutation == [1, 0, 2]:
            newdict["float_center_x_"] = olddict["float_center_y_"]
            newdict["float_center_y_"] = olddict["float_center_x_"]
            newdict["float_center_z_"] = olddict["float_center_z_"]
        elif permutation == [2, 1, 0]:
            newdict["float_center_x_"] = olddict["float_center_z_"]
            newdict["float_center_y_"] = olddict["float_center_y_"]
            newdict["float_center_z_"] = olddict["float_center_x_"]
        elif permutation == [1, 2, 0]:
            newdict["float_center_x_"] = olddict["float_center_y_"]
            newdict["float_center_y_"] = olddict["float_center_z_"]
            newdict["float_center_z_"] = olddict["float_center_x_"]
        elif permutation == [2, 0, 1]:
            newdict["float_center_x_"] = olddict["float_center_z_"]
            newdict["float_center_y_"] = olddict["float_center_x_"]
            newdict["float_center_z_"] = olddict["float_center_y_"]

    LOD_gt_int = None

    LOD_gt_float = np.full([grid_size_1, grid_size_1, grid_size_1, 3], -1, np.float32)
    LOD_gt_float[:-1, :-1, :-1, 0] = newdict["float_center_x_"]
    LOD_gt_float[:-1, :-1, :-1, 1] = newdict["float_center_y_"]
    LOD_gt_float[:-1, :-1, :-1, 2] = newdict["float_center_z_"]

    LOD_input = np.ones([grid_size_1, grid_size_1, grid_size_1], np.float32)
    LOD_input[:, :, :] = newdict["input_sdf"]
    if inversion_flag:
        LOD_input = -LOD_input

    return LOD_gt_int, LOD_gt_float, LOD_input, grid_sdf_aug


def read_data(hdf5_dir, grid_size):
    hdf5_file = h5py.File(hdf5_dir, "r")
    LOD_gt_int = None
    # LOD_gt_float = hdf5_file[str(grid_size)+"_float"][:].astype(np.float32)
    LOD_input = hdf5_file[str(grid_size) + "_sdf"][:]
    LOD_input = LOD_input * grid_size  # denormalize
    hdf5_file.close()
    return LOD_input


def pad_inp_test(gt_input_, grid_pad=3, grid_size=64):
    gt_input_ = np.expand_dims(gt_input_, axis=0).astype(np.float32)
    xmin = 0
    xmax = grid_size + 1
    ymin = 0
    ymax = grid_size + 1
    zmin = 0
    zmax = grid_size + 1

    xmin = xmin - grid_pad
    xmax = xmax + grid_pad
    ymin = ymin - grid_pad
    ymax = ymax + grid_pad
    zmin = zmin - grid_pad
    zmax = zmax + grid_pad

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
    if xmax > grid_size + 1:
        xmax_pad += grid_size + 1 - xmax
        xmax = grid_size + 1
    if ymin < 0:
        ymin_pad -= ymin
        ymin = 0
    if ymax > grid_size + 1:
        ymax_pad += grid_size + 1 - ymax
        ymax = grid_size + 1
    if zmin < 0:
        zmin_pad -= zmin
        zmin = 0
    if zmax > grid_size + 1:
        zmax_pad += grid_size + 1 - zmax
        zmax = grid_size + 1

    gt_input[:, xmin_pad:xmax_pad, ymin_pad:ymax_pad, zmin_pad:zmax_pad] = gt_input_[
        :, xmin:xmax, ymin:ymax, zmin:zmax
    ]

    return gt_input[0]
