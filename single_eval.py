import argparse
import os
from glob import glob
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm

from Networks import decoders
from aug_utils import pad_inp_test
from dc_utils import dual_contouring_NDC
from utils import numpied, safe_make_dirs, trimesh_from_vf


def load_decoder(dec_name, path, dec_in_dim, use_sig, sig_scale):
    """Load a trained decoder model."""
    decoder_cls = getattr(decoders, dec_name)
    decoder = (
        decoder_cls(input_dim=dec_in_dim, use_sig=use_sig, sig_scale=sig_scale)
        .float()
        .cuda()
    )
    loaded_th = torch.load(path)
    decoder.load_state_dict(loaded_th["decoder_sd"])
    decoder.eval()
    print(f"Loaded decoder from {path}")
    return decoder


def evaluate(args):
    """Run the evaluation script."""
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    decoder = load_decoder(
        args.dec_name, args.dec_weight, args.dec_in_dim, args.use_sig, args.sig_scale
    )

    if args.split_file:
        eval_names = [i.rstrip() for i in open(args.split_file).readlines()]
        all_hdf5 = [os.path.join(args.data_dir, f"{name}.hdf5") for name in eval_names]
    else:
        all_hdf5 = sorted(list(Path(args.data_dir).glob("*.hdf5")))

    safe_make_dirs(args.save_dir)

    for hdf5_path in tqdm(all_hdf5):
        with h5py.File(hdf5_path, "r") as my_h5:
            grid_sdf = np.array(my_h5[f"{args.grid_size}_sdf"])

        npz_path = os.path.join(args.data_dir, f"{Path(hdf5_path).stem}.npz")
        np_data = np.load(npz_path)
        dc_f = np_data["dc_f"].astype(np.int32)

        grid_size_float = float(grid_sdf.shape[0] - 1)
        grid_sdf = np.clip(grid_sdf * grid_size_float, -args.clip_val, args.clip_val)
        grid_sdf_pad = pad_inp_test(
            grid_sdf, grid_pad=args.grid_pad, grid_size=args.grid_size
        )

        with torch.no_grad():
            out_grid = decoder(
                torch.from_numpy(grid_sdf_pad).cuda().unsqueeze(0).unsqueeze(0).float()
            )

        dc_int_vox = np.where(
            grid_sdf > 0.0, np.zeros_like(grid_sdf), np.ones_like(grid_sdf)
        )
        pred_v, pred_f = dual_contouring_NDC(dc_int_vox, numpied(out_grid.squeeze()))
        mesh = trimesh_from_vf(pred_v, pred_f)
        mesh.export(os.path.join(args.save_dir, f"{Path(hdf5_path).stem}.obj"))


def main():
    """Main function to run the evaluation."""
    parser = argparse.ArgumentParser(description="Single model evaluation script.")
    parser.add_argument(
        "--dec_name", type=str, default="CNN_3d_rec7Def", help="Decoder class name."
    )
    parser.add_argument(
        "--dec_weight", type=str, required=True, help="Path to decoder weights."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to HDF5 data directory."
    )
    parser.add_argument(
        "--save_dir", type=str, required=True, help="Path to save output OBJ files."
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional path to a file with names for evaluation.",
    )
    parser.add_argument("--gpu", type=str, default="0", help="GPU to use.")
    parser.add_argument("--grid_size", type=int, default=128, help="Grid size.")
    parser.add_argument(
        "--clip_val", type=float, default=3.5, help="SDF clipping value."
    )
    parser.add_argument("--grid_pad", type=int, default=3, help="Padding for the grid.")
    parser.add_argument(
        "--dec_in_dim", type=int, default=256, help="Decoder input dimension."
    )
    parser.add_argument(
        "--use_sig", action="store_true", help="Use sigmoid in decoder."
    )
    parser.add_argument(
        "--no-use_sig",
        dest="use_sig",
        action="store_false",
        help="Do not use sigmoid in decoder.",
    )
    parser.set_defaults(use_sig=True)
    parser.add_argument(
        "--sig_scale", type=float, default=10.0, help="Sigmoid scale factor."
    )

    args = parser.parse_args()

    evaluate(args)


if __name__ == "__main__":
    main()
