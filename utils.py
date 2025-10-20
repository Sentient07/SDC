
import os
import os.path as osp

import numpy as np
import torch
import trimesh
from pytorch3d import _C
from pytorch3d.loss.point_mesh_distance import \
    _PointEdgeDistance as PointEdgeDistance
from pytorch3d.loss.point_mesh_distance import \
    _PointFaceDistance as PointFaceDistance
from pytorch3d.ops import knn_points
from pytorch3d.structures import Meshes
from scipy.spatial import cKDTree as KDTree
from torch import FloatTensor
from torch.linalg import norm as th_norm
from torch_scatter import scatter_mean


def face_area(trig):
    """Calculates the area of a set of triangles using the cross product method."""
    vec_A = trig[:, 1, :] - trig[:, 0, :]
    vec_B = trig[:, 2, :] - trig[:, 0, :]

    raw_normal = torch.cross(vec_A, vec_B, dim=-1)
    return 0.5 * torch.norm(raw_normal, dim=len(raw_normal.shape) - 1)


def face_area_heron(trig):
    """Calculates the area of a set of triangles using Heron's formula."""
    sides = torch.zeros((trig.shape[0], 3)).to(trig.device)
    sides[:, 0] = torch.norm(trig[:, 0, :] - trig[:, 1, :], dim=-1)
    sides[:, 1] = torch.norm(trig[:, 1, :] - trig[:, 2, :], dim=-1)
    sides[:, 2] = torch.norm(trig[:, 2, :] - trig[:, 0, :], dim=-1)
    s = sides.sum(dim=-1) / 2
    areas = torch.sqrt(s * (s - sides[:, 0]) * (s - sides[:, 1]) * (s - sides[:, 2]))
    return areas


def scale_to_unit_cube(vertices, ret_scale=False):
    """Scales a set of vertices to fit within a unit cube."""
    x_max = np.max(vertices[:, 0])
    y_max = np.max(vertices[:, 1])
    z_max = np.max(vertices[:, 2])
    x_min = np.min(vertices[:, 0])
    y_min = np.min(vertices[:, 1])
    z_min = np.min(vertices[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = np.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

    vertices[:, 0] = (vertices[:, 0] - x_mid) / scale
    vertices[:, 1] = (vertices[:, 1] - y_mid) / scale
    vertices[:, 2] = (vertices[:, 2] - z_mid) / scale
    if ret_scale:
        return vertices, scale
    return vertices


def scale_to_unit_cube_th(vertices):
    """Scales a set of PyTorch tensor vertices to fit within a unit cube."""
    x_max = torch.max(vertices[:, 0])
    y_max = torch.max(vertices[:, 1])
    z_max = torch.max(vertices[:, 2])
    x_min = torch.min(vertices[:, 0])
    y_min = torch.min(vertices[:, 1])
    z_min = torch.min(vertices[:, 2])
    x_mid = (x_max + x_min) / 2
    y_mid = (y_max + y_min) / 2
    z_mid = (z_max + z_min) / 2
    x_scale = x_max - x_min
    y_scale = y_max - y_min
    z_scale = z_max - z_min
    scale = torch.sqrt(x_scale * x_scale + y_scale * y_scale + z_scale * z_scale)

    vertices[:, 0] = (vertices[:, 0] - x_mid) / scale
    vertices[:, 1] = (vertices[:, 1] - y_mid) / scale
    vertices[:, 2] = (vertices[:, 2] - z_mid) / scale
    return vertices


def numpied(th_tensor):
    """Converts a PyTorch tensor to a NumPy array."""
    if isinstance(th_tensor, np.ndarray):
        return th_tensor
    elif isinstance(th_tensor, torch.Tensor):
        return th_tensor.detach().cpu().numpy()
    else:
        raise ValueError("Unknown format")


def normalize(x: torch.Tensor, eps: float = 1e-8):
    """Normalizes a PyTorch tensor along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def moller_trumbore(ray_o: torch.Tensor, ray_d: torch.Tensor, tris: torch.Tensor, eps=1e-8):
    """Implements the Moller-Trumbore algorithm for ray-triangle intersection."""
    E1 = tris[:, 1] - tris[:, 0]
    E2 = tris[:, 2] - tris[:, 0]

    N = torch.cross(E1, E2)

    invdet = 1. / -(torch.einsum('md,nd->mn', ray_d, N) + eps)

    A0 = ray_o[:, None] - tris[None, :, 0]
    DA0 = torch.cross(A0, ray_d[:, None].expand(*A0.shape))

    u = torch.einsum('mnd,nd->mn', DA0, E2) * invdet
    v = -torch.einsum('mnd,nd->mn', DA0, E1) * invdet
    t = torch.einsum('mnd,nd->mn', A0, N) * invdet

    return u, v, t


def trimesh_from_vf(v, f):
    """Creates a Trimesh object from vertices and faces."""
    return trimesh.Trimesh(vertices=numpied(v).reshape(-1, 3), faces=numpied(f).reshape(-1, 3), process=False)


def torched(np_array, device='cuda', dtype=torch.float32):
    """Converts a NumPy array or list to a PyTorch tensor."""
    if isinstance(np_array, torch.Tensor):
        return np_array.to(device=device).to(dtype)
    if isinstance(np_array, list):
        np_array = np.array(np_array)

    if np_array.dtype == 'O':
        raise ValueError("Shouldn't be Object")
    return torch.from_numpy(np_array).to(device).to(dtype)


def get_incenter(trig):
    """Calculates the incenter of a set of triangles."""
    trig_abc_l = [th_norm(trig[:, i2, :] - trig[:, i1, :], dim=1) for i1, i2 in zip([1, 2, 0], [2, 0, 1])]
    trig_abc_th = torch.stack(trig_abc_l).T + 1e-8
    trig_abc_sum = trig_abc_th.sum(dim=1, keepdim=True)
    A = trig[:, 0, :]
    B = trig[:, 1, :]
    C = trig[:, 2, :]
    incenter = trig_abc_th[:, :1] * A + trig_abc_th[:, 1:2] * B + trig_abc_th[:, 2:3] * C
    return incenter / (trig_abc_sum + 1e-12)


def rand_unif_th(shape, start, end, device='cuda'):
    """Generates a tensor of a given shape with uniformly distributed random numbers."""
    return FloatTensor(shape).uniform_(start, end).to(device).float()


def custom_sample_trig(gen_trig, extra_sample=0):
    """Samples points within triangles using a custom strategy."""
    in_center = get_incenter(gen_trig)
    median = gen_trig.mean(dim=1)
    a_coord = gen_trig[:, 0, :]
    b_coord = gen_trig[:, 1, :]
    c_coord = gen_trig[:, 2, :]
    ab_coord = 0.5 * (a_coord + b_coord)
    bc_coord = 0.5 * (b_coord + c_coord)
    ca_coord = 0.5 * (c_coord + a_coord)

    n_pts = gen_trig.shape[0]
    r_1 = (torch.rand((n_pts), device=gen_trig.device, dtype=gen_trig.dtype) * (0.8 - 0.2) + 0.2).unsqueeze(-1)
    r_2 = (torch.rand((n_pts), device=gen_trig.device, dtype=gen_trig.dtype) * (0.8 - 0.2) + 0.2).unsqueeze(-1)
    r_3 = (torch.rand((n_pts), device=gen_trig.device, dtype=gen_trig.dtype) * (0.8 - 0.2) + 0.2).unsqueeze(-1)

    mp_1 = r_1 * a_coord + (1 - r_1) * median
    mp_2 = r_2 * b_coord + (1 - r_2) * median
    mp_3 = r_3 * c_coord + (1 - r_3) * median
    icp_1 = r_3 * ab_coord + (1 - r_3) * in_center
    icp_2 = r_1 * bc_coord + (1 - r_1) * in_center
    icp_3 = r_2 * ca_coord + (1 - r_2) * in_center

    samp_pts = [mp_1, mp_2, mp_3, icp_1, icp_2, icp_3, median,
                a_coord, b_coord, c_coord]
    samp_f_idx = torch.arange(gen_trig.shape[0]).repeat_interleave(len(samp_pts)).view(-1, 1)
    samp_pts_stack = torch.stack(samp_pts, dim=1).view(-1, 3)
    return samp_pts_stack, samp_f_idx


def safe_make_dirs(cur_dir):
    """Safely creates a directory, ignoring errors if it already exists."""
    if not osp.isdir(cur_dir):
        os.makedirs(cur_dir, exist_ok=True)


def check_is_trimesh(t_obj):
    """Checks if an object is a Trimesh or PointCloud instance."""
    if isinstance(t_obj, trimesh.Trimesh) or isinstance(t_obj, trimesh.points.PointCloud):
        return True


def v_from_m(m):
    """Extracts vertices from a mesh object."""
    if isinstance(m, str):
        return np.array(trimesh.load(m, process=False).vertices)
    elif isinstance(m, np.ndarray):
        return m
    elif check_is_trimesh(m):
        return np.array(m.vertices)
    else:
        raise ValueError("Unknown format")


def manual_vox_ind(max_size):
    """Manually generates voxel indices."""
    manual_ind = []
    for x_ind in range(int(max_size - 1)):
        for y_ind in range(int(max_size - 1)):
            for z_ind in range(int(max_size - 1)):
                manual_ind.append([x_ind, y_ind, z_ind])
    return torch.from_numpy(np.array(manual_ind)).long()


def scale_to_unit_sphere(points, center=None, buffer=1.):
    """Scales a set of points to fit within a unit sphere."""
    midpoints = (np.max(points, axis=0) + np.min(points, axis=0)) / 2
    points = points - midpoints
    scale = np.max(np.sqrt(np.sum(points ** 2, axis=1))) * buffer
    points = points / scale
    return points


def gen_grid_vert(start, end, vox_size):
    """Generates grid vertices."""
    xdim = np.linspace(start, end, vox_size)
    ydim = np.linspace(start, end, vox_size)
    zdim = np.linspace(start, end, vox_size)
    trans_grid = np.array(np.meshgrid(xdim, ydim, zdim)).T.transpose(0, 2, 1, 3)
    trans_grid[:, :, :, [0, 2]] = trans_grid[:, :, :, [2, 0]]
    grid = trans_grid.reshape(-1, 3)
    return grid


def load_m(m):
    """Loads a mesh from a file or Trimesh object."""
    if isinstance(m, trimesh.Trimesh):
        return m
    loaded_obj = trimesh.load(m, process=False)
    if isinstance(loaded_obj, trimesh.Scene):
        loaded_obj = loaded_obj.dump().sum()
    return loaded_obj


def get_closest_f_ids(gt_samp, gen_trig):
    """Finds the closest face IDs for a set of sample points."""
    pts_idx = torch.tensor([0], dtype=torch.int64, device='cuda')
    trig_idx = torch.tensor([0], dtype=torch.int64, device='cuda')
    max_points = gt_samp.shape[0]
    min_area = 0.
    nn_dist, nn_f_ids = _C.point_face_dist_forward(gt_samp, pts_idx, gen_trig, trig_idx, max_points, min_area)
    return nn_f_ids


def scale_m_to_unit_sphere(mesh, scale=1.):
    """Scales a mesh to fit within a unit sphere."""
    m = load_m(mesh)
    return trimesh_from_vf(scale_to_unit_sphere(v_from_m(m), buffer=scale), f_from_m(m))


def scale_m_to_unit_cube(mesh):
    """Scales a mesh to fit within a unit cube."""
    m = load_m(mesh)
    return trimesh_from_vf(scale_to_unit_cube(v_from_m(m)), f_from_m(m))


def f_from_m(m):
    """Extracts faces from a mesh object."""
    return np.array(trimesh.load(m, process=False).faces)


class dotdict(dict):
    """A dictionary that allows dot notation access to its attributes."""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def trig_cross(triangles):
    """Computes the cross product of triangle edges."""
    vectors = torch.diff(triangles, dim=1)
    crosses = torch.cross(vectors[:, 0], vectors[:, 1])
    return crosses


def scatter_mean_to_vox(pc_glob, vox_size, cell_coord=False):
    """Scatters points to voxels and computes their mean."""
    pc_glob_fl = pc_glob.floor().long()
    clus_idx_ = (pc_glob_fl * torch.Tensor([vox_size ** 2, vox_size, 1.]).to(pc_glob.device).long()).sum(dim=1)
    scatter_z = torch.zeros((vox_size, vox_size, vox_size, 3)).to(pc_glob.device).reshape(-1, 3)
    scatter = scatter_mean(pc_glob, clus_idx_, dim=0, out=scatter_z)
    scatter = scatter.reshape(vox_size, vox_size, vox_size, 3)
    scatter_m1 = torch.ones_like(scatter) * -10.
    mean_c = scatter[pc_glob_fl[:, 0], pc_glob_fl[:, 1], pc_glob_fl[:, 2]]
    if cell_coord:
        scatter_m1[pc_glob_fl[:, 0], pc_glob_fl[:, 1], pc_glob_fl[:, 2]] = mean_c - mean_c.floor()
    else:
        scatter_m1[pc_glob_fl[:, 0], pc_glob_fl[:, 1], pc_glob_fl[:, 2]] = mean_c
    return scatter_m1


def angle_bw_th(a, b):
    """Computes the cosine of the angle between two sets of vectors."""
    inner_product = (a * b).sum(dim=1)
    a_norm = a.pow(2).sum(dim=1).pow(0.5) + 1e-10
    b_norm = b.pow(2).sum(dim=1).pow(0.5) + 1e-10
    cos = inner_product / (2 * a_norm * b_norm)
    return cos


def put_to_device(batch, ret_keys=None):
    """Moves a batch of data to the GPU."""
    if ret_keys is None:
        return to_device_list(batch)
    else:
        return to_device_dict(batch, ret_keys)


def get_face_normals(verts, faces):
    """Computes the normals of faces in a mesh."""
    coords = verts[faces]
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]

    raw_normal = torch.cross(vec_A, vec_B, dim=-1)

    return normalize(raw_normal)


def sample_along_edge(verts, edges_unique, samp_per_e=5):
    """Samples points along the edges of a mesh."""
    lines = verts[edges_unique]
    num_fracs = lines.shape[0] * samp_per_e
    line_frac = np.random.uniform(0.1, .9, size=num_fracs).reshape(-1, samp_per_e, 1)
    complim_frac = 1. - line_frac
    pts = line_frac * lines[:, 0:1, :] + complim_frac * lines[:, 1:2, :]
    return pts.reshape(-1, 3)


def sample_along_edge_th(verts, edges_unique, samp_per_e=5):
    """Samples points along the edges of a mesh using PyTorch."""
    lines = verts[edges_unique]
    num_fracs = lines.shape[0] * samp_per_e
    line_frac = rand_unif_th((num_fracs), 0.1, .9).reshape(-1, samp_per_e, 1)
    complim_frac = 1. - line_frac
    pts = line_frac * lines[:, 0:1, :] + complim_frac * lines[:, 1:2, :]
    return pts.reshape(-1, 3)


def get_closest_f_dist(gt_samp, gen_trig):
    """Computes the distance from points to the closest face of a mesh."""
    from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
    point_face_distance = _PointFaceDistance.apply
    pts_idx = torch.tensor([0], dtype=torch.int64, device='cuda')
    trig_idx = torch.tensor([0], dtype=torch.int64, device='cuda')
    max_points = gt_samp.shape[0]
    min_area = 0.
    nn_dist = point_face_distance(gt_samp, pts_idx, gen_trig, trig_idx, max_points, min_area)
    return nn_dist


def get_close_grid_idx(grid_coord_v, pcloud):
    """Gets the indices of grid cells close to a point cloud."""
    assert pcloud.min() >= 0, "pcloud min should be >=0"
    cloud_int = pcloud.floor().long()
    pad_coord = torch.Tensor([[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 1, 1]]).to(
        cloud_int.device)
    cloud_int_pad_1 = cloud_int.unsqueeze(1) + pad_coord.unsqueeze(0).clamp(0, grid_coord_v.shape[0] - 1)
    cloud_int_pad_2 = cloud_int.unsqueeze(1) - pad_coord.unsqueeze(0).clamp(0, grid_coord_v.shape[0] - 1)
    cloud_int_pad = torch.cat([cloud_int_pad_1.reshape(-1, 3), cloud_int_pad_2.reshape(-1, 3), cloud_int],
                              dim=0).unique(dim=0).long()
    return cloud_int_pad.clamp(0, grid_coord_v.shape[0] - 1)


def get_usdf_py3d(query_p, mesh_v, mesh_f):
    """Computes the unsigned distance from points to the closest surface of a mesh."""
    gen_trig = mesh_v[mesh_f]
    point_to_face = get_closest_f_dist(query_p, gen_trig)
    point_to_face = (point_to_face + 1e-8).sqrt()
    point_to_edge = get_pt_e_dist(query_p, mesh_v, mesh_f).sqrt()
    py3d_sdf = torch.where(point_to_face < point_to_edge, point_to_face, point_to_edge)
    return py3d_sdf


def get_pt_e_dist(query_p, mesh_v, mesh_f):
    """Computes the distance from points to the closest edge of a mesh."""
    from pytorch3d.structures import Meshes
    from pytorch3d.loss.point_mesh_distance import _PointEdgeDistance
    point_edge_distance = _PointEdgeDistance.apply

    py3M = Meshes(verts=[mesh_v], faces=[mesh_f])
    verts_packed = py3M.verts_packed()
    edges_packed = py3M.edges_packed()
    segms = verts_packed[edges_packed]
    segms_first_idx = py3M.mesh_to_edges_packed_first_idx()

    pts_idx = torch.tensor([0], dtype=torch.int64, device='cuda')
    max_points = query_p.shape[0]
    nn_dist = point_edge_distance(query_p, pts_idx, segms, segms_first_idx, max_points)
    return nn_dist + 1e-8
