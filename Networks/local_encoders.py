from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as strc
from torch.autograd import grad
import numpy as np
from torch.nn.utils import weight_norm as WN 
from utils import fourier_encode
from torch_scatter import scatter_mean
from torch_cluster import grid_cluster

KNN_num = 8

class pc_conv_first(nn.Module):
    def __init__(self, ef_dim):
        super(pc_conv_first, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, KNN_xyz):
        output = KNN_xyz
        #[newpointnum*KNN_num,3]
        output = self.linear_1(output)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        #[newpointnum*KNN_num,ef_dim]
        output = output.view(-1,KNN_num,self.ef_dim)
        #[newpointnum,KNN_num,ef_dim]
        output = torch.max(output,1)[0]
        #[newpointnum,ef_dim]
        return output
class pc_resnet_block(nn.Module):
    def __init__(self, ef_dim):
        super(pc_resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        output = self.linear_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        output = output+input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output
class pc_conv(nn.Module):
    def __init__(self, ef_dim):
        super(pc_conv, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim+3, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input, KNN_idx, KNN_xyz):
        output = input
        #[pointnum,ef_dim]
        output = output[KNN_idx]
        #[newpointnum*KNN_num,ef_dim]
        output = torch.cat([output,KNN_xyz],1)
        #[newpointnum*KNN_num,ef_dim+3]
        output = self.linear_1(output)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.linear_2(output)
        #[newpointnum*KNN_num,ef_dim]
        output = output.view(-1,KNN_num,self.ef_dim)
        #[newpointnum,KNN_num,ef_dim]
        output = torch.max(output,1)[0]
        #[newpointnum,ef_dim]
        return output


class LocalPointNet(nn.Module):

    def __init__(self,latent_size, grid_size=32, v_start=-.5, v_end=0.5, use_glob=False):
        super(LocalPointNet, self).__init__()
        self.v_start = v_start
        self.v_end = v_end
        self.ef_dim = latent_size
        self.grid_size = grid_size
        self.use_glob = use_glob
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)
        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)
        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)
        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)
        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)
        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)
        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)
        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        # self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        # self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        # self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 7, stride=1, padding=3, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 5, stride=1, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        
        self.maxpool = nn.MaxPool3d(2)
        
        self.conv_4 = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        self.conv_4a = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        
        self.conv_5 = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        self.conv_5a = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        
        self.conv_6 = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        self.conv_6a = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        
        self.fc_0 = nn.Conv1d(self.ef_dim, self.ef_dim, 1)
        self.fc_1 = nn.Conv1d(self.ef_dim, self.ef_dim, 1)
        

    def shorten_w_zeros(self, voxel_xyz_int, out, glob_feat):
        tmp_grid = torch.zeros(self.grid_size+1,self.grid_size+1,self.grid_size+1,self.ef_dim, device=out.device)
        if self.use_glob:
            tmp_grid[:,:,:,:] = glob_feat
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        return tmp_grid.permute(3,0,1,2).unsqueeze(0)

    def forward(self, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz, query_pt,
                per_idx=None):
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        glob_feat = torch.max(out, dim=0)[0]
        tmp_grid = self.shorten_w_zeros(voxel_xyz_int, out, glob_feat)
        out = tmp_grid
        out = self.conv_1(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, inplace=True)
        
        if per_idx is not None:
            perm_idx = (0,1,)+tuple((per_idx+2).cpu().numpy().tolist())
            out = out.permute(perm_idx)
        out = self.conv_3(out)
        self.grid_feat = out.clone()
        full_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))

        out = self.maxpool(out)
        out = F.leaky_relu(self.conv_4(out))
        out = F.leaky_relu(self.conv_4a(out))
        half_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))
        
        out = self.maxpool(out)
        out = F.leaky_relu(self.conv_5(out))
        out = F.leaky_relu(self.conv_5a(out))
        quat_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))
        
        out = self.maxpool(out)
        out = F.leaky_relu(self.conv_6(out))
        out = F.leaky_relu(self.conv_6a(out))
        oct_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))
        query_feat = torch.cat((full_feat, half_feat, quat_feat, oct_feat), dim=1).squeeze().unsqueeze(0)
        
        out = self.fc_0(query_feat)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        
        out = self.fc_1(out)
        return out.transpose(2,1)
        
class PurelyLocalPointNet(nn.Module):

    def __init__(self,latent_size, grid_size=32, v_start=-.5, v_end=0.5):
        super(PurelyLocalPointNet, self).__init__()
        self.v_start = v_start
        self.v_end = v_end
        self.ef_dim = latent_size
        self.grid_size = grid_size
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)

        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)

        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)

        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)

        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)

        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)

        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)

        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)


    def shorten_wo_b(self, voxel_xyz_int, out):
        voxel_xyz_int_max = torch.max(voxel_xyz_int,0)[0]
        voxel_xyz_int_min = torch.min(voxel_xyz_int,0)[0]
        voxel_xyz_int_size = voxel_xyz_int_max-voxel_xyz_int_min+1
        voxel_xyz_int = voxel_xyz_int-voxel_xyz_int_min.view(1,-1)
        tmp_grid = torch.zeros(voxel_xyz_int_size[0],voxel_xyz_int_size[1],voxel_xyz_int_size[2],self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        tmp_grid = tmp_grid.permute(3,0,1,2).unsqueeze(0)
        tmp_grid = F.interpolate(tmp_grid, size=(self.grid_size,self.grid_size,self.grid_size),mode='trilinear')
        return tmp_grid

    def shorten_w_zeros(self, voxel_xyz_int, out):
        tmp_grid = torch.zeros(self.grid_size+1,self.grid_size+1,self.grid_size+1,self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        return tmp_grid.permute(3,0,1,2).unsqueeze(0)

    def to_int_coord(self, xyz, vox_size):
        min_val = -1 * self.v_start
        max_val = self.v_end - self.v_start
        size = vox_size - 1
        int_coord = (((xyz+min_val)/max_val)*size).long()
        return int_coord
    
    def get_vox_idx(self, xyz, cell_size):
        # 4 cells
        grid_coord = self.to_int_coord(xyz, vox_size=cell_size)
        grid_fl_cl = torch.clamp(grid_coord, 0, cell_size-1)
        return grid_fl_cl
    
    def scatter_quert_feat(self, query_pt, vox_feat):
        vox_feat = vox_feat.permute(1,2,3,0)
        vox_size = vox_feat.shape[0]
        feat_dim = vox_feat.shape[-1]
        query_vox_id = self.get_vox_idx(query_pt, vox_size).long()
        query_feat = vox_feat[query_vox_id[:,0], query_vox_id[:,1], query_vox_id[:,2], :]
        return query_feat

    def forward(self, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz, query_pt):
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        tmp_grid = self.shorten_w_zeros(voxel_xyz_int, out)
        out = tmp_grid

        out = self.conv_1(out)
        out = F.relu(out, inplace=True)

        out = self.conv_2(out)
        out = F.relu(out, inplace=True)

        out = self.conv_3(out)
        self.grid_feat = out.clone()
        query_feat = self.scatter_quert_feat(query_pt.squeeze(), out.squeeze())
        # query_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1)).squeeze().transpose(1,0)
        return query_feat.unsqueeze(0)

    
class LocalPointNetFiner(nn.Module):

    def __init__(self,latent_size, grid_size=32, v_start=-.5, v_end=0.5, use_glob=False):
        super(LocalPointNetFiner, self).__init__()
        self.v_start = v_start
        self.v_end = v_end
        self.ef_dim = latent_size
        self.grid_size = grid_size
        self.use_glob = use_glob
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)
        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)
        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)
        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)
        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)
        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)
        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)
        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        # self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        # self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        # self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 7, stride=1, padding=3, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 5, stride=1, padding=2, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=2, padding=1, bias=True)
        
        # self.maxpool = nn.MaxPool3d(2)
        
        # self.conv_4 = nn.Conv3d(self.ef_dim//2, self.ef_dim//2, 3, stride=1, padding=1, bias=True)
        # self.conv_4a = nn.Conv3d(self.ef_dim//2, self.ef_dim//4, 3, stride=2, padding=1, bias=True)
        # self.conv_5 = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=1, padding=1, bias=True)
        # self.conv_5a = nn.Conv3d(self.ef_dim//4, self.ef_dim//4, 3, stride=2, padding=1, bias=True)
        
        # self.fc_0 = nn.Conv1d(int(3* self.ef_dim//4), self.ef_dim, 1)
        # self.fc_1 = nn.Conv1d(self.ef_dim, self.ef_dim, 1)
        

    def shorten_w_zeros(self, voxel_xyz_int, out, glob_feat):
        tmp_grid = torch.zeros(self.grid_size+1,self.grid_size+1,self.grid_size+1,self.ef_dim, device=out.device)
        if self.use_glob:
            tmp_grid[:,:,:,:] = glob_feat
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        return tmp_grid.permute(3,0,1,2).unsqueeze(0)

    def to_int_coord(self, xyz, vox_size):
        min_val = -1 * self.v_start
        max_val = self.v_end - self.v_start
        size = vox_size - 1
        int_coord = (((xyz+min_val)/max_val)*size).long()
        return int_coord
    
    def get_vox_idx(self, xyz, cell_size):
        # 4 cells
        grid_coord = self.to_int_coord(xyz, vox_size=cell_size)
        grid_fl_cl = torch.clamp(grid_coord, 0, cell_size-1)
        return grid_fl_cl
    
    def forward(self, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz, query_pt,
                per_idx=None):
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        glob_feat = torch.max(out, dim=0)[0]
        tmp_grid = self.shorten_w_zeros(voxel_xyz_int, out, glob_feat)
        out = tmp_grid
        out = self.conv_1(out)
        out = F.leaky_relu(out, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, inplace=True)
        
        if per_idx is not None:
            perm_idx = (0,1,)+tuple((per_idx+2).cpu().numpy().tolist())
            out = out.permute(perm_idx)
        out = self.conv_3(out)
        self.grid_feat = out.clone()
        full_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))

        # out = F.leaky_relu(self.conv_4(out))
        # out = self.conv_4a(out)
        # half_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))
        
        # out = F.leaky_relu(self.conv_5(out))
        # out = self.conv_5a(out)
        # quat_feat = F.grid_sample(out, query_pt.unsqueeze(1).unsqueeze(1))

        # query_feat = torch.cat((full_feat, half_feat, quat_feat), dim=1).squeeze().unsqueeze(0)
        query_feat = full_feat.squeeze().unsqueeze(0)
        return query_feat.transpose(2,1)

class NDCPointNet(nn.Module):

    def __init__(self,latent_size, grid_size=65, v_start=-.5, v_end=0.5):
        super(NDCPointNet, self).__init__()
        self.v_start = v_start
        self.v_end = v_end
        self.ef_dim = latent_size
        self.grid_size = grid_size
        
        self.pc_conv_0 = pc_conv_first(self.ef_dim)

        self.pc_res_1 = pc_resnet_block(self.ef_dim)
        self.pc_conv_1 = pc_conv(self.ef_dim)

        self.pc_res_2 = pc_resnet_block(self.ef_dim)
        self.pc_conv_2 = pc_conv(self.ef_dim)

        self.pc_res_3 = pc_resnet_block(self.ef_dim)
        self.pc_conv_3 = pc_conv(self.ef_dim)

        self.pc_res_4 = pc_resnet_block(self.ef_dim)
        self.pc_conv_4 = pc_conv(self.ef_dim)

        self.pc_res_5 = pc_resnet_block(self.ef_dim)
        self.pc_conv_5 = pc_conv(self.ef_dim)

        self.pc_res_6 = pc_resnet_block(self.ef_dim)
        self.pc_conv_6 = pc_conv(self.ef_dim)

        self.pc_res_7 = pc_resnet_block(self.ef_dim)

        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)
        
        self.conv_4 = nn.Linear(self.ef_dim, self.ef_dim)
        self.conv_5 = nn.Linear(self.ef_dim, self.ef_dim)
        
        self.pc_conv_out_float = nn.Linear(self.ef_dim, 1)
        
        # self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 7, stride=1, padding=3, bias=True)
        # self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 5, stride=1, padding=2, bias=True)
        # self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=1, bias=True)

    def shorten_wo_b(self, voxel_xyz_int, out):
        voxel_xyz_int_max = torch.max(voxel_xyz_int,0)[0]
        voxel_xyz_int_min = torch.min(voxel_xyz_int,0)[0]
        voxel_xyz_int_size = voxel_xyz_int_max-voxel_xyz_int_min+1
        voxel_xyz_int = voxel_xyz_int-voxel_xyz_int_min.view(1,-1)
        tmp_grid = torch.zeros(voxel_xyz_int_size[0],voxel_xyz_int_size[1],voxel_xyz_int_size[2],self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        tmp_grid = tmp_grid.permute(3,0,1,2).unsqueeze(0)
        tmp_grid = F.interpolate(tmp_grid, size=(self.grid_size,self.grid_size,self.grid_size),mode='trilinear')
        return tmp_grid

    def shorten_w_zeros(self, voxel_xyz_int, out):
        tmp_grid = torch.zeros(self.grid_size+1,self.grid_size+1,self.grid_size+1,self.ef_dim, device=out.device)
        tmp_grid[voxel_xyz_int[:,0],voxel_xyz_int[:,1],voxel_xyz_int[:,2]] = out
        return tmp_grid.permute(3,0,1,2).unsqueeze(0)

    def to_int_coord(self, xyz, vox_size):
        min_val = -1 * self.v_start
        max_val = self.v_end - self.v_start
        size = vox_size - 1
        int_coord = (((xyz+min_val)/max_val)*size).long()
        return int_coord
    
    def get_vox_idx(self, xyz, cell_size):
        # 4 cells
        grid_coord = self.to_int_coord(xyz, vox_size=cell_size)
        grid_fl_cl = torch.clamp(grid_coord, 0, cell_size-1)
        return grid_fl_cl
    
    def scatter_quert_feat(self, query_pt, vox_feat):
        vox_feat = vox_feat.permute(1,2,3,0)
        vox_size = vox_feat.shape[0]
        feat_dim = vox_feat.shape[-1]
        query_vox_id = self.get_vox_idx(query_pt, vox_size).long()
        query_feat = vox_feat[query_vox_id[:,0], query_vox_id[:,1], query_vox_id[:,2], :]
        return query_feat

    def forward(self, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz, query_xyz_int):
        out = pc_KNN_xyz
        
        out = self.pc_conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_1(out)
        out = self.pc_conv_1(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_2(out)
        out = self.pc_conv_2(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_3(out)
        out = self.pc_conv_3(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_4(out)
        out = self.pc_conv_4(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_5(out)
        out = self.pc_conv_5(out, pc_KNN_idx, pc_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_6(out)
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)

        tmp_grid = self.shorten_w_zeros(voxel_xyz_int, out)
        out = tmp_grid

        out = self.conv_1(out)
        out = F.relu(out, inplace=True)

        out = self.conv_2(out)
        out = F.relu(out, inplace=True)

        out = self.conv_3(out).squeeze(0).permute(1,2,3,0)
        
        out_ext = out[query_xyz_int[:,0],query_xyz_int[:,1],query_xyz_int[:,2]]
        
        out_ext = F.leaky_relu(out_ext, negative_slope=0.01, inplace=True)

        out_ext = self.conv_4(out_ext)
        out_ext = F.leaky_relu(out_ext, negative_slope=0.01, inplace=True)
        
        out_ext = self.conv_5(out_ext)
        out_ext = F.leaky_relu(out_ext, negative_slope=0.01, inplace=True)
        out_float = self.pc_conv_out_float(out_ext)
        return out_float
