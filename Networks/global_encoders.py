import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as WN 
from torch_scatter import scatter_mean
from torch_cluster import grid_cluster
from pdb import set_trace as strc
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
    
class GlobPointNet(nn.Module):
    def __init__(self, latent_size, grid_size=32, v_start=-.5, v_end=0.5):
        """Encoder"""

        super(GlobPointNet, self).__init__()
        self.v_start = v_start
        self.v_end = v_end
        latent_dim = latent_size
        self.latent_size = latent_size
        
        self.conv1 = torch.nn.Linear(3, self.latent_size)
        self.res1 = pc_resnet_block(self.latent_size)
        self.conv2 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res2 = pc_resnet_block(self.latent_size)
        self.conv3 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res3 = pc_resnet_block(self.latent_size)
        self.conv4 = torch.nn.Linear(self.latent_size, self.latent_size)


    def forward(self, enc_xyz, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int,voxel_KNN_idx,voxel_KNN_xyz, query_pt):
        x = enc_xyz
        pc = x.clone().unsqueeze(0)
        x = self.res1(F.relu(self.conv1(x)))
        x = self.res2(F.relu(self.conv2(x)))
        x = self.res3(F.relu(self.conv3(x)))
        x = self.conv4(x)
        glob_feat, _ = torch.max(x, 0)
        glob_feat_exp = glob_feat.squeeze().unsqueeze(0).expand(query_pt.shape[0], -1)
        return glob_feat_exp.unsqueeze(0)
        

class GlobPointNetFeatBN(nn.Module):
    def __init__(self, latent_size):
        """Encoder"""

        super(GlobPointNetFeatBN, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, latent_size, 1)
        self.lin1 = nn.Linear(latent_size, latent_size)
        self.lin2 = nn.Linear(latent_size, latent_size)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(latent_size)
        self.bn4 = torch.nn.BatchNorm1d(latent_size)
        self.bn5 = torch.nn.BatchNorm1d(latent_size)

        self.latent_size = latent_size

    def forward(self, x):
        batchsize = x.size()[0]
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, self.latent_size)
        x = self.bn4(F.relu(self.lin1(x).unsqueeze(-1)))
        x = self.bn5(F.relu(self.lin2(x.squeeze(2)).unsqueeze(-1)))
        return x.squeeze(2)
    
class GlobLocPointNet(nn.Module):
    def __init__(self, latent_size, grid_size=32, v_start=-.5, v_end=0.5):
        """Encoder"""

        super(GlobLocPointNet, self).__init__()
        self.v_start = v_start
        self.v_end = v_end
        latent_dim = latent_size
        self.ef_dim = latent_size
        self.latent_size = latent_size//4
        
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
    
    def get_subdiv_feat(self, pc, pw_feat, div):
        assert pc.shape[0] == 1 # Warning batch_size=1
        pc = pc.squeeze()
        pw_feat = pw_feat.squeeze()
        # Because it's now F, N
        pw_feat = pw_feat.transpose(1,0)
        feat_dim = pw_feat.shape[-1]
        # WARNING: Fortran indexing
        clus_idx = grid_cluster(self.get_vox_idx(pc,div).float(), size=torch.tensor([1.,1.,1.]).to(pw_feat.device))
        four_feat_f = torch.zeros((div,div,div,feat_dim)).to(pw_feat.device).reshape(-1,feat_dim)
        scatter_f = scatter_mean(pw_feat, clus_idx, dim=0, out=four_feat_f).reshape(div,div,div,feat_dim)
        # back to 'C' indexing.
        return scatter_f.swapaxes(0,2)

    def forward(self, pc_KNN_idx,pc_KNN_xyz, voxel_xyz_int, voxel_KNN_idx,voxel_KNN_xyz, query_pt,
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
        strc()
        pw_feat = out.clone().unsqueeze(0).transpose(1,2)
        # glob_feat, _ = torch.max(out, 0)
        
        out = self.pc_conv_6(out, voxel_KNN_idx, voxel_KNN_xyz)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.pc_res_7(out)
        pw_feat = out.clone().unsqueeze(0).transpose(1,2)
        glob_feat, _ = torch.max(out, 0)
        four_feat = self.get_subdiv_feat(pc, pw_feat, 4)
        eight_feat = self.get_subdiv_feat(pc, pw_feat, 8)
        sixteen_feat = self.get_subdiv_feat(pc, pw_feat, 16)
        # TODO: What for a batch?
        query_pt = query_pt.squeeze()
        
        #TODO: This can be speed up from dataloader
        q_idx_4 = self.get_vox_idx(query_pt, 4)
        q_idx_8 = self.get_vox_idx(query_pt, 8)
        q_idx_16 = self.get_vox_idx(query_pt, 16)
        
        q_feat_4 = four_feat[q_idx_4[:,0], q_idx_4[:,1], q_idx_4[:,2], :]
        q_feat_8 = eight_feat[q_idx_8[:,0], q_idx_8[:,1], q_idx_8[:,2], :]
        q_feat_16 = sixteen_feat[q_idx_16[:,0], q_idx_16[:,1], q_idx_16[:,2], :]
        glob_feat_exp = glob_feat.squeeze().unsqueeze(0).expand(q_feat_4.shape[0], -1)
        feat_cat = torch.cat((glob_feat_exp, q_feat_4, q_feat_8, q_feat_16), dim=-1)
        # Make it batch and return
        return feat_cat.unsqueeze(0)
