
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import fourier_encode # type: ignore
from torch.nn.utils import weight_norm as WN 
from pdb import set_trace as strc



class pc_resnet_block(nn.Module):
    def __init__(self, ef_dim, use_wn=False):
        super(pc_resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        out = self.linear_1(input)
        out = F.relu(out)
        out = self.linear_2(out)
        out = out+input
        out = F.relu(out)
        return out

class SimpleSDFNet(nn.Module):
    def __init__(self, inp_dim, n_feat=256, do_pe=True,
                 use_wn=False):
        """Encoder"""

        super(SimpleSDFNet, self).__init__()
        self.latent_size = n_feat
        
        self.inp_dim = inp_dim
        self.do_pe = do_pe
        in_dim = 252+inp_dim if do_pe else inp_dim+3
        # self.lin0 = WN(nn.Linear(3, n_feat))
        self.lin1 = nn.Linear(in_dim, n_feat)
        self.res1 = pc_resnet_block(self.latent_size)
        self.lin2 = nn.Linear(n_feat, n_feat)
        self.res2 = pc_resnet_block(self.latent_size)
        self.lin3 = nn.Linear(n_feat, n_feat)
        self.res3 = pc_resnet_block(self.latent_size)
        self.lin4 = nn.Linear(n_feat, 1)
        
        self.dp = nn.Dropout(p=0.2)


    def forward(self, query_pt, lat_vec, nn_idx=None, has_grad=False):
        return_dict = {}
        query_in = query_pt.clone()
        if self.do_pe:
            model_in = fourier_encode(query_in, embedding_size=128)
        else:
            model_in = query_in
        
        lat_vec_exp = lat_vec.expand(-1, model_in.shape[1], -1).to(query_pt.device)
        x_cat = torch.cat((lat_vec_exp, model_in), dim=-1)
        # query_f = self.lin0(query_in)
        # query_f = F.relu(self.lin2(query_f))
        out = self.dp(F.relu(self.lin1(x_cat)))
        out = self.res1(out)
        out = self.dp(F.relu(self.lin2(out)))
        out = self.res2(out)
        out = self.dp(F.relu(self.lin3(out)))
        out = self.res3(out)
        out = self.lin4(out)
        return_dict['model_out'] = out
        # TODO: account batch
        return_dict['model_in'] = query_in
        return return_dict

    
class EvenDeeperSDF(nn.Module):
    def __init__(self, inp_dim, n_feat=256, do_pe=True,
                 use_wn=False):
        
        self.inp_dim = inp_dim
        self.do_pe = do_pe
        in_dim = 252+inp_dim if do_pe else inp_dim+3
        self.latent_size = n_feat
        super(EvenDeeperSDF, self).__init__()
        self.conv1 = torch.nn.Linear(in_dim, self.latent_size)
        self.res1 = pc_resnet_block(self.latent_size)
        self.conv2 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res2 = pc_resnet_block(self.latent_size)
        self.conv3 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res3 = pc_resnet_block(self.latent_size)
        self.conv4 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res4 = pc_resnet_block(self.latent_size)
        self.conv5 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res5 = pc_resnet_block(self.latent_size)
        self.conv6 = torch.nn.Linear(self.latent_size, 1)

    def forward(self, query_pt, lat_vec, nn_idx=None, has_grad=False):
        if self.do_pe:
            model_in = fourier_encode(query_pt, embedding_size=128)
        else:
            model_in = query_pt
        
        lat_vec_exp = lat_vec.expand(-1, model_in.shape[1], -1).to(query_pt.device)
        x_cat = torch.cat((lat_vec_exp, model_in), dim=-1)
        x = self.res1(F.relu(self.conv1(x_cat)))
        x = self.res2(F.relu(self.conv2(x)))
        x = self.res3(F.relu(self.conv3(x)))
        x = self.res4(F.relu(self.conv4(x)))
        x = self.res5(F.relu(self.conv5(x)))
        ret_feat = x.clone()
        out = self.conv6(x)
        return_dict = {}
        return_dict['model_out'] = out
        # TODO: account batch
        return_dict['ret_feat'] = ret_feat
        return return_dict
    

class OldSimpleSDFNet(nn.Module):
    def __init__(self, inp_dim, n_feat=256, do_pe=True,
                 use_wn=False):
        """Encoder"""

        super(OldSimpleSDFNet, self).__init__()
        
        self.inp_dim = inp_dim
        self.do_pe = do_pe
        in_dim = 252+inp_dim if do_pe else inp_dim+3
        # self.lin0 = WN(nn.Linear(3, n_feat))
        self.lin1 = WN(nn.Linear(in_dim, n_feat)) if use_wn else nn.Linear(in_dim, n_feat)
        self.lin2 = WN(nn.Linear(n_feat, n_feat)) if use_wn else nn.Linear(n_feat, n_feat)
        self.lin3 = WN(nn.Linear(n_feat, n_feat)) if use_wn else nn.Linear(n_feat, n_feat)
        self.lin4 = WN(nn.Linear(n_feat, n_feat)) if use_wn else nn.Linear(n_feat, n_feat)
        self.lin5 = WN(nn.Linear(n_feat, 1)) if use_wn else nn.Linear(n_feat, 1)
        
        self.dp = nn.Dropout(p=0.2)

        self.latent_size = n_feat

    def forward(self, query_pt, lat_vec, nn_idx=None, has_grad=False):
        return_dict = {}
        query_in = query_pt.clone()
        if self.do_pe:
            model_in = fourier_encode(query_in, embedding_size=128)
        else:
            model_in = query_in
        
        lat_vec_exp = lat_vec.expand(-1, model_in.shape[1], -1).to(query_pt.device)
        x_cat = torch.cat((lat_vec_exp, model_in), dim=-1)
        # query_f = self.lin0(query_in)
        # query_f = F.relu(self.lin2(query_f))
        out = F.relu(self.lin1(x_cat))
        out = F.relu(self.lin2(out))
        out = F.relu(self.lin3(out))
        out = F.relu(self.lin4(out))
        
        ret_feat = out.clone()
        out = self.lin5(out)
        return_dict['model_out'] = out
        # TODO: account batch
        return_dict['ret_feat'] = ret_feat
        return_dict['model_in'] = query_in
        return return_dict