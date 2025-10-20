from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class pc_resnet_block(nn.Module):
    def __init__(self, ef_dim, use_wn=False):
        super(pc_resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.linear_1 = nn.Linear(self.ef_dim, self.ef_dim)
        self.linear_2 = nn.Linear(self.ef_dim, self.ef_dim)

    def forward(self, input):
        out = self.linear_1(input)
        out = F.leaky_relu(out)
        out = self.linear_2(out)
        out = out + input
        out = F.leaky_relu(out)
        return out


class SimpleLinear(nn.Module):
    def __init__(
        self,
        input_dim=256,
        feature_dim=256,
        stats=False,
        weight_std=0.001,
        bias=0.0,
        sig_scale=10.0,
    ):
        super(SimpleLinear, self).__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.stats = stats
        self.weight_std = weight_std
        self.bias = bias

        self.vtx = nn.Sequential(
            OrderedDict(
                [
                    ("mlp1", nn.Linear(self.input_dim, self.feature_dim, bias=False)),
                    (
                        "bn1",
                        nn.BatchNorm1d(
                            self.feature_dim, track_running_stats=self.stats
                        ),
                    ),
                    ("relu1", nn.ReLU(inplace=True)),
                    ("mlp2", nn.Linear(self.feature_dim, self.feature_dim, bias=False)),
                    (
                        "bn2",
                        nn.BatchNorm1d(
                            self.feature_dim, track_running_stats=self.stats
                        ),
                    ),
                    ("relu2", nn.ReLU(inplace=True)),
                    ("mlp3", nn.Linear(self.feature_dim, self.feature_dim, bias=False)),
                    (
                        "bn3",
                        nn.BatchNorm1d(
                            self.feature_dim, track_running_stats=self.stats
                        ),
                    ),
                    ("relu3", nn.ReLU(inplace=True)),
                    ("vtx", nn.Linear(self.feature_dim, 3, bias=True)),
                ]
            )
        )

        with torch.no_grad():
            self.vtx[-1].weight.data.normal_(std=self.weight_std)
            nn.init.constant_(self.vtx[-1].bias.data, self.bias)

    def forward(self, query_pt):
        ### Sigmoid ###
        return F.relu(
            self.vtx(query_pt.reshape(-1, self.input_dim)).reshape(
                query_pt.shape[0], query_pt.shape[1], 3
            )
        )

        ### Tanh ###
        # return torch.add(
        #     torch.tanh(self.vtx(query_pt.reshape(-1, self.input_dim))).reshape(query_pt.shape[0], query_pt.shape[1], 3),
        #     0.5
        # )


class DeeperLinear(nn.Module):
    def __init__(
        self,
        latent_size=256,
        input_dim=128,
        pooling_type="max",
        use_sig=False,
        sig_scale=10.0,
    ):
        self.latent_size = latent_size
        self.use_sig = use_sig
        super(DeeperLinear, self).__init__()
        self.conv1 = torch.nn.Linear(input_dim, self.latent_size)
        self.res1 = pc_resnet_block(self.latent_size)
        self.conv2 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res2 = pc_resnet_block(self.latent_size)
        self.conv3 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res3 = pc_resnet_block(self.latent_size)
        self.conv4 = torch.nn.Linear(self.latent_size, 3)
        self.sg = nn.Sigmoid()

    def forward(self, query_pt):
        x = self.res1(F.leaky_relu(self.conv1(query_pt)))
        x = self.res2(F.leaky_relu(self.conv2(x)))
        x = self.res3(F.leaky_relu(self.conv3(x)))
        x = self.conv4(x)
        if self.use_sig:
            x = self.sg(x)
        return x


class EvenDeeperLinear(nn.Module):
    def __init__(
        self,
        latent_size=512,
        input_dim=128,
        pooling_type="max",
        use_sig=False,
        sig_scale=10.0,
    ):
        self.latent_size = latent_size
        self.use_sig = use_sig
        super(EvenDeeperLinear, self).__init__()
        self.conv1 = torch.nn.Linear(input_dim, self.latent_size)
        self.res1 = pc_resnet_block(self.latent_size)
        self.conv2 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res2 = pc_resnet_block(self.latent_size)
        self.conv3 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res3 = pc_resnet_block(self.latent_size)
        self.conv4 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res4 = pc_resnet_block(self.latent_size)
        self.conv5 = torch.nn.Linear(self.latent_size, self.latent_size)
        self.res5 = pc_resnet_block(self.latent_size)
        self.conv6 = torch.nn.Linear(self.latent_size, 3)
        self.sg = nn.Sigmoid()

    def forward(self, query_pt):
        x = self.res1(F.relu(self.conv1(query_pt)))
        x = self.res2(F.relu(self.conv2(x)))
        x = self.res3(F.relu(self.conv3(x)))
        x = self.res4(F.relu(self.conv4(x)))
        x = self.res5(F.relu(self.conv5(x)))
        x = self.conv6(x)
        if self.use_sig:
            x = self.sg(x)
        return x


class CNN_3d_rec7(nn.Module):

    def __init__(self, input_dim=None, use_sig=False, ef_dim=64, sig_scale=10.0):
        super(CNN_3d_rec7, self).__init__()

        self.ef_dim = 64
        self.input_dim = input_dim
        self.use_sig = use_sig
        self.sig_scale = sig_scale

        self.conv3d_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv3d_1",
                        nn.Conv3d(
                            1, ef_dim, kernel_size=3, stride=1, padding=0, bias=True
                        ),
                    ),
                    ("relu_1", nn.LeakyReLU(inplace=True, negative_slope=0.01)),
                    (
                        "conv3d_2",
                        nn.Conv3d(
                            ef_dim,
                            ef_dim,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("relu_2", nn.LeakyReLU(inplace=True, negative_slope=0.01)),
                    (
                        "conv3d_3",
                        nn.Conv3d(
                            ef_dim,
                            ef_dim,
                            kernel_size=3,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("relu_3", nn.LeakyReLU(inplace=True, negative_slope=0.01)),
                    (
                        "conv3d_4",
                        nn.Conv3d(
                            ef_dim,
                            ef_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("relu_4", nn.LeakyReLU(inplace=True, negative_slope=0.01)),
                    (
                        "conv3d_5",
                        nn.Conv3d(
                            ef_dim,
                            ef_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=True,
                        ),
                    ),
                    ("relu_5", nn.LeakyReLU(inplace=True, negative_slope=0.01)),
                    (
                        "conv3d_6",
                        nn.Conv3d(
                            ef_dim, 3, kernel_size=1, stride=1, padding=0, bias=True
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        out = x
        out = self.conv3d_stack(out)
        if self.use_sig:
            out = torch.sigmoid(out * self.sig_scale)
        return out.permute(0, 2, 3, 4, 1)


class CNN3D_Lite(nn.Module):

    def __init__(self, input_dim=None, use_sig=False, ef_dim=64, sig_scale=10.0):
        super(CNN3D_Lite, self).__init__()

        self.ef_dim = 64
        self.input_dim = input_dim
        self.use_sig = use_sig
        self.sig_scale = sig_scale

        self.conv3d_stack = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv3d_1",
                        nn.Conv3d(
                            1, ef_dim, kernel_size=4, stride=1, padding=1, bias=True
                        ),
                    ),
                    # ('conv3d_1_bn', nn.BatchNorm3d(self.ef_dim)),
                    ("relu_1", nn.LeakyReLU(inplace=True)),
                    (
                        "conv3d_6",
                        nn.Conv3d(
                            ef_dim, 3, kernel_size=1, stride=1, padding=0, bias=True
                        ),
                    ),
                ]
            )
        )

    def forward(self, x):
        out = x
        out = self.conv3d_stack(out)
        if self.use_sig:
            out = torch.sigmoid(out * self.sig_scale)
        return out.permute(0, 2, 3, 4, 1)


# class CNN_3d_rec7Def(nn.Module):

#     def __init__(self, input_dim=None, use_sig=False, sig_scale=10.):
#         super(CNN_3d_rec7Def, self).__init__()
#         self.ef_dim = 64

#         self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
#         self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
#         self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)
#         self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
#         self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
#         self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
#         self.use_sig = use_sig
#         self.sig_scale = sig_scale

#     def forward(self, x):
#         out = x

#         out = self.conv_0(out)
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.conv_1(out)
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.conv_2(out)
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.conv_3(out)
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out = self.conv_4(out)
#         out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

#         out_float = self.conv_out_float(out)
#         if self.use_sig:
#             out_float = F.sigmoid(out_float*self.sig_scale)
#         return out_float.permute(0, 2, 3, 4, 1)


class CNN_3d_rec7Def(nn.Module):

    def __init__(self, input_dim=None, use_sig=False, sig_scale=10.0):
        super(CNN_3d_rec7Def, self).__init__()
        self.ef_dim = 64

        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.conv_1 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True
        )
        self.conv_2 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True
        )
        self.conv_3 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True
        )
        self.conv_4 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True
        )
        self.conv_out_float = nn.Conv3d(
            self.ef_dim, 3, 1, stride=1, padding=0, bias=True
        )
        self.use_sig = use_sig
        self.sig_scale = sig_scale

    def forward(self, x):
        out = x

        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out_float = self.conv_out_float(out)
        if self.use_sig:
            out_float = F.sigmoid(out_float * self.sig_scale)
        return out_float.permute(0, 2, 3, 4, 1)


class CNN_3d_rec7DefBN(nn.Module):

    def __init__(self, input_dim=None, use_sig=False, sig_scale=10.0):
        super(CNN_3d_rec7DefBN, self).__init__()
        self.ef_dim = 64

        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)
        self.bn_0 = nn.BatchNorm3d(self.ef_dim)
        self.conv_1 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True
        )
        self.bn_1 = nn.BatchNorm3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True
        )
        self.bn_2 = nn.BatchNorm3d(self.ef_dim)
        self.conv_3 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True
        )
        self.bn_3 = nn.BatchNorm3d(self.ef_dim)
        self.conv_4 = nn.Conv3d(
            self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True
        )
        self.bn_4 = nn.BatchNorm3d(self.ef_dim)
        self.conv_out_float = nn.Conv3d(
            self.ef_dim, 3, 1, stride=1, padding=0, bias=True
        )
        self.use_sig = use_sig
        self.sig_scale = sig_scale

    def forward(self, x):
        out = x

        out = self.bn_0(self.conv_0(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_1(self.conv_1(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_2(self.conv_2(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_3(self.conv_3(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.bn_4(self.conv_4(out))
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out_float = self.conv_out_float(out)
        if self.use_sig:
            out_float = F.sigmoid(out_float * self.sig_scale)
        return out_float.permute(0, 2, 3, 4, 1)
