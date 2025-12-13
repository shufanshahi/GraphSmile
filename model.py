from module import HeterGConv_Edge, HeterGConvLayer, SenShift_Feat
import torch.nn as nn
import torch
from utils import batch_to_all_tva


class GraphSmile(nn.Module):

    def __init__(self, args, embedding_dims, n_classes_emo):
        super(GraphSmile, self).__init__()
        self.textf_mode = args.textf_mode
        self.no_cuda = args.no_cuda
        self.win_p = args.win[0]
        self.win_f = args.win[1]
        self.modals = args.modals
        self.shift_win = args.shift_win

        if self.textf_mode == 'concat4' or self.textf_mode == 'sum4':
            self.used_t_indices = [0, 1, 2, 3]
        elif self.textf_mode == 'concat2' or self.textf_mode == 'sum2':
            self.used_t_indices = [0, 1]
        elif self.textf_mode == 'textf0':
            self.used_t_indices = [0]
        elif self.textf_mode == 'textf1':
            self.used_t_indices = [1]
        elif self.textf_mode == 'textf2':
            self.used_t_indices = [2]
        elif self.textf_mode == 'textf3':
            self.used_t_indices = [3]
        else:
            raise ValueError(f"unsupported: {self.textf_mode}")
        self.batchnorms_t = nn.ModuleList([
            nn.BatchNorm1d(embedding_dims[0]) for _ in self.used_t_indices])

        if self.textf_mode.startswith('concat'):
            in_dims_t = len(self.used_t_indices) * embedding_dims[0]
        else:
            in_dims_t = embedding_dims[0]
        self.dim_layer_t = nn.Sequential(nn.Linear(in_dims_t, args.hidden_dim),
                                         nn.LeakyReLU(), nn.Dropout(args.drop))
        self.dim_layer_v = nn.Sequential(
            nn.Linear(embedding_dims[1], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )
        self.dim_layer_a = nn.Sequential(
            nn.Linear(embedding_dims[2], args.hidden_dim),
            nn.LeakyReLU(),
            nn.Dropout(args.drop),
        )

        hetergconvLayer_tv = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_tv = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_tv,
            args.heter_n_layers[0],
            args.drop,
            args.no_cuda,
        )
        hetergconvLayer_ta = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_ta = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_ta,
            args.heter_n_layers[1],
            args.drop,
            args.no_cuda,
        )
        hetergconvLayer_va = HeterGConvLayer(args.hidden_dim, args.drop,
                                             args.no_cuda)
        self.hetergconv_va = HeterGConv_Edge(
            args.hidden_dim,
            hetergconvLayer_va,
            args.heter_n_layers[2],
            args.drop,
            args.no_cuda,
        )

        self.modal_fusion = nn.Sequential(
            nn.Linear(args.hidden_dim, args.hidden_dim),
            nn.LeakyReLU(),
        )

        self.emo_output = nn.Linear(args.hidden_dim, n_classes_emo)
        self.sen_output = nn.Linear(args.hidden_dim, 3)
        self.senshift = SenShift_Feat(args.hidden_dim, args.drop,
                                      args.shift_win)

    def forward(self, feature_t0, feature_t1, feature_t2, feature_t3,
                feature_v, feature_a, umask, qmask, dia_lengths):

        all_t_features = [feature_t0, feature_t1, feature_t2, feature_t3]
        seq_len_t, batch_size_t, feature_dim_t = feature_t0.shape
        used_t_features = []
        for idx, bn in zip(self.used_t_indices, self.batchnorms_t):
            feat = all_t_features[idx]
            feat_bn = bn(feat.transpose(0, 1).reshape(-1, feature_dim_t))
            feat_bn = feat_bn.reshape(batch_size_t, seq_len_t, feature_dim_t).transpose(1, 0)
            used_t_features.append(feat_bn)

        if self.textf_mode in ['concat4', 'concat2']:
            merged_t_feat = torch.cat(used_t_features, dim=-1)
        elif self.textf_mode in ['sum4', 'sum2']:
            merged_t_feat = sum(used_t_features) / len(used_t_features)
        else:
            merged_t_feat = used_t_features[0]
        featdim_t = self.dim_layer_t(merged_t_feat)
        featdim_v, featdim_a = self.dim_layer_v(feature_v), self.dim_layer_a(
            feature_a)

        emo_t, emo_v, emo_a = featdim_t, featdim_v, featdim_a

        emo_t, emo_v, emo_a = batch_to_all_tva(emo_t, emo_v, emo_a,
                                               dia_lengths, self.no_cuda)

        featheter_tv, heter_edge_index = self.hetergconv_tv(
            (emo_t, emo_v), dia_lengths, self.win_p, self.win_f)
        featheter_ta, heter_edge_index = self.hetergconv_ta(
            (emo_t, emo_a), dia_lengths, self.win_p, self.win_f,
            heter_edge_index)
        featheter_va, heter_edge_index = self.hetergconv_va(
            (emo_v, emo_a), dia_lengths, self.win_p, self.win_f,
            heter_edge_index)

        feat_fusion = (self.modal_fusion(featheter_tv[0]) + self.modal_fusion(
            featheter_ta[0]) + self.modal_fusion(featheter_tv[1]) +
                       self.modal_fusion(featheter_va[0]) +
                       self.modal_fusion(featheter_ta[1]) +
                       self.modal_fusion(featheter_va[1])) / 6

        logit_emo = self.emo_output(feat_fusion)
        logit_sen = self.sen_output(feat_fusion)

        logit_shift = self.senshift(feat_fusion, feat_fusion, dia_lengths)

        return logit_emo, logit_sen, logit_shift, feat_fusion
