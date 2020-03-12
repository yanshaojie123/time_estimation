import torch
import torch.nn as nn
import numpy as np

from models.ShortLongTrafficFeatures import  ShortTermLSTM, LongTermLSTM
from models.SpatialTemporal import SpatialTemporal
from models.PredictionBiLSTM import PredictionBiLSTM


class DeepTravel(nn.Module):
    def __init__(self, alpha=0.3):
        super(DeepTravel, self).__init__()

        self.alpha = alpha
        self.build()
        self.init_weight()

    def build(self):
        self.spatial_temporal = SpatialTemporal()
        self.short_term_lstm = ShortTermLSTM()
        self.long_term_lstm = LongTermLSTM()

        self.prediction = PredictionBiLSTM()

        self.linear = torch.nn.Linear(in_features=200, out_features=1)

    def evaluate(self, stats, temporal, spatial, dr_state, short_ttf, long_ttf, helpers):
        H_cells = self(stats, temporal, spatial, dr_state, short_ttf, long_ttf)
        loss = self.dual_loss(H_cells, helpers['time_gap'], helpers['borders'], helpers['mask'])

        return loss

    def forward(self, stats, temporal, spatial, dr_state, short_ttf, long_ttf):
        V_sp, V_tp = self.spatial_temporal(stats, temporal, spatial)    # [path_len, 10]; [path_len, 12]

        V_dri = dr_state.view(-1, 4)                                    # [path_len, 4]
        # print(short_ttf)
        V_short = self.short_term_lstm(short_ttf)                       # [300] for each cell
        V_long = self.long_term_lstm(long_ttf)                          # [100] for each cell

        V_cells = []
        for i in range(len(short_ttf)):
            V_cells.append(torch.cat([V_sp[0], V_tp[0], V_dri[0], V_short[0], V_long[0]]))      # path_len x [904]

        H_cells = self.prediction(V_cells)

        return H_cells

    def dual_loss(self, H_cells, times, borders, mask):
        device = 'cuda'
        n = len(H_cells)

        h_f = []
        h_b = []

        for i in range(1, n):
            h_f.append(sum(H_cells[:i]))
            h_b.append(sum(H_cells[i:]))

        h_f.append(sum(H_cells))

        T_f_hat = self.linear(torch.stack(h_f)).squeeze(dim=1)
        T_b_hat = self.linear(torch.stack(h_b)).squeeze(dim=1)

        T_b_hat = torch.cat([T_b_hat, torch.Tensor([0]).to(device)])

        M = np.zeros(n)
        T_f = np.ones(n)
        T_b = np.ones(n)

        for ind in range(len(H_cells)):
            if not mask[ind]:
                borders = np.insert(borders, ind, 0)
            if mask[ind]:
                M[ind] = 1
                T_f[ind] = borders[ind]
                T_b[ind] = times[-1] - borders[ind]

        M = torch.Tensor(M).to(device)
        T_f = torch.Tensor(T_f).to(device)
        T_b = torch.Tensor(T_b).to(device)

        loss = (M * ((T_f_hat - T_f) / T_f) ** 2 + M * ((T_b_hat - T_b) / T_b) ** 2) / (M * 2)

        # Temporary forced operation
        loss[loss != loss] = 0

        return loss

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.uniform_(param.data)
