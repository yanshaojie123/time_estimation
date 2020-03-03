import torch
import torch.nn as nn
from torch.autograd import Variable


class PredictionBiLSTM(nn.Module):

    def __init__(self, ):
        super(PredictionBiLSTM, self).__init__()
        # 616 426
        self.bi_lstm = nn.LSTM(
            input_size=904,
            hidden_size=100,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        nn.init.uniform_(self.bi_lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

        # self.h_n = Variable(torch.zeros(self.bi_lstm.num_layers * 2, 1, self.bi_lstm.hidden_size)).cuda()
        # self.c_n = Variable(torch.zeros(self.bi_lstm.num_layers * 2, 1, self.bi_lstm.hidden_size)).cuda()

    def forward(self, V_cells):

        H_cells = []

        for V_vector in V_cells:
            inputs = V_vector.view(1, 1, -1)

            # hiddens, (self.h_n, self.c_n) = self.bi_lstm(inputs, (self.h_n, self.c_n))
            hiddens, (h_n, c_n) = self.bi_lstm(inputs)
            H_cells.append(hiddens.squeeze(dim=0).squeeze(dim=0))               # len [200]

        return H_cells
