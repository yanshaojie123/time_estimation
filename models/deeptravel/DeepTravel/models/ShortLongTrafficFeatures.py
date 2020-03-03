import torch
import torch.nn as nn


class ShortTermLSTM(nn.Module):

    def __init__(self, ):
        super(ShortTermLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        nn.init.uniform_(self.lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, short_ttf):

        V_short = []

        for cell in short_ttf:

            inputs_0, inputs_1, inputs_2 = cell[0], cell[1], cell[2]
            inputs_0 = torch.unsqueeze(inputs_0, dim=0)
            inputs_1 = torch.unsqueeze(inputs_1, dim=0)
            inputs_2 = torch.unsqueeze(inputs_2, dim=0)

            outputs_0, (h_n, c_n) = self.lstm(inputs_0)
            outputs_1, (h_n, c_n) = self.lstm(inputs_1)
            outputs_2, (h_n, c_n) = self.lstm(inputs_2)

            hiddens_v = torch.squeeze(torch.cat([outputs_0[:, -1], outputs_1[:, -1], outputs_2[:, -1]], dim=1), dim=0)

            V_short.append(hiddens_v)

        return V_short


class LongTermLSTM(nn.Module):

    def __init__(self, ):
        super(LongTermLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=4,
            hidden_size=100,
            num_layers=1,
            batch_first=True
        )

        nn.init.uniform_(self.lstm.state_dict()['weight_hh_l0'], a=-0.05, b=0.05)

    def forward(self, long_ttf):

        V_long = []

        for cell in long_ttf:
            inputs = torch.unsqueeze(cell, dim=0)
            outputs, (h_n, c_n) = self.lstm(inputs)

            V_long.append(torch.squeeze(outputs[:, -1], dim=0))

        return V_long
