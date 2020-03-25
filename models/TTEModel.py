import torch
from torch import nn
from models.MHA import MyBlock
import torch.nn.functional as F


class TTEModel(nn.Module):
    def __init__(self, input_dim, out_dim, seq_input_dim, seq_hidden_dim, seq_layer):
        '''
        edge feature: highway, bridge, tunnel, length, lanes, maxspeed, width
        :param input_dim:
        '''
        super(TTEModel, self).__init__()
        self.highwayembed = nn.Embedding(15, 5, padding_idx=0)
        # self.bridgeembed = nn.Embedding(3, 3, padding_idx=0)
        # self.tunnelembed = nn.Embedding(4, 4, padding_idx=0)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = nn.Embedding(366, 10)
        self.timeembed = nn.Embedding(1440, 20)
        self.gpsrep = nn.Linear(4, 16)
        self.represent = nn.Sequential(
            nn.Linear(input_dim, out_dim),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(out_dim, out_dim),
            # nn.ReLU(),
            # nn.Dropout(0.3),
            # nn.Linear(out_dim, out_dim)
            # nn.ReLU()
        )
        self.sequence = nn.LSTM(seq_input_dim, seq_hidden_dim, seq_layer, batch_first=True)
        # self.sequence = nn.GRU(seq_input_dim, seq_hidden_dim, seq_layer, batch_first=True)
        # self.output = nn.Sequential(
        #     nn.Linear(seq_hidden_dim, int(seq_hidden_dim/2)),
        #     nn.ReLU(),
        #     nn.Linear(int(seq_hidden_dim/2), 1),
        # )
        residual_dim = seq_hidden_dim
        # self.input2hid = nn.Linear(residual_dim, residual_dim)
        self.input2hid = nn.Linear(residual_dim+33, residual_dim)
        self.residuals = nn.ModuleList()
        for i in range(3):
            self.residuals.append(nn.Linear(residual_dim, residual_dim))
        self.hid2out = nn.Linear(residual_dim, 1)
        # self.bn1 = nn.BatchNorm1d(52, affine=False)

        # self.conv = nn.Conv1d(52, 52, 3, padding=False)
        # self.sequence = MyBlock(1, seq_input_dim, seq_hidden_dim, seq_input_dim, a_query=15, a_value=15)
        # self.output = nn.Sequential(
        #     nn.Linear(seq_input_dim, int(seq_hidden_dim/2)),
        #     nn.ReLU(),
        #     nn.Linear(int(seq_hidden_dim/2), 1)
        # )

    def pooling(self, hiddens, lens):
        # result = torch.sum(hiddens, dim=1, keepdim=False)
        # print(hiddens2.shape)
        lens = lens.to(hiddens.device)
        #         if torch.cuda.is_available():
        #             lens = torch.cuda.FloatTensor(lens)
        #         else:
        #             lens = torch.FloatTensor(lens)
        #
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)
        result = hiddens[list(range(hiddens.shape[0])), lens.squeeze()-1]
        # hiddens = hiddens / lens
        # return hiddens
        return result

    def forward(self, inputs, args):
        feature = inputs['links']
        # highway bridge tunnel week date time
        # length sumlength lanes maxspeed width
        # 32 Skip Gram embedding
        lens = inputs['lens']

        highwayrep = self.highwayembed(feature[:, :, 0].long())  # index to vector 5
        weekrep = self.weekembed(feature[:, :, 3].long())  # 33
        daterep = self.dateembed(feature[:, :, 4].long())
        timerep = self.timeembed(feature[:, :, 5].long())
        gpsrep = self.gpsrep(feature[:, :, 6:10])  # 16
        # highwayrep = torch.zeros(list(highwayrep.shape)).to(args.device)
        # weekrep = torch.zeros(list(weekrep.shape)).to(args.device)
        # daterep = torch.zeros(list(daterep.shape)).to(args.device)
        # timerep = torch.zeros(list(timerep.shape)).to(args.device)
        # gpsrep = torch.zeros(list(gpsrep.shape)).to(args.device)

        # representation = self.represent(torch.cat([feature[..., 1:3], highwayrep, gpsrep], dim=-1))
        # representation = self.represent(torch.cat([feature[..., 1:3], feature[...,10:], highwayrep, gpsrep, weekrep, daterep, timerep], dim=-1))
        representation = self.represent(torch.cat([feature[..., 1:3], highwayrep, gpsrep, weekrep, daterep, timerep], dim=-1))
        # representation = torch.cat([feature[..., 1:3], highwayrep, gpsrep, weekrep, daterep, timerep], dim=-1)
        # representation = self.represent(torch.cat([ torch.zeros(list(feature.shape[:-1]) + [2]).to(feature.device), highwayrep, gpsrep, weekrep, daterep, timerep], dim=-1))

        representation = torch.cat([representation, feature[...,10:]], dim = -1)
        # representation = self.represent(representation)
        # representation = torch.cat([representation, torch.zeros(list(feature.shape[:-1]) + [32]).to(feature.device)], dim = -1)
        # representation = torch.cat([representation, feature[...,10:], weekrep, daterep, timerep], dim = -1)

        # representation = F.elu(self.conv(representation.permute(0, 2, 1))).permute(0, 2, 1)
        # lens = lens-2

        # representation = torch.ones(representation.shape).to(representation.device)

        # hiddens = self.sequence(representation.unsqueeze(2)).squeeze()

        packed_inputs = nn.utils.rnn.pack_padded_sequence(representation, lens, batch_first=True, enforce_sorted=False)
        packed_hiddens, c_n = self.sequence(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        pooled_hidden = self.pooling(hiddens, lens)
        pooled_hidden = torch.cat([pooled_hidden, weekrep[:,0], daterep[:,0], timerep[:,0]],dim=-1)

        hidden = F.leaky_relu(self.input2hid(pooled_hidden))
        for layer in self.residuals:
            residual = F.leaky_relu(layer(hidden))
            hidden = hidden + residual
        output = self.hid2out(hidden)
        output = args.scaler.inverse_transform(output)
        return output

