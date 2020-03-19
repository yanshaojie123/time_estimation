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
        self.highwayembed = nn.Embedding(15, 10, padding_idx=0)
        self.bridgeembed = nn.Embedding(3, 3, padding_idx=0)
        self.tunnelembed = nn.Embedding(4, 4, padding_idx=0)
        self.weekembed = nn.Embedding(8, 3)
        self.dateembed = nn.Embedding(366, 10)
        self.timeembed = nn.Embedding(1440, 20)
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
        self.output = nn.Sequential(
            nn.Linear(seq_hidden_dim, int(seq_hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(seq_hidden_dim/2), 1),
        )
        self.conv = nn.Conv1d(52, 52, 3, padding=False)
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
        # print(hiddens.shape)
        # print(lens.shape)
        # result = hiddens[:, -1, :]
        result = hiddens[list(range(hiddens.shape[0])), lens.squeeze()-1]
        # print(result.shape)
        # hiddens = hiddens / lens
        # return hiddens
        return result

    def forward(self, inputs, args):
        feature = inputs['links']
        # highway bridge tunnel week date time
        # length sumlength lanes maxspeed width
        # 32 Skip Gram embedding
        # date = inputs['date']
        lens = inputs['lens']

        highwayrep = self.highwayembed(feature[:, :, 0].long())  # index to vector
        bridgerep = self.bridgeembed(feature[:, :, 1].long())
        tunnelrep = self.tunnelembed(feature[:, :, 2].long())
        weekrep = self.weekembed(feature[:, :, 3].long())
        daterep = self.dateembed(feature[:, :, 4].long())
        timerep = self.timeembed(feature[:, :, 5].long())
        # print(bridgerep.shape)
        # print(feature.shape)
        # highwayrep = torch.zeros(list(highwayrep.shape)).to(args.device)
        # bridgerep = torch.zeros(list(bridgerep.shape)).to(args.device)
        # tunnelrep = torch.zeros(list(tunnelrep.shape)).to(args.device)
        # weekrep = torch.zeros(list(weekrep.shape)).to(args.device)
        # daterep = torch.zeros(list(daterep.shape)).to(args.device)
        # timerep = torch.zeros(list(timerep.shape)).to(args.device)


        # representation = self.represent(torch.cat([feature[:, :, 6:43], highwayrep, bridgerep, tunnelrep, weekrep, daterep, timerep], dim=-1))

        representation = self.represent(torch.cat([feature[..., 6:11], highwayrep, bridgerep, tunnelrep, weekrep, daterep, timerep], dim=-1))
        representation = torch.cat([representation, feature[...,11:]], dim = -1)
        # print(representation.shape)
        # representation = F.elu(self.conv(representation.permute(0, 2, 1))).permute(0, 2, 1)
        # lens = lens-2
        # representation = torch.cat([representation, torch.zeros(feature[...,11:].shape).to(args.device)], dim = -1)

        # representation = torch.ones(representation.shape).to(representation.device)

        # hiddens = self.sequence(representation.unsqueeze(2)).squeeze()

        packed_inputs = nn.utils.rnn.pack_padded_sequence(representation, lens, batch_first=True, enforce_sorted=False)
        # print(representation.shape)
        packed_hiddens, c_n = self.sequence(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        pooled_hidden = self.pooling(hiddens, lens)
        # pooled_hidden = hiddens[:, -1]

        output = self.output(pooled_hidden)
        output = output * 231.2591+490.5749
        return output

        # return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

