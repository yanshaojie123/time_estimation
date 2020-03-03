import torch
from torch import nn
from models.MHA import MyBlock


class TTEModel(nn.Module):
    def __init__(self, input_dim, seq_input_dim, seq_hidden_dim, seq_layer):
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
            nn.Linear(input_dim, seq_input_dim),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(seq_input_dim, seq_input_dim),
            nn.ReLU(),
            # nn.Dropout(0.3),
            nn.Linear(seq_input_dim, seq_input_dim)
            # nn.ReLU()
        )
        # self.sequence = nn.LSTM(seq_input_dim, seq_hidden_dim, seq_layer, batch_first=True)
        self.sequence = nn.GRU(seq_input_dim, seq_hidden_dim, seq_layer, batch_first=True)
        self.output = nn.Sequential(
            nn.Linear(seq_hidden_dim, int(seq_hidden_dim/2)),
            nn.ReLU(),
            nn.Linear(int(seq_hidden_dim/2), 1)
        )
        # self.sequence = MyBlock(1, seq_input_dim, seq_hidden_dim, seq_input_dim, a_query=15, a_value=15)
        # self.output = nn.Sequential(
        #     nn.Linear(seq_input_dim, int(seq_hidden_dim/2)),
        #     nn.ReLU(),
        #     nn.Linear(int(seq_hidden_dim/2), 1)
        # )

    def mean_pooling(self, hiddens, lens):
        hiddens = torch.sum(hiddens, dim=1, keepdim=False)
        lens = lens.to(hiddens.device)
        #         if torch.cuda.is_available():
        #             lens = torch.cuda.FloatTensor(lens)
        #         else:
        #             lens = torch.FloatTensor(lens)
        #
        lens = torch.autograd.Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)

        # hiddens = hiddens / lens
        return hiddens

    def forward(self, inputs, args):
        feature = inputs['links']
        # date = inputs['date']
        lens = inputs['lens']

        highwayrep = self.highwayembed(feature[:, :, 0].long())
        bridgerep = self.bridgeembed(feature[:, :, 1].long())
        tunnelrep = self.tunnelembed(feature[:, :, 2].long())
        weekrep = self.weekembed(feature[:, :, 3].long())
        daterep = self.dateembed(feature[:, :, 4].long())
        timerep = self.timeembed(feature[:, :, 5].long())
        # print(bridgerep.shape)
        # print(feature.shape)
        representation = self.represent(torch.cat([feature[:, :, -5:], highwayrep, bridgerep, tunnelrep, weekrep, daterep, timerep], dim=-1))

        # hiddens = self.sequence(representation.unsqueeze(2)).squeeze()

        packed_inputs = nn.utils.rnn.pack_padded_sequence(representation, lens, batch_first=True, enforce_sorted=False)
        packed_hiddens, c_n = self.sequence(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        pooled_hidden = self.mean_pooling(hiddens, lens)
        # pooled_hidden = hiddens[:, -1]

        output = self.output(pooled_hidden)
        return output

        # return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

