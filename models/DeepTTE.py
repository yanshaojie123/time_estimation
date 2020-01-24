import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

EPS = 10


def get_local_seq(full_seq, kernel_size, mean=0, std=1):
    seq_len = full_seq.size()[1]

    # if torch.cuda.is_available():
    #     indices = torch.cuda.LongTensor(seq_len)
    # else:
    #     indices = torch.LongTensor(seq_len)
    indices = torch.LongTensor(seq_len).to(full_seq.device)

    torch.arange(0, seq_len, out=indices)

    indices = Variable(indices, requires_grad=False)

    first_seq = torch.index_select(full_seq, dim=1, index=indices[kernel_size - 1:])
    second_seq = torch.index_select(full_seq, dim=1, index=indices[:-kernel_size + 1])

    local_seq = first_seq - second_seq

    local_seq = (local_seq - mean) / std

    return local_seq


class Attr(nn.Module):
    embed_dims = [('driverID', 24000, 16), ('weekID', 7, 3), ('timeID', 1440, 8)]  # todo: adaptive embeded_dim

    def __init__(self):
        super(Attr, self).__init__()
        # whether to add the two ends of the path into Attribute Component
        for name, dim_in, dim_out in Attr.embed_dims:
            self.add_module(name + '_em', nn.Embedding(dim_in, dim_out))
#
#     # def out_size(self):
#     #     sz = 0
#     #     for name, dim_in, dim_out in Attr.embed_dims:
#     #         sz += dim_out
#     #     # append total distance
#     #     return sz + 1

    def forward(self, attr, config):
        '''

        :param attr: {"driverID": (N*24000dim), "weekID": (N*7dim), "timeID": (N*1440dim)}
        :return: N*(16+3+8+1)dim
        '''

        em_list = []
        for name, dim_in, dim_out in Attr.embed_dims:
            embed = getattr(self, name + '_em')
            attr_t = attr[name].view(-1, 1)
            # attr_t = attr_t.to('cuda:0')
            # print(attr_t.device)
            attr_t = torch.squeeze(embed(attr_t))
            em_list.append(attr_t)

        # dist_mean = config.data_config['dist_mean']
        # dist_std = config.data_config['dist_std']
        # dist = (attr['dist'] - dist_mean) / dist_std
        em_list.append(attr['dist'].view(-1, 1))
        # em_list.append(attr['dist'].view(-1, 1))

        return torch.cat(em_list, dim=1)


class GeoConv(nn.Module):
    def __init__(self, kernel_size, num_filter, embedding_dim=16):
        super(GeoConv, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter

        self.state_em = nn.Embedding(2, 2)
        self.process_coords = nn.Linear(4, embedding_dim)
        self.conv = nn.Conv1d(embedding_dim, self.num_filter, self.kernel_size)


    def forward(self, traj, config):
        '''

        :param traj: {'lngs':, 'lats', 'states', 'dist_gap'}
        :param config:
        :return: (B, S-K+1, num_filter+1)
        '''
        lngs = torch.unsqueeze(traj['lngs'], dim=2)
        lats = torch.unsqueeze(traj['lats'], dim=2)
        # print(traj.keys())
        states = self.state_em(traj['states'].long())

        locs = torch.cat((lngs, lats, states), dim=2)

        # map the coords into 16-dim vector
        # print(locs.device)
        locs = F.tanh(self.process_coords(locs))

        locs = locs.permute(0, 2, 1)

        # locs: (B, embedding_size, seqlen)
        conv_locs = F.elu(self.conv(locs)).permute(0, 2, 1)

        # conv_locs: (B, Seqlen-kernel_size+1, num_filter)
        # calculate the dist for local paths
        local_dist = get_local_seq(traj['dist_gap'], self.kernel_size)

        dist_gap_mean = config.data_config['dist_gap_mean']
        dist_gap_std = config.data_config['dist_gap_std']
        local_dist = (local_dist - dist_gap_mean) / dist_gap_std
        local_dist = torch.unsqueeze(local_dist, dim=2)

        conv_locs = torch.cat((conv_locs, local_dist), dim=2)
        return conv_locs


class SpatioTemporal(nn.Module):
    '''
    attr_size: the dimension of attr_net output
    pooling optitions: last, mean, attention
    '''

    def __init__(self, attr_size, kernel_size=3, num_filter=32, pooling_method='attention', rnn='lstm', rnn_size=128):
        super(SpatioTemporal, self).__init__()

        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method

        self.geo_conv = GeoConv(kernel_size=kernel_size, num_filter=num_filter)
        # num_filter: output size of each GeoConv + 1:distance of local path + attr_size: output size of attr component
        if rnn == 'lstm':
            self.rnn = nn.LSTM(input_size=num_filter + 1 + attr_size,
                               hidden_size=rnn_size,
                               num_layers=2,
                               batch_first=True
                               )
        elif rnn == 'rnn':
            self.rnn = nn.RNN(input_size=num_filter + 1 + attr_size,
                              hidden_size=rnn_size,
                              num_layers=1,
                              batch_first=True
                              )

#         if pooling_method == 'attention':
#             self.attr2atten = nn.Linear(attr_size, 128)
#

    def mean_pooling(self, hiddens, lens):
        # note that in pad_packed_sequence, the hidden states are padded with all 0
        hiddens = torch.sum(hiddens, dim=1, keepdim=False)
        lens = lens.to(hiddens.device)
#         if torch.cuda.is_available():
#             lens = torch.cuda.FloatTensor(lens)
#         else:
#             lens = torch.FloatTensor(lens)
#
        lens = Variable(torch.unsqueeze(lens, dim=1), requires_grad=False)

        hiddens = hiddens / lens

        return hiddens

#     def attent_pooling(self, hiddens, lens, attr_t):  # todo: attention pooling
#         attent = F.tanh(self.attr2atten(attr_t)).permute(0, 2, 1)
#
#         # hidden b*s*f atten b*f*1 alpha b*s*1 (s is length of sequence)
#         alpha = torch.bmm(hiddens, attent)
#         alpha = torch.exp(-alpha)
#
#         # The padded hidden is 0 (in pytorch), so we do not need to calculate the mask
#         alpha = alpha / torch.sum(alpha, dim=1, keepdim=True)
#
#         hiddens = hiddens.permute(0, 2, 1)
#         hiddens = torch.bmm(hiddens, alpha)
#         hiddens = torch.squeeze(hiddens)
#
#         return hiddens
#
    def forward(self, traj, attr_t, config):
        conv_locs = self.geo_conv(traj, config)

        attr_t = torch.unsqueeze(attr_t, dim=1)
        # attr_t): (B,1,attr_size)
        expand_attr_t = attr_t.expand(conv_locs.size()[:2] + (attr_t.size()[-1],))

        # concat the loc_conv and the attributes
        conv_locs = torch.cat((conv_locs, expand_attr_t), dim=2)

        lens = list(map(lambda x: x - self.kernel_size + 1, traj['lens']))

        packed_inputs = nn.utils.rnn.pack_padded_sequence(conv_locs, lens, batch_first=True, enforce_sorted=False)

        packed_hiddens, (h_n, c_n) = self.rnn(packed_inputs)
        hiddens, lens = nn.utils.rnn.pad_packed_sequence(packed_hiddens, batch_first=True)

        if self.pooling_method == 'mean':
            return packed_hiddens, lens, self.mean_pooling(hiddens, lens)

#         elif self.pooling_method == 'attention':
#             return packed_hiddens, lens, self.attent_pooling(hiddens, lens, attr_t)


class EntireEstimator(nn.Module):
    def __init__(self, input_size, num_final_fcs, hidden_size=128):
        super(EntireEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, hidden_size)

        self.residuals = nn.ModuleList()
        for i in range(num_final_fcs):
            self.residuals.append(nn.Linear(hidden_size, hidden_size))

        self.hid2out = nn.Linear(hidden_size, 1)

    def forward(self, attr_t, sptm_t):
        inputs = torch.cat((attr_t, sptm_t), dim=1)

        hidden = F.leaky_relu(self.input2hid(inputs))

        for layer in self.residuals:
            residual = F.leaky_relu(layer(hidden))
            hidden = hidden + residual

        out = self.hid2out(hidden)

        return out


class LocalEstimator(nn.Module):
    def __init__(self, input_size):
        super(LocalEstimator, self).__init__()

        self.input2hid = nn.Linear(input_size, 64)
        self.hid2hid = nn.Linear(64, 32)
        self.hid2out = nn.Linear(32, 1)

    def forward(self, sptm_s):
        hidden = F.leaky_relu(self.input2hid(sptm_s))

        hidden = F.leaky_relu(self.hid2hid(hidden))

        out = self.hid2out(hidden)

        return out

    def eval_on_batch(self, pred, lens, label, mean, std):
        label = nn.utils.rnn.pack_padded_sequence(label, lens, batch_first=True)[0]
        label = label.view(-1, 1)

        label = label * std + mean
        pred = pred * std + mean

        loss = torch.abs(pred - label) / (label + EPS)

        return loss.mean()


class DeepTTE(nn.Module):
    def __init__(self, kernel_size=3, num_filter=32, pooling_method='attention', num_final_fcs=3, final_fc_size=128,
                 alpha=0.3, **kwargs):
        super(DeepTTE, self).__init__()
        self.kernel_size = kernel_size
        self.num_filter = num_filter
        self.pooling_method = pooling_method
        self.num_final_fcs = num_final_fcs
        self.final_fc_size = final_fc_size
        self.alpha = alpha

        rnn_size = 128
        attr_size = 28
        self.attr_net = Attr()
        self.spatio_temporal = SpatioTemporal(attr_size=attr_size,  # todo: outsize
                                              kernel_size=self.kernel_size,
                                              num_filter=self.num_filter,
                                              pooling_method=self.pooling_method,
                                              rnn_size=rnn_size
                                              )
        self.entire_estimate = EntireEstimator(input_size=rnn_size + attr_size,
                                               num_final_fcs=self.num_final_fcs, hidden_size=self.final_fc_size)
        self.local_estimate = LocalEstimator(input_size=rnn_size)

        self.init_weight()

    def init_weight(self):
        for name, param in self.named_parameters():
            if name.find('.bias') != -1:
                param.data.fill_(0)
            elif name.find('.weight') != -1:
                nn.init.xavier_uniform(param.data)

    def forward(self, inputs, truth_data, config, **kwargs):
        attr = inputs['attr']
        traj = inputs['traj']

        for key in ['dist']:
            attr[key] = (attr[key] - config.data_config[key + '_mean']) / config.data_config[key + '_std']
        for key in ['lngs', 'lats', 'dist_gap']:
            traj[key] = (traj[key] - config.data_config[key+'_mean']) / config.data_config[key+'_std']

        attr_t = self.attr_net(attr, config)

        # sptm_s: hidden sequence (B * T * F); sptm_l: lens (list of int); sptm_t: merged tensor after attention/mean pooling
        sptm_s, sptm_l, sptm_t = self.spatio_temporal(traj, attr_t, config)

        entire_out = self.entire_estimate(attr_t, sptm_t)
        entire_label = truth_data.view(-1, 1)

        entire_std = config.data_config['time_std']
        entire_mean = config.data_config['time_mean']
        # entire_label = entire_label * entire_std + entire_mean
        entire_out = entire_out * entire_std + entire_mean

        entire_loss = torch.abs(entire_out - entire_label) / entire_label

        entire_loss = entire_loss.mean()
        # sptm_s is a packed sequence (see pytorch doc for details), only used during the training
        if self.training:
            local_pred = self.local_estimate(sptm_s[0])

            # get ground truth of each local path
            local_label = get_local_seq(traj['time_gap'], self.kernel_size)
            local_label = nn.utils.rnn.pack_padded_sequence(local_label, sptm_l, batch_first=True, enforce_sorted=False)[0]
            local_label = local_label.view(-1, 1)

            local_mean = (self.kernel_size - 1) * config.data_config['time_gap_mean']
            local_std = (self.kernel_size - 1) * config.data_config['time_gap_std']
            # local_label = local_label * local_std + local_mean
            local_pred = local_pred * local_std + local_mean

            local_loss = torch.abs(local_pred - local_label) / (local_label + EPS)
            local_loss = local_loss.mean()

            return (1 - self.alpha) * entire_loss + self.alpha * local_loss, entire_out
        else:
            return entire_loss, entire_out
