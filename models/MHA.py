import torch
from torch import nn
import math
import numpy as np
import torch.nn.functional as F


class MHAttention(nn.Module):
    def __init__(self, input_dim, qinput_dim, query_dim, value_dim, output_dim, num_heads, drop_prob=0.3):
        """

        :param input_dim: input dimension
        :param query_dim: dimension of query vector and key vector
        :param value_dim: value vector dimension
        :param output_dim: output dimension
        :param num_heads: number of attention heads
        :param drop_prob: dropout probability
        """
        super(MHAttention, self).__init__()
        self.num_heads = num_heads
        self.query_dim = query_dim
        self.output_dim = output_dim
        self.input_dim = input_dim

        self.QLinear = nn.Linear(input_dim, query_dim * num_heads)
        self.KLinear = nn.Linear(input_dim, query_dim * num_heads)
        self.VLinear = nn.Linear(input_dim, value_dim * num_heads)
        self.Output = nn.Sequential(nn.Linear(value_dim * num_heads, output_dim),
                                    nn.Dropout(0.2),
                                    )
        self.Norm = nn.LayerNorm(output_dim, eps=1e-5)
        self.Softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(drop_prob)

    def reshape_for_score(self, x):
        """

        :param x:
        :return: (Batchsize, numnodes, Time, query_dim/value_dim)
        """
        x = x.reshape([x.shape[0], x.shape[1], self.num_heads, -1])
        return x.permute(0, 2, 1, 3).contiguous()

    def forward(self, source):
        """
        :param encoder_hidden_states:
        :param attention_mask:
        :param inputs:  (Batchsize, Time, input_dim/NF)
        :return:  (Batchsize, Time, output_dim/N*)
        """
        inputs, attention_mask, encoder_hidden_states = source
        residual = inputs
        # todo residual size

        # (Batchsize, num_heads, Time, query_dim/value_dim)
        q = self.reshape_for_score(self.QLinear(inputs))
        if encoder_hidden_states is not None:
            k = self.reshape_for_score(self.KLinear(encoder_hidden_states))
            v = self.reshape_for_score(self.VLinear(encoder_hidden_states))
        else:
            k = self.reshape_for_score(self.KLinear(inputs))
            v = self.reshape_for_score(self.VLinear(inputs))

        attention_score = q.matmul(k.transpose(-1, -2))/math.sqrt(self.query_dim)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_score = attention_score + attention_mask
        # (Time, Time)
        attention_prob = self.Softmax(attention_score)
        attention_prob = self.dropout(attention_prob)

        value = attention_prob.matmul(v).transpose(1, 2).contiguous()
        # (Batchsize, Time, query_dim * num_heads)
        value = value.reshape([value.shape[0], value.shape[1], -1])
        value = self.Output(value)
        return self.Norm(value + residual), attention_mask, encoder_hidden_states


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """

        :param input_dim: input dimension of first fc layer
        :param hidden_dim: input dimension of second fc layer
        :param output_dim: output dimension of second fc layer
        """
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.FF = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.Norm = nn.LayerNorm(output_dim, eps=1e-5)

    def forward(self, inputs):
        """

        :param inputs: (**, input_dim)
        :return: (**, output_dim)
        """
        inputs = inputs.reshape([-1, self.input_dim])
        residual = inputs
        # todo: residual size
        # print(residual.shape)
        output = self.Norm(residual + self.FF(inputs))
        return output


class MyBlock(nn.Module):
    def __init__(self, numnodes, input_dim, hidden_dim, out_dim,
                 a_query=64, a_value=64, a_nheads=8):
        super(MyBlock, self).__init__()
        self.num_nodes = numnodes
        self.infeatures = input_dim
        self.outfeatures = out_dim
        # a_value*a_nheads maybe bigger than numnodes*features
        self.MHAttention = MHAttention(numnodes*input_dim, numnodes*input_dim, a_query, a_value, numnodes*input_dim, a_nheads)
        # if support is None:
        #     self.FFN = FFN(numnodes*input_dim, numnodes * hidden_dim, numnodes*out_dim)
        self.FFN = FFN(numnodes*input_dim, numnodes * hidden_dim, numnodes*out_dim)
        # else:
        #     self.FFN = nn.Sequential(
        #         GraphConv(support, input_dim, hidden_dim, numnodes),
        #         nn.ReLU(),
        #         GraphConv(support, hidden_dim, out_dim, numnodes)
        #     )
        # self.FFN1 = GraphConv(support, input_dim, hidden_dim, numnodes)
        # self.FFN2 = GraphConv(support, hidden_dim, out_dim, numnodes)
    def forward(self, inputs):
        """

        :param encoder_state:
        :param inputs: (Batchsize, seq_input, Numnodes, Features)
        :return: (Batchsize, seq_input, Numnodes, out_dim)
        """
        # inputs = inputs[:, :self.seq_input]
        # print(inputs.shape)
        batch_size = inputs.shape[0]
        seq_input = inputs.shape[1]
        attention_output = self.MHAttention((inputs.reshape([batch_size, seq_input, -1]), None, None))
        attention_output = attention_output[0].reshape([batch_size*seq_input, self.num_nodes, self.infeatures])
        output = self.FFN(attention_output)
        # output = self.FFN2(F.relu(self.FFN1(attention_output, support=support)), support=support)
        return output.reshape([batch_size, seq_input, self.num_nodes, self.outfeatures])
