import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import math

class TRANS(torch.nn.Module):
    def __init__(self, cuda_use, feature_dim, embed_dim, mem_dim, outputdim, dropout):
        super(TRANS, self).__init__()

        n_layers = 8
        attention_dropout_rate = 0.1
        head_size = 8
        ffn_dim = 32

        self.use_time = 0.0
        self.cuda_use = cuda_use
        self.feature_dim = feature_dim
        self.input_dim = embed_dim
        self.mem_dim = mem_dim
        self.outputdim = outputdim
        self.operat_dim = 4
        self.table_dim = 11
        self.filter_dim = 18
        self.join_dim = 37
        self.max_len = 16

        self.dropout = nn.Dropout(p = dropout)
        pe = torch.zeros(self.max_len, self.input_dim)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.input_dim, 2) * 
                             -(math.log(10000.0) / (self.input_dim)))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term) 
        self.register_buffer('pe', pe)

        self.feature_mpl_operation = torch.nn.Linear(self.operat_dim, self.input_dim)
        self.feature_mpl_table = torch.nn.Linear(self.table_dim, self.input_dim)
        self.feature_mpl_filter = torch.nn.Linear(self.filter_dim, self.input_dim)
        self.feature_mpl_join = torch.nn.Linear(self.join_dim, self.input_dim)

        self.feature_mpl_operation_2 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.feature_mpl_table_2 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.feature_mpl_filter_2 = torch.nn.Linear(self.input_dim, self.input_dim)
        self.feature_mpl_join_2 = torch.nn.Linear(self.input_dim, self.input_dim)

        self.transformer_encoder_1 = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(self.input_dim * 2, nhead = 1),
            num_layers = 1
        )

        self.out_mlp1 = nn.Linear(self.input_dim * 4, self.input_dim * 2)
        self.out_mlp2 = nn.Linear(self.input_dim * 2, 1)

        


    def forward(self, op_feat, tb_feat, ft_feat, join_feat, node_order, adjacency_list, edge_order):
        edge_order = torch.cat((torch.tensor([node_order.max() + 1]), edge_order), dim=0)

        op_feat = F.relu(self.feature_mpl_operation(op_feat))
        op_feat = F.relu(self.feature_mpl_operation_2(op_feat))
        tb_feat = F.relu(self.feature_mpl_table(tb_feat))
        tb_feat = F.relu(self.feature_mpl_table_2(tb_feat))
        ft_feat = F.relu(self.feature_mpl_filter(ft_feat))
        ft_feat = F.relu(self.feature_mpl_filter_2(ft_feat))
        join_feat = F.relu(self.feature_mpl_join(join_feat))
        join_feat = F.relu(self.feature_mpl_join_2(join_feat))

        batch_size = node_order.shape[0]
        h2 = torch.zeros(batch_size, self.input_dim)
        h4 = torch.zeros(batch_size, self.input_dim)

        for n in range(1, node_order.max() + 1):
            edge_mask = edge_order == n
            height = node_order.max() + 1 - n
            h2[edge_mask, :] = tb_feat[edge_mask, :] + self.pe[height,:]
            h4[edge_mask, :] = join_feat[edge_mask, :] + self.pe[height,:]

        enc_output = torch.cat([h2, h4], dim=1)

        enc_output = enc_output.unsqueeze(0)
        enc_output = enc_output.permute(1, 0, 2) 
        enc_output = self.transformer_encoder_1(enc_output) 
        enc_output = enc_output.permute(1, 0, 2)
        enc_output = enc_output.squeeze(0)
        
        enc_output = torch.cat([enc_output, op_feat, ft_feat], dim=1)
        
        hid_output = F.relu(self.out_mlp1(enc_output))
        out = torch.sigmoid(self.out_mlp2(hid_output))
        return out