import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import time
import math

class TRANS(torch.nn.Module):
    def __init__(self, cuda_use, feature_dim, embed_dim, mem_dim, outputdim, dropout):
        super(TRANS, self).__init__()

    #     embed_dim = 256
    # mem_dim = 1024 #
    # output_dim = 1024 #
    # batch_size = 50 #
    # lr_set = 0.0001 #
    # cuda_use = False
    # GPU_no = 1
    # wd_set = 0.00001
    # feature_dim = 70
    # train_path = './data/train_10K'
    # train_size = 11000
    # test_size = 520
    # dropout = 0.2

        n_layers = 8
        attention_dropout_rate = 0.1
        head_size = 8
        ffn_dim = 32

        self.use_time = 0.0
        self.cuda_use = cuda_use

        self.feature_dim = feature_dim
        self.input_dim = embed_dim # 输入模型的维度：256
        self.mem_dim = mem_dim # 输入模型的维度 x 4：1024
        self.outputdim = outputdim # 1024

        self.operat_dim = 4
        self.table_dim = 11
        self.filter_dim = 18
        self.join_dim = 37

        self.max_len = 16 # 对于层数的 embedding
        # 采用固定的对于层数的 embedding
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
        # self.transformer_encoder_2 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(self.input_dim, nhead = 1),
        #     num_layers = 1
        # )
        # self.transformer_encoder_3 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(self.input_dim, nhead = 1),
        #     num_layers = 1
        # )
        # self.transformer_encoder_4 = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(self.input_dim, nhead = 1),
        #     num_layers = 1
        # )

        self.out_mlp1 = nn.Linear(self.input_dim * 4, self.input_dim * 2)
        self.out_mlp2 = nn.Linear(self.input_dim * 2, 1)

        


    def forward(self, op_feat, tb_feat, ft_feat, join_feat, node_order, adjacency_list, edge_order):
        # 对于节点所在层数的编码
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
        # h1 = torch.zeros(batch_size, self.input_dim)
        h2 = torch.zeros(batch_size, self.input_dim)
        # h3 = torch.zeros(batch_size, self.input_dim)
        h4 = torch.zeros(batch_size, self.input_dim)

        for n in range(1, node_order.max() + 1):
            edge_mask = edge_order == n
            height = node_order.max() + 1 - n
            # h1[edge_mask, :] = op_feat[edge_mask, :] + self.pe[height,:]
            h2[edge_mask, :] = tb_feat[edge_mask, :] + self.pe[height,:]
            # h3[edge_mask, :] = ft_feat[edge_mask, :] + self.pe[height,:]
            h4[edge_mask, :] = join_feat[edge_mask, :] + self.pe[height,:]

        # h1 = h1.unsqueeze(0)
        # h1 = h1.permute(1, 0, 2) 
        # h1 = self.transformer_encoder_1(h1) 
        # h1 = h1.permute(1, 0, 2)
        # h1 = h1.squeeze(0)

        # h2 = h2.unsqueeze(0)
        # h2 = h2.permute(1, 0, 2) 
        # h2 = self.transformer_encoder_1(h2) 
        # h2 = h2.permute(1, 0, 2)
        # h2 = h2.squeeze(0)

        # h3 = h3.unsqueeze(0)
        # h3 = h3.permute(1, 0, 2) 
        # h3 = self.transformer_encoder_1(h3) 
        # h3 = h3.permute(1, 0, 2)
        # h3 = h3.squeeze(0)

        # h4 = h4.unsqueeze(0)
        # h4 = h4.permute(1, 0, 2) 
        # h4 = self.transformer_encoder_1(h4) 
        # h4 = h4.permute(1, 0, 2)
        # h4 = h4.squeeze(0)

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
    
    
# class EncoderLayer(nn.Module):
#     def __init__(self, hidden_size, ffn_size, dropout_rate, attention_dropout_rate, head_size):
#         super(EncoderLayer, self).__init__()

#         self.self_attention_norm = nn.LayerNorm(hidden_size)
#         self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
#         self.self_attention_dropout = nn.Dropout(dropout_rate)

#         self.ffn_norm = nn.LayerNorm(hidden_size)
#         self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
#         self.ffn_dropout = nn.Dropout(dropout_rate)

#     def forward(self, x, attn_bias=None):
#         y = self.self_attention_norm(x)
#         y = self.self_attention(y, y, y, attn_bias)
#         y = self.self_attention_dropout(y)
#         x = x + y

#         y = self.ffn_norm(x)
#         y = self.ffn(y)
#         y = self.ffn_dropout(y)
#         x = x + y
#         return x
    
# class MultiHeadAttention(nn.Module):
#     def __init__(self, hidden_size, attention_dropout_rate, head_size):
#         super(MultiHeadAttention, self).__init__()

#         self.head_size = head_size

#         self.att_size = att_size = hidden_size // head_size
#         self.scale = att_size ** -0.5

#         self.linear_q = nn.Linear(hidden_size, head_size * att_size)
#         self.linear_k = nn.Linear(hidden_size, head_size * att_size)
#         self.linear_v = nn.Linear(hidden_size, head_size * att_size)
#         self.att_dropout = nn.Dropout(attention_dropout_rate)

#         self.output_layer = nn.Linear(head_size * att_size, hidden_size)

#     def forward(self, q, k, v, attn_bias=None):
#         orig_q_size = q.size()

#         d_k = self.att_size
#         d_v = self.att_size
#         batch_size = q.size(0)

#         # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
#         q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)
#         k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)
#         v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)

#         q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
#         v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
#         k = k.transpose(1, 2).transpose(2, 3)  # [b, h, d_k, k_len]

#         # Scaled Dot-Product Attention.
#         # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
#         q = q * self.scale
#         x = torch.matmul(q, k)  # [b, h, q_len, k_len]
#         if attn_bias is not None:
#             x = x + attn_bias

#         x = torch.softmax(x, dim=3)
#         x = self.att_dropout(x)
#         x = x.matmul(v)  # [b, h, q_len, attn]

#         x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn]
#         x = x.view(batch_size, -1, self.head_size * d_v)

#         x = self.output_layer(x)

#         assert x.size() == orig_q_size
#         return x
    
# class FeedForwardNetwork(nn.Module):
#     def __init__(self, hidden_size, ffn_size, dropout_rate):
#         super(FeedForwardNetwork, self).__init__()

#         self.layer1 = nn.Linear(hidden_size, ffn_size)
#         self.gelu = nn.GELU()
#         self.layer2 = nn.Linear(ffn_size, hidden_size)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.gelu(x)
#         x = self.layer2(x)
#         return x