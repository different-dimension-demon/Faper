import numpy as np
import os, time, argparse
from tqdm import tqdm
import torch
import time
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
from torch.autograd import Variable as Var
from faper.utils import *


class Trainer(object):
    def __init__(self, cuda, model, optimizer, min_label, max_label):
        super(Trainer, self).__init__()
        self.cuda_use = cuda
        self.model = model
        self.optimizer = optimizer
        self.min_label = min_label
        self.max_label = max_label
        self.norm_min_label = np.log(min_label)
        self.norm_max_label = np.log(max_label)
        self.epoch = 0
        self.operat_dim = 4
        self.table_dim = 11
        self.filter_dim = 18
        self.join_dim = 37

    def train(self, dataset, batch_size):
        self.model.train()
        total_loss = 0.0
        indices = torch.randperm(len(dataset)//batch_size)
        self.optimizer.zero_grad()

        for idx in tqdm(range(len(dataset)//batch_size), desc='Training epoch ' + str(self.epoch + 1) + ''):
            batch_dic=()
            for batch_idx in range(0, batch_size):
                feature, adjacency_list, node_order, edge_order, label = dataset[indices[idx]*batch_size + batch_idx]
                feature = np.array(feature)
                operation_feat = feature[:, 0:self.operat_dim]
                table_feat = feature[:, self.operat_dim : self.operat_dim + self.table_dim ]
                filter_feat = feature[:, self.operat_dim + self.table_dim : self.operat_dim + self.table_dim + self.filter_dim]
                join_feat = feature[:, self.operat_dim + self.table_dim + self.filter_dim : self.operat_dim + self.table_dim + self.filter_dim+self.join_dim]

                operation_feat = torch.from_numpy(operation_feat)
                table_feat = torch.from_numpy(table_feat)
                filter_feat = torch.from_numpy(filter_feat)
                join_feat = torch.from_numpy(join_feat)
                adjacency_list = torch.from_numpy(np.array(adjacency_list))
                node_order = torch.from_numpy(np.array(node_order))
                edge_order = torch.from_numpy(np.array(edge_order))

                pos_order = node_order+1
                pos_order[:,1:] = edge_order

                label = torch.from_numpy(np.array(label))

                operation_feat = operation_feat.float()
                table_feat = table_feat.float()
                filter_feat = filter_feat.float()
                join_feat = join_feat.float()
                label = label.float()
                node_order = torch.squeeze(node_order, 0)
                edge_order = torch.squeeze(edge_order, 0)
                pos_order = torch.squeeze(pos_order, 0)

                data ={'operation_feat': operation_feat,
                       'table_feat': table_feat,
                       'filter_feat': filter_feat,
                       'join_feat': join_feat,
                       'node_order': node_order,
                       'adjacency_list': adjacency_list,
                       'edge_order': edge_order,
                       'pos_order': pos_order,
                       'labels': label}
                batch_dic = batch_dic + (data,)

            dic = batch_tree_input_comp(batch_dic)


            if self.cuda_use:
                dic['operation_feat'] = dic['operation_feat'].cuda()
                dic['table_feat'] = dic['table_feat'].cuda()
                dic['join_feat'] = dic['join_feat'].cuda()
                dic['filter_feat'] = dic['filter_feat'].cuda()
                dic['node_order'] = dic['node_order'].cuda()
                dic['adjacency_list'] = dic['adjacency_list'].cuda()
                dic['edge_order'] = dic['edge_order'].cuda()
                dic['labels'] = dic['labels'].cuda()
                dic['pos_order'] = dic['pos_order'].cuda()

            # print(dic)
            estimate_output = self.model(
                dic['operation_feat'],
                dic['table_feat'],
                dic['filter_feat'],
                dic['join_feat'],
                dic['node_order'],
                dic['adjacency_list'],
                dic['edge_order'],
                dic['pos_order']
            )
            loss = self.loss_function(estimate_output, dic['labels'])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            total_loss += loss

        self.epoch += 1
        epoch_loss = total_loss/(len(dataset)//batch_size)
        print("     Train Epoch {}, loss: {}".format(self.epoch, epoch_loss))
        self.model.use_time = 0.0
        return epoch_loss
    
    def loss_function(self, predict, label):
        qerror = []
        predict = self.unnormalize_torch(predict)

        for i in range(len(label)):
            if (predict[i] > label[i]):
                qerror.append((predict[i] / label[i])**2)
            else:
                qerror.append((label[i] / predict[i])**2)

        return torch.mean(torch.cat(qerror))

    def unnormalize_torch(self, vals):
        vals = (vals * (self.norm_max_label - self.norm_min_label)) + self.norm_min_label
        return torch.exp(vals)

    def test(self, dataset, itr):
        global estimate_output
        self.model.train()

        with torch.no_grad():
            use_time = 0.0
            total_final_estimate_output = []
            total_final_label = []
            for idx in tqdm(range(0, len(dataset))):
                feature, adjacency_list, node_order, edge_order, label = dataset[idx]
                feature = np.array(feature)
                operation_feat = feature[:, 0:self.operat_dim]
                table_feat = feature[:, self.operat_dim : self.operat_dim + self.table_dim ]
                filter_feat = feature[:, self.operat_dim + self.table_dim : self.operat_dim + self.table_dim + self.filter_dim]
                join_feat = feature[:, self.operat_dim + self.table_dim + self.filter_dim : self.operat_dim + self.table_dim + self.filter_dim+self.join_dim]

                operation_feat = torch.from_numpy(operation_feat)
                table_feat = torch.from_numpy(table_feat)
                filter_feat = torch.from_numpy(filter_feat)
                join_feat = torch.from_numpy(join_feat)
                adjacency_list = torch.from_numpy(np.array(adjacency_list))
                node_order = torch.from_numpy(np.array(node_order))
                edge_order = torch.from_numpy(np.array(edge_order))

                pos_order = node_order+1
                pos_order[:,1:] = edge_order

                label = torch.from_numpy(np.array(label))

                operation_feat = operation_feat.float()
                table_feat = table_feat.float()
                filter_feat = filter_feat.float()
                join_feat = join_feat.float()
                label = label.float()
                node_order = torch.squeeze(node_order, 0)
                edge_order = torch.squeeze(edge_order, 0)
                pos_order = torch.squeeze(pos_order, 0)

                data ={'operation_feat': operation_feat,
                    'table_feat': table_feat,
                    'filter_feat': filter_feat,
                    'join_feat': join_feat,
                    'node_order': node_order,
                    'adjacency_list': adjacency_list,
                    'edge_order': edge_order,
                    'labels': label,
                    'pos_order': pos_order}

                if self.cuda_use:
                    data['operation_feat'] = data['operation_feat'].cuda()
                    data['table_feat'] = data['table_feat'].cuda()
                    data['join_feat'] = data['join_feat'].cuda()
                    data['filter_feat'] = data['filter_feat'].cuda()
                    data['node_order'] = data['node_order'].cuda()
                    data['adjacency_list'] = data['adjacency_list'].cuda()
                    data['edge_order'] = data['edge_order'].cuda()
                    data['labels'] = data['labels'].cuda()
                    data['pos_order'] = data['pos_order'].cuda()

                time_start = time.time()

                num_runs = itr
                predictions = []

                for _ in range(num_runs):
                    estimate_output = self.model(
                        data['operation_feat'],
                        data['table_feat'],
                        data['filter_feat'],
                        data['join_feat'],
                        data['node_order'],
                        data['adjacency_list'],
                        data['edge_order'],
                        data['pos_order']
                    )
                    
                    predictions.append(self.unnormalize_torch(estimate_output).detach().numpy())

                predictions = np.array(predictions)
                mean = np.mean(predictions, axis=0)
                variance = np.var(predictions/mean, axis=0)

                use_time += time.time() - time_start

                self.node_qerror(mean, data['labels'].detach().numpy(), variance)
                # print(use_time * 1000.0)

                total_final_estimate_output.append(mean[0])
                total_final_label.append(data['labels'][0])
                
            print("     Testing take time in total: {} ms".format(use_time * 1000.0))
            qerror = self.print_qerror(total_final_estimate_output, total_final_label)
            return qerror
    
    def node_qerror(self, predict, label, variance):
        qerror = []
        for i in range(len(predict)):
            if (predict[i] > label[i]):
                qerror.append(predict[i] / label[i])
            else:
                qerror.append(label[i] / predict[i])

    def print_qerror(self, predict, label):
        qerror = []
        for i in range(len(predict)):
            if predict[i] > float(label[i]):
                qerror.append(float(predict[i]) / float(label[i]))
            else:
                qerror.append(float(label[i]) / float(predict[i]))

        print("     =========Q-error Accuracy=========")
        print("     50th percentile: {}".format(np.percentile(qerror, 50)))
        print("     75th percentile: {}".format(np.percentile(qerror, 75)))
        print("     90th percentile: {}".format(np.percentile(qerror, 90)))
        print("     99th percentile: {}".format(np.percentile(qerror, 99)))
        print("     Max: {}".format(np.max(qerror)))
        print("     Mean: {}".format(np.mean(qerror)))
        return qerror
