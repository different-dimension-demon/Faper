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
from spiking_t_sru.utils import *


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

        #len(dataset)//batch_size
        # #training one epoch
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
                label = torch.from_numpy(np.array(label))

                operation_feat = operation_feat.float()
                table_feat = table_feat.float()
                filter_feat = filter_feat.float()
                join_feat = join_feat.float()
                label = label.float()
                node_order = torch.squeeze(node_order, 0)
                edge_order = torch.squeeze(edge_order, 0)

                data ={'operation_feat': operation_feat,
                       'table_feat': table_feat,
                       'filter_feat': filter_feat,
                       'join_feat': join_feat,
                       'node_order': node_order,
                       'adjacency_list': adjacency_list,
                       'edge_order': edge_order,
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

            estimate_output = self.model(
                dic['operation_feat'],
                dic['table_feat'],
                dic['filter_feat'],
                dic['join_feat'],
                dic['node_order'],
                dic['adjacency_list'],
                dic['edge_order']
            )
            # 4
            # estimate_output_4 = []
            # estimate_label_4 = []
            # for n in range(0, len(estimate_output)):
            #     if (1 < n % 13 < 11):
            #         estimate_output_4.append(estimate_output[n])
            #         estimate_label_4.append(dic['labels'][n])

            # 6
            # estimate_output_6 = []
            # estimate_label_6 = []
            # for n in range(0, len(estimate_output)):
            #     if (3 < n % 13 < 9):
            #         estimate_output_6.append(estimate_output[n])
            #         estimate_label_6.append(dic['labels'][n])

            
            # print(estimate_output.shape)
            # loss = self.loss_function(torch.cat(estimate_output_4), estimate_label_4)
            # loss = self.loss_function(torch.cat(estimate_output_6), estimate_label_6)
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
                qerror.append(predict[i] / label[i])
            else:
                qerror.append(label[i] / predict[i])

        return torch.mean(torch.cat(qerror))


    def unnormalize_torch(self, vals):
        vals = (vals * (self.norm_max_label - self.norm_min_label)) + self.norm_min_label
        return torch.exp(vals)

    def test(self, dataset, itr):
        global estimate_output
        self.model.train()
        # self.model.eval()

        with torch.no_grad():

            # final_error = 0.0
            use_time = 0.0
            total_final_estimate_output = []
            
            total_final_label = []
            reliable_estimate_output1 = []
            reliable_label1 = []
            reliable_estimate_output2 = []
            reliable_label2 = []
            reliable_estimate_output3 = []
            reliable_label3 = []
            reliable_estimate_output4 = []
            reliable_label4 = []
            reliable_estimate_output5 = []
            reliable_label5 = []
            reliable_estimate_output6 = []
            reliable_label6 = []
            reliable_estimate_output7 = []
            reliable_label7 = []
            # print(len(dataset))
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
                label = torch.from_numpy(np.array(label))

                operation_feat = operation_feat.float()
                table_feat = table_feat.float()
                filter_feat = filter_feat.float()
                join_feat = join_feat.float()
                label = label.float()
                node_order = torch.squeeze(node_order, 0)
                edge_order = torch.squeeze(edge_order, 0)

                data ={'operation_feat': operation_feat,
                    'table_feat': table_feat,
                    'filter_feat': filter_feat,
                    'join_feat': join_feat,
                    'node_order': node_order,
                    'adjacency_list': adjacency_list,
                    'edge_order': edge_order,
                    'labels': label}

                if self.cuda_use:
                    data['operation_feat'] = data['operation_feat'].cuda()
                    data['table_feat'] = data['table_feat'].cuda()
                    data['join_feat'] = data['join_feat'].cuda()
                    data['filter_feat'] = data['filter_feat'].cuda()
                    data['node_order'] = data['node_order'].cuda()
                    data['adjacency_list'] = data['adjacency_list'].cuda()
                    data['edge_order'] = data['edge_order'].cuda()
                    data['labels'] = data['labels'].cuda()

                time_start = time.time()

                # nnpg 度量的关键
                # 运行模型多次并收集输出结果
                num_runs = itr  # 设置运行次数
                predictions = []

                for _ in range(num_runs):
                    estimate_output = self.model(
                        data['operation_feat'],
                        data['table_feat'],
                        data['filter_feat'],
                        data['join_feat'],
                        data['node_order'],
                        data['adjacency_list'],
                        data['edge_order']
                    )
                    
                    predictions.append(self.unnormalize_torch(estimate_output).detach().numpy())

                predictions = np.array(predictions)
                mean = np.mean(predictions, axis=0)
                variance = np.var(predictions/mean, axis=0)

                use_time += time.time() - time_start

                # 通过两阶段训练过程实现多头的输出模型，以下代码为生成训练数据使用（次选）
                # train_result = data['labels'].detach().numpy()
                # with open("test_result.txt", "a") as file:
                #     line = ' '.join(str(x) for x in train_result[0]) + '\n'
                #     file.write(line)

                self.node_qerror(mean, data['labels'].detach().numpy(), variance)
                # print(use_time * 1000.0)

                total_final_estimate_output.append(mean[0])
                total_final_label.append(data['labels'][0])
                
                if(variance[0] <= 1):
                    reliable_estimate_output1.append(mean[0])
                    reliable_label1.append(data['labels'][0])
                if(variance[0] <= 0.8):
                    reliable_estimate_output2.append(mean[0])
                    reliable_label2.append(data['labels'][0])
                if(variance[0] <= 0.7):
                    reliable_estimate_output3.append(mean[0])
                    reliable_label3.append(data['labels'][0])
                if(variance[0] <= 0.5):
                    reliable_estimate_output4.append(mean[0])
                    reliable_label4.append(data['labels'][0])
                if(variance[0] <= 0.3):
                    reliable_estimate_output5.append(mean[0])
                    reliable_label5.append(data['labels'][0])
                if(variance[0] <= 0.2):
                    reliable_estimate_output6.append(mean[0])
                    reliable_label6.append(data['labels'][0])
                if(variance[0] <= 0.1):
                    reliable_estimate_output7.append(mean[0])
                    reliable_label7.append(data['labels'][0])
                

            # test_loss = final_error/(len(dataset))
            # print("     Testing q-error for final card: {}".format(test_loss))
            print("     Testing take time in total: {} ms".format(use_time * 1000.0))
            print(len(total_final_estimate_output))
            qerror = self.print_qerror(total_final_estimate_output, total_final_label)
            print(len(reliable_estimate_output1))
            qerror = self.print_qerror(reliable_estimate_output1, reliable_label1)
            print(len(reliable_estimate_output2))
            qerror = self.print_qerror(reliable_estimate_output2, reliable_label2)
            print(len(reliable_estimate_output3))            
            qerror = self.print_qerror(reliable_estimate_output3, reliable_label3)
            print(len(reliable_estimate_output4))
            qerror = self.print_qerror(reliable_estimate_output4, reliable_label4)
            print(len(reliable_estimate_output5))
            qerror = self.print_qerror(reliable_estimate_output5, reliable_label5)
            print(len(reliable_estimate_output6))
            qerror = self.print_qerror(reliable_estimate_output6, reliable_label6)
            print(len(reliable_estimate_output7))
            qerror = self.print_qerror(reliable_estimate_output7, reliable_label7)
            return qerror
    
    

    def node_qerror(self, predict, label, variance):
        qerror = []
        # predict = self.unnormalize_torch(predict)
        for i in range(len(predict)):
            if (predict[i] > label[i]):
                qerror.append(predict[i] / label[i])
                # if(predict[i] / label[i] > 3):
                #     print("estimated mean: " + str(predict[i]) + " estimated variance: " + str(variance[i]) + " qerror: " + str(qerror[i]))
            else:
                qerror.append(label[i] / predict[i])
                
            # print("estimated mean: " + str(predict[i]) + " estimated variance: " + str(variance[i]) + " qerror: " + str(qerror[i]))


    def print_qerror(self, predict, label):
        qerror = []
        for i in range(len(predict)):
            # predict[i] = self.unnormalize_torch(predict[i])
            if predict[i] > float(label[i]):
                qerror.append(float(predict[i]) / float(label[i]))
            else:
                qerror.append(float(label[i]) / float(predict[i]))
        

        print("     =========Q-error Accuracy=========")
        print("     50th percentile: {}".format(np.percentile(qerror, 50)))
        print("     75th percentile: {}".format(np.percentile(qerror, 75)))
        print("     90th percentile: {}".format(np.percentile(qerror, 90)))
        print("     99th percentile: {}".format(np.percentile(qerror, 99)))
        # with open("accuracy.txt", "a") as file:
        #         # 将准确度写入文件，并添加换行符
        #         file.write(str(np.percentile(qerror, 50))+" ")
        #         file.write(str(np.percentile(qerror, 75))+" ")
        #         file.write(str(np.percentile(qerror, 90))+" ")
        #         file.write(str(np.percentile(qerror, 99))+" \n")
        print("     Max: {}".format(np.max(qerror)))
        print("     Mean: {}".format(np.mean(qerror)))

        return qerror

    def test_batch(self, dataset):
        self.model.eval()
        batch_size = len(dataset)
        use_time = 0.0

        batch_dic=()
        for batch_idx in range(0, batch_size):
            feature, adjacency_list, node_order, edge_order, label = dataset[batch_idx]
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
            label = torch.from_numpy(np.array(label))

            operation_feat = operation_feat.float()
            table_feat = table_feat.float()
            filter_feat = filter_feat.float()
            join_feat = join_feat.float()
            label = label.float()
            node_order = torch.squeeze(node_order, 0)
            edge_order = torch.squeeze(edge_order, 0)

            data ={'operation_feat': operation_feat,
                   'table_feat': table_feat,
                   'filter_feat': filter_feat,
                   'join_feat': join_feat,
                   'node_order': node_order,
                   'adjacency_list': adjacency_list,
                   'edge_order': edge_order,
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

        time_start = time.time()
        estimate_output = self.model(
            dic['operation_feat'],
            dic['table_feat'],
            dic['filter_feat'],
            dic['join_feat'],
            dic['node_order'],
            dic['adjacency_list'],
            dic['edge_order']
        )
        use_time += time.time() - time_start
        final_error = self.final_qerror_inbatch(estimate_output, dic['labels'], dic['tree_sizes'])

        return final_error


    def final_qerror_inbatch(self, predict, label, tree_size):
        qerror = []
        predict = self.unnormalize_torch(predict)
        final_pred = 0

        for idx in range(len(tree_size)):
            if (predict[final_pred] > label[final_pred]).cpu().data.numpy()[0]:
                qerror.append(predict[final_pred] / label[final_pred])
            else:
                qerror.append(label[final_pred] / predict[final_pred])
            final_pred += tree_size[idx]


        qerror = torch.cat(qerror)
        print("     qerror shape: ", qerror.shape)
        qerror = qerror.cpu().detach().numpy()
        print("     =========Q-error Accuracy=========")
        print("     50th percentile: {}".format(np.percentile(qerror, 50)))
        print("     75th percentile: {}".format(np.percentile(qerror, 75)))
        print("     90th percentile: {}".format(np.percentile(qerror, 90)))
        print("     99th percentile: {}".format(np.percentile(qerror, 99)))
        print("     Max: {}".format(np.max(qerror)))
        print("     Mean: {}".format(np.mean(qerror)))

        return np.mean(qerror)









