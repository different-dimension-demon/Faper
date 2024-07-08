import torch
import torch.optim as optim
from spiking_t_sru.loaddata import Dataloader
from spiking_t_sru.spiking_t_sru import *
from spiking_t_sru.trainer import *
from spiking_t_sru import utils


if __name__ == '__main__':
    # mode = 'Test_SRU'
    mode = 'Test_Spiking_T_SRU'
    # mode = 'Train'
    # mode = 'Train_nnpg'
    epoch_num = 300

    # embed_dim is the hidden unit of embed module
    # mem_dim is the hidden unit of SRU module
    # output_dim is the hidden unit of output module
    embed_dim = 256
    mem_dim = 1024 #
    output_dim = 1024 #
    batch_size = 50 #
    lr_set = 0.0001 #
    cuda_use = False
    GPU_no = 1
    wd_set = 0.00001
    feature_dim = 70
    # train_path = './data/train_10K'
    # train_path = './data/train_8K'
    # train_path = './data/train_6K'
    # train_path = './data/train_4K'
    train_path = './data/train_8_join'
    # train_path = './data/train_6_join'
    train_size = 11000
    test_size = 520

    # torch.cuda.set_device(GPU_no)
    # max_label, min_label = utils.get_max_min_label('./data/train_10K')
    # max_label, min_label = utils.get_max_min_label('./data/train_8K')
    # max_label, min_label = utils.get_max_min_label('./data/train_6K')
    # max_label, min_label = utils.get_max_min_label('./data/train_4K')
    # max_label, min_label = utils.get_max_min_label('./data/train_8_join')
    max_label, min_label = utils.get_max_min_label('./data/train_6_join')
    
    print("card domain: ", str(min_label)+ "(" + str(np.log(min_label))+ ")  " + str(max_label) + "(" + str(np.log(max_label)) + ")")
    # prepare training set
    dataset_size = train_size
    train_select, vaild_select = utils.random_split(dataset_size)

    # if mode == 'Train' or mode == 'Train_nnpg':
    #     train_dataset = Dataloader(train_path, train_select, True)
    #     vaild_dataset = Dataloader(train_path, vaild_select, True)
    # else:
    #     test_dataset = Dataloader(train_path, vaild_select, True)

    if mode == 'Train':
        train_dataset = Dataloader(train_path, train_select, True)
        vaild_dataset = Dataloader(train_path, vaild_select, True)
    else:
        test_dataset = Dataloader(train_path, vaild_select, True)
    
    # if mode == 'Train_nnpg':
    #     model = Spiking_T_SRU(cuda_use, feature_dim, embed_dim, mem_dim, output_dim)
    #     model.load_state_dict(torch.load('./model/Spiking_T_SRU.pth', map_location = torch.device('cpu')))
    #     optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_set, weight_decay=wd_set)
    #     trainer = Trainer(cuda_use, model, optimizer, min_label, max_label)
    #     trainer.test(vaild_dataset)

    if mode == 'Train':
        model = Spiking_T_SRU(cuda_use, feature_dim, embed_dim, mem_dim, output_dim)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_set, weight_decay=wd_set)
        trainer = Trainer(cuda_use, model, optimizer, min_label, max_label)
        for epoch in range(epoch_num):
            loss = trainer.train(train_dataset, batch_size)

            # 通过两阶段训练过程实现多头的输出模型，以下代码为生成训练数据使用（次选）
            # with open("accuracy.txt", "a") as file:
            #     # 将准确度写入文件，并添加换行符
            #     file.write(str(loss.item())+" ")

            print("Evaluation:")
            trainer.test(vaild_dataset)
        torch.save(model.state_dict(), './model/Spiking_T_SRU_2_joinsfsfdsf'+'.pth')

    if mode == 'Test_Spiking_T_SRU':
        model = Spiking_T_SRU(cuda_use, feature_dim, embed_dim, mem_dim, output_dim)
        model.load_state_dict(torch.load('./model/Spiking_T_SRU_4_joins.pth'))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_set, weight_decay=wd_set)
        trainer = Trainer(cuda_use, model, optimizer, min_label, max_label)
        len = 10
        for _ in range(len):
            qerror = trainer.test(test_dataset)

    if mode == 'Test_SRU':
        model = SRU(cuda_use, feature_dim, embed_dim, mem_dim, output_dim)
        model.load_state_dict(torch.load('./model/example.pth'))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_set, weight_decay=wd_set)
        trainer = Trainer(cuda_use, model, optimizer, min_label, max_label)
        qerror = trainer.test(test_dataset)



