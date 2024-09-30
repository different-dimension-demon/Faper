import torch
import torch.optim as optim
from faper.loaddata import Dataloader
from faper.spiking_t_sru import *
from faper.trainer import *
from faper import utils


if __name__ == '__main__':
    # mode = 'Test'
    mode = 'Train'

    epoch_num = 100
    embed_dim = 256
    mem_dim = 1280 #
    output_dim = 1024 #
    batch_size = 50 #
    lr_set = 0.0001 #
    cuda_use = False
    # GPU_no = 0
    wd_set = 0.00001
    feature_dim = 70
    train_path = './workload/IMDB'
    train_size = 11000
    test_size = 520

    # torch.cuda.set_device(GPU_no)
    max_label, min_label = utils.get_max_min_label('./workload/IMDB')
    
    print("card domain: ", str(min_label)+ "(" + str(np.log(min_label))+ ")  " + str(max_label) + "(" + str(np.log(max_label)) + ")")
    dataset_size = train_size
    train_select, vaild_select = utils.random_split(dataset_size)

    if mode == 'Train':
        train_dataset = Dataloader(train_path, train_select, True)
        vaild_dataset = Dataloader(train_path, vaild_select, True)
    else:
        test_dataset = Dataloader(train_path, vaild_select, True)

    if mode == 'Train':
        model = Faper(cuda_use, feature_dim, embed_dim, mem_dim, output_dim)
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_set, weight_decay=wd_set)
        trainer = Trainer(cuda_use, model, optimizer, min_label, max_label)
        for epoch in range(epoch_num):
            loss = trainer.train(train_dataset, batch_size)
            print("Evaluation:")
            # if epoch%10 == 9:
            trainer.test(vaild_dataset, 10)
        torch.save(model.state_dict(), './model/Faper'+'.pth')

    if mode == 'Test':
        model = Faper(cuda_use, feature_dim, embed_dim, mem_dim, output_dim)
        model.load_state_dict(torch.load('./model/example.pth'))
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr_set, weight_decay=wd_set)
        trainer = Trainer(cuda_use, model, optimizer, min_label, max_label)
        qerror = trainer.test(test_dataset, 10)
        
