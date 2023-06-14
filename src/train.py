import os
import torch
from tqdm import tqdm
import torch.optim as optim
import numpy as np
import argparse
from torch.utils import data as D
from models import data_loader, regression_2d

def train_model(train_params, epoch, model_path):
    model.train()
    torch.cuda.set_device(train_params['device'])
    running_loss = []
    pbar =  tqdm(total=len(train_params['train_loader']))

    loss_fn = train_params['loss_fn']
    train_loader = train_params['train_loader']
    val_loader = train_params['val_loader']
    optimizer = train_params['optim']

    for feat2d,A,B, label, _, _, _ in train_loader:
        feat2d_batch, A_batch, B_batch, label_batch =  feat2d.cuda().unsqueeze(1), A.cuda(), B.cuda(), label.unsqueeze(1).cuda()
        out_batch = model(feat2d_batch)
        loss = loss_fn(out_batch,label_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())
        pbar.update()
        pbar.set_description(str(epoch) + '/'  + str(train_params['training_epoch']) + '  ' + str(np.mean(running_loss))[:6])

    tmp_loss = []
    with torch.no_grad():
        for feat2d,A,B, label in val_loader:
            feat2d_batch, A_batch, B_batch, label_batch =  feat2d.cuda().unsqueeze(1), A.cuda(), B.cuda(), label.unsqueeze(1).cuda()
            out_batch = model(feat2d_batch)
            loss = loss_fn(out_batch,label_batch)
            tmp_loss.append(loss.item())
    torch.cuda.empty_cache()
    pbar.set_description(str(epoch) + '/' + str(np.mean(tmp_loss))[:6])

    model_states = {"epoch" : epoch, "state_dict": model.state_dict(), "optimizer":optimizer.state_dict(), "loss":running_loss}
    torch.save(model_states, model_path)
    return np.mean(tmp_loss), model_states

if __name__ == "__main__":
    print('------------Starting Training------------' + '\n')

    parser = argparse.ArgumentParser(description='Training script for LL1')
    parser.add_argument('--Device', type=int, help='CUDA device for training', default=0)
    parser.add_argument('--lr', type=float,  help='Learning rate for the optimizer',default=1e-4)
    parser.add_argument('--BatchSize', help='Size of the minibatch' ,type=int, default=4)
    parser.add_argument('--ModelOut', help='Destination for saving the trained model' ,type=str)
    parser.add_argument('--Epoch', help='Number of training epochs', type=int, default=100)
    args = parser.parse_args()


    if args.ModelOut == None:
        print('Error: Please provide model output directory')
    elif os.path.isdir(os.path.join(args.ModelOut, '')):
        pass
    else:
        cmd = 'mkdir -p ' + os.path.join(args.ModelOut, '')
        os.system(cmd)


    #preparing the data loader
    dset = data_loader.data_loader(library='LL1', data_path='../data/')
    train_len = int(0.8 * dset.__len__())
    val_len = dset.__len__() - train_len
    train_dset, val_dset = torch.utils.data.random_split(dset, [train_len, val_len],generator=torch.Generator().manual_seed(888))
    train_loader = D.DataLoader(train_dset, batch_size = args.BatchSize, num_workers=20, shuffle=True,drop_last=True)
    val_loader = D.DataLoader(val_dset, batch_size = args.BatchSize, num_workers=20,drop_last=True)
    #create the model
    torch.cuda.set_device(args.Device)
    model = regression_2d.regression2d().cuda()
    #prepare the optimizer
    optimizer = optim.Adam(model.parameters(),lr = args.lr)
    loss_fn = torch.nn.MSELoss()

    train_param_dict = {
        'model':model,
        'optim':optimizer,
        'loss_fn':loss_fn,
        'train_loader':train_loader,
        'val_loader':val_loader,
        'device':args.Device,
        'training_epoch': args.Epoch
    }

    print('Number of Training Sequence: ' + str(len(train_dset)))
    print('Batch Size: ' + str(args.BatchSize))
    print('Learning Rate: ' +  str(args.lr))
    print('Number of Epochs: ' + str(args.Epoch))
    print('Saving trained model at: ' + args.ModelOut)


    patience = 10
    best_val_loss = np.inf
    best_epoch = 0
    best_model = None
    counter = 0
    model_path = args.ModelOut + 'Coevolve_model.pt'
    for epoch in range(1,args.Epoch + 1):
        val_loss, current_model = train_model(train_params = train_param_dict, epoch = epoch ,model_path = model_path)
        torch.save(current_model, model_path)
        if val_loss<best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            counter = 0
            best_model=current_model
        counter += 1
        if counter > patience:
            break
    print('Training stopped after' + str(best_epoch) + ' Epochs')
    print('Saving models at' + model_path)
    torch.save(best_model, model_path)
