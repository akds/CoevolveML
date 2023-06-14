import numpy as np
import torch
from torch.utils import data as D

class data_loader(D.Dataset):
    def __init__(self, data_path, library = 'LL1'):
        self.data = torch.load(data_path + library + '_processed.pt')
        self.data_list = []

        for i in self.data:
            self.data_list.append(self.data[i])

    def __getitem__(self, index):
        A_embedding = torch.from_numpy(self.data_list[index][2]).float()
        B_embedding = torch.from_numpy(self.data_list[index][3]).float()
        label = torch.from_numpy(np.array(self.data_list[index][-1])).float()
        feat_2d = torch.outer(A_embedding, B_embedding)
        return feat_2d, A_embedding, B_embedding, label, self.data_list[index][0], self.data_list[index][1], self.data_list[index][-2]

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    Dset=data_loader(data_path='../../data/', library='LL1')
    for i in Dset:
        #print(i[0].size())
        #print(i[1].size())
        #print(i[2].size())
        print(i[3])
        break
        
