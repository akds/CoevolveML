import torch
import torch.nn as nn

class regression2d(torch.nn.Module):
    def __init__(self):
        super(regression2d, self).__init__()
        self.conv_kernel = (5,5)
        self.pool_kernel = (5,5)

        self.H1 = 64
        self.H2 = 128
        self.H3 = 256
        self.input_channels = 1

        self.conv_net = nn.Sequential(
            nn.Conv2d(self.input_channels, self.H1, kernel_size = self.conv_kernel),
            nn.MaxPool2d(kernel_size = self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.H1),

            nn.Conv2d(self.H1, self.H2, kernel_size = self.conv_kernel),
            nn.MaxPool2d(kernel_size = self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.H2),

            nn.Conv2d(self.H2, self.H3, kernel_size = self.conv_kernel),
            nn.MaxPool2d(kernel_size = self.pool_kernel),
            nn.ReLU(),
            nn.InstanceNorm2d(self.H3)
            )
        self.classifier = nn.Sequential(
            nn.Linear(9 * 9 * 256, 256),
            nn.ReLU(inplace = True),
            nn.Linear(256,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        out = self.conv_net(x)
        out = out.view(out.size(0), out.size(-1) * out.size(-2) * self.H3)
        out = self.classifier(out)
        return out

if __name__ == '__main__':
    import data_loader
    model = regression2d()
    Dset = data_loader.data_loader(CV_set = 1, partition = 'train')
    for A, B, label in Dset:
        infeat = torch.outer(A,B).unsqueeze(0)
        out = model(infeat.unsqueeze(0))
        print(out.size())
        #print(infeat.size())
        #break
        #out = model()

