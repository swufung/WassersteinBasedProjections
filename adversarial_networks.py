import torch
import torch.nn as nn

class discriminator_net_Huber(nn.Module):
    def __init__(self):

        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.prelu1 = nn.PReLU()

        # state size. (ndf) x 64 x 64
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(64, 1, kernel_size=4, stride=2, padding=1, bias=True)
        self.prelu3 = nn.PReLU()

        # state size, 1 ch x (16 x 16)
        self.fc1 = nn.Linear(16*16, 4*4, bias=True)
        self.prelu4 = nn.PReLU()
        self.fc2 = nn.Linear(4*4, 1, bias=True)

    def forward(self, u):

        n_samples = u.shape[0]

        u = self.conv1(u)
        u = self.prelu1(u)

        u = self.conv2(u)
        u = self.prelu2(u)

        u = self.conv3(u)
        u = self.prelu3(u)

        u = u.view(n_samples, -1)

        u = self.fc1(u)
        u = self.prelu4(u)

        u = self.fc2(u)

        bool_ind = torch.abs(u) > 1
        u[bool_ind] = torch.abs(u[bool_ind])
        u[~bool_ind] = 0.5 * u[~bool_ind]**2

        return u
