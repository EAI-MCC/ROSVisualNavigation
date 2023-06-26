import typing
import torch
import torch.nn as nn 
import torch.nn.functional as F 

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.constant_(m.bias.data, 0.0)
        nn.init.orthogonal_(m.weight.data, gain=1.41)

class PolicyNet(nn.Module):
    def __init__(self, obsv_dim, map_size, act_dim, hidden_size=(256, 256)):
        super().__init__()
        self.act_dim = act_dim 

        self.net = [nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(inplace=True)]
        self.net.append(nn.MaxPool2d(2))
        self.net.append(nn.Conv2d(16, 32, kernel_size=3))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.MaxPool2d(2))

        self.net.append(nn.Conv2d(32, 64, kernel_size=3))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.MaxPool2d(2))

        self.net_fc = []
        self.net_fc.append(nn.Linear(18496, 256))
        self.net_fc.append(nn.ReLU(inplace=True))
        self.net_fc.append(nn.Dropout(p=0.25))
        self.net_fc.append(nn.Linear(256, 256))
        self.net_fc.append(nn.ReLU(inplace=True))
        self.net_fc.append(nn.Dropout(p=0.25))

        self.net = nn.Sequential(*self.net)
        self.net_fc = nn.Sequential(*self.net_fc)

        self.goal_fc = nn.Sequential(nn.Linear(3, 128), nn.ReLU(inplace=True))

        # map network
        self.map_net = []
        self.map_net.append(nn.Conv2d(1, 16, kernel_size=3))
        self.map_net.append(nn.ReLU(inplace=True))
        self.map_net.append(nn.MaxPool2d(2))
        self.map_net.append(nn.Conv2d(16, 32, kernel_size=3))
        self.map_net.append(nn.ReLU(inplace=True))
        self.map_net.append(nn.MaxPool2d(2))
        self.map_net.append(nn.Conv2d(32, 64, kernel_size=3))
        self.map_net.append(nn.ReLU(inplace=True))
        self.map_net.append(nn.MaxPool2d(2))

        self.map_net_fc = []
        self.map_net_fc.append(nn.Linear(18496, 256))
        self.map_net_fc.append(nn.ReLU(inplace=True))
        self.map_net_fc.append(nn.Dropout(p=0.25))
        self.map_net_fc.append(nn.Linear(256, 256))
        self.map_net_fc.append(nn.ReLU(inplace=True))
        self.map_net_fc.append(nn.Dropout(p=0.25))

        self.map_net = nn.Sequential(*self.map_net)
        self.map_net_fc = nn.Sequential(*self.map_net_fc)

        # human map network
        self.hmap_net = []
        self.hmap_net.append(nn.Conv2d(1, 16, kernel_size=3))
        self.hmap_net.append(nn.ReLU(inplace=True))
        self.hmap_net.append(nn.MaxPool2d(2))
        self.hmap_net.append(nn.Conv2d(16, 32, kernel_size=3))
        self.hmap_net.append(nn.ReLU(inplace=True))
        self.hmap_net.append(nn.MaxPool2d(2))
        self.hmap_net.append(nn.Conv2d(32, 64,kernel_size=3))
        self.hmap_net.append(nn.ReLU(inplace=True))
        self.hmap_net.append(nn.MaxPool2d(2))

        self.hmap_net_fc = []
        self.hmap_net_fc.append(nn.Linear(18496, 256))
        self.hmap_net_fc.append(nn.ReLU(inplace=True))
        self.hmap_net_fc.append(nn.Dropout(p=0.25))
        self.hmap_net_fc.append(nn.Linear(256, 256))
        self.hmap_net_fc.append(nn.ReLU(inplace=True))
        self.hmap_net_fc.append(nn.Dropout(p=0.25))

        self.hmap_net = nn.Sequential(*self.hmap_net)
        self.hmap_net_fc = nn.Sequential(*self.hmap_net_fc)

        self.act_net = nn.Sequential(nn.Linear(896, 256), nn.ReLU(inplace=True),
                                     nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.act_fc = nn.Linear(256, act_dim)

    def forward(self, x, imap, hmap, g):
        x = self.net(x)
        x = torch.flatten(x, 1)
        x = self.net_fc(x)

        g = F.normalize(self.goal_fc(g), dim=1)

        m = self.map_net(imap)
        m = torch.flatten(m, 1)
        m = self.map_net_fc(m)

        h = self.hmap_net(hmap)
        h = torch.flatten(h, 1)
        h = self.hmap_net_fc(h)

        x = torch.cat([x, m, h, g], dim=1)
        x = self.act_net(x)
        logits = self.act_fc(x)
        prob = F.softmax(logits, dim=-1)
        return prob 
    
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)
    
class VNet(nn.Module):
    def __init__(self, obsv_dim, map_size, act_dim, hidden_size=(256, 256)):
        super().__init__()
        self.net = [nn.Conv2d(3, 16, kernel_size=3), nn.ReLU(inplace=True)]
        self.net.append(nn.MaxPool2d(2))
        self.net.append(nn.Conv2d(16, 32, kernel_size=3))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.MaxPool2d(2))

        self.net.append(nn.Conv2d(32, 64, kernel_size=3))
        self.net.append(nn.ReLU(inplace=True))
        self.net.append(nn.MaxPool2d(2))

        self.net_fc = []
        self.net_fc.append(nn.Linear(18496, 256))
        self.net_fc.append(nn.ReLU(inplace=True))
        self.net_fc.append(nn.Dropout(p=0.25))
        self.net_fc.append(nn.Linear(256, 256))
        self.net_fc.append(nn.ReLU(inplace=True))
        self.net_fc.append(nn.Dropout(p=0.25))

        self.net = nn.Sequential(*self.net)
        self.net_fc = nn.Sequential(*self.net_fc)

        self.goal_fc = nn.Sequential(nn.Linear(3, 128), nn.ReLU(inplace=True))

        # map network
        self.map_net = []
        self.map_net.append(nn.Conv2d(1, 16, kernel_size=3))
        self.map_net.append(nn.ReLU(inplace=True))
        self.map_net.append(nn.MaxPool2d(2))
        self.map_net.append(nn.Conv2d(16, 32, kernel_size=3))
        self.map_net.append(nn.ReLU(inplace=True))
        self.map_net.append(nn.MaxPool2d(2))
        self.map_net.append(nn.Conv2d(32, 64, kernel_size=3))
        self.map_net.append(nn.ReLU(inplace=True))
        self.map_net.append(nn.MaxPool2d(2))

        self.map_net_fc = []
        self.map_net_fc.append(nn.Linear(18496, 256))
        self.map_net_fc.append(nn.ReLU(inplace=True))
        self.map_net_fc.append(nn.Dropout(p=0.25))
        self.map_net_fc.append(nn.Linear(256, 256))
        self.map_net_fc.append(nn.ReLU(inplace=True))
        self.map_net_fc.append(nn.Dropout2d(p=0.25))

        self.map_net = nn.Sequential(*self.map_net)
        self.map_net_fc = nn.Sequential(*self.map_net_fc)

        # human map network
        self.hmap_net = []
        self.hmap_net.append(nn.Conv2d(1, 16, kernel_size=3))
        self.hmap_net.append(nn.ReLU(inplace=True))
        self.hmap_net.append(nn.MaxPool2d(2))
        self.hmap_net.append(nn.Conv2d(16, 32, kernel_size=3))
        self.hmap_net.append(nn.ReLU(inplace=True))
        self.hmap_net.append(nn.MaxPool2d(2))
        self.hmap_net.append(nn.Conv2d(32, 64, kernel_size=3))
        self.hmap_net.append(nn.ReLU(inplace=True))
        self.hmap_net.append(nn.MaxPool2d(2))

        self.hmap_net_fc = []
        self.hmap_net_fc.append(nn.Linear(18496, 256))
        self.hmap_net_fc.append(nn.ReLU(inplace=True))
        self.hmap_net_fc.append(nn.Dropout(p=0.25))
        self.hmap_net_fc.append(nn.Linear(256, 256))
        self.hmap_net_fc.append(nn.ReLU(inplace=True))
        self.hmap_net_fc.append(nn.Dropout(p=0.25))

        self.hmap_net = nn.Sequential(*self.hmap_net)
        self.hmap_net_fc = nn.Sequential(*self.hmap_net_fc)

        self.act_net = nn.Sequential(nn.Linear(896, 256), nn.ReLU(inplace=True),
                                     nn.Linear(256, 256), nn.ReLU(inplace=True))
        self.val_fc = nn.Linear(256, 1)

    def forward(self, obs, imap, hmap, g):
        x = self.net(obs)
        x = torch.flatten(x, 1)
        x = self.net_fc(x)

        g = F.normalize(self.goal_fc(g))

        m = self.map_net(imap)
        m = torch.flatten(m, 1)
        m = self.map_net_fc(m)

        h = self.hmap_net(hmap)
        h = torch.flatten(h, 1)
        h = self.hmap_net_fc(h)

        x = torch.cat([x, m, h, g], dim=1)

        x = self.act_net(x)

        qs = self.val_fc(x).sequeeze(-1)

        return qs 
    
    def __call__(self, *input, **kwargs) -> typing.Any:
        return super().__call__(*input, **kwargs)
    
