import scipy.signal
import torch

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, -discount], x[::-1], axis=0)[::-1]

class Buffer():
    def __init__(self, obs_dim, map_size, act_dim, size, num_envs, gamma=0.99, lam=0.95, device='cpu'):
        self.obs_buf = torch.zeros((size, obs_dim[0], obs_dim[1], obs_dim[2]),
                                   dtype=torch.float32, device=device)
        self.map_buf = torch.zeros((size, 1, map_size, map_size), dtype=torch.float32,
                                   device=device)
        self.hmap_buf = torch.zeros((size, 1, map_size, map_size), dtype=torch.float32, device=device)
        self.act_buf = torch.zeros((size,), dtype=torch.int64, device=device)
        self.logp_buf = torch.zeros((size,), dtype=torch.float32, device=device)
        self.rew_buf = torch.zeros((size,), dtype=torch.float32, device=device)
        self.remain_buf = torch.zeros((size,), dtype=torch.float32, device=device)
        self.goal_buf = torch.zeros((size, 3), dtype=torch.float32, device=device)

        self.val_buf = torch.zeros((size+1,), dtype=torch.float32, device=device)
        self.adv_buf = torch.zeros((size,), dtype=torch.float32, device=device)
        self.ret_buf = torch.zeros((size,), dtype=torch.float32, device=device)

        self.gamma = gamma
        self.lam = lam 
        self.device = torch.device(device)
        self.start_idx,self.ptr, self.max_size = 0, 0, size 

        self.default_last_val = torch.zeros((1,), dtype=torch.float32, device=device)

    def add(self, obs, imap, hmap, act, rew, done, val, logp, goal):
        self.obs_buf[self.ptr] = obs
        self.map_buf[self.ptr] = imap 
        self.hmap_buf[self.ptr] = hmap
        self.act_buf[self.ptr] = act.squeeze(1)
        self.rew_buf[self.ptr] = rew 
        self.remain_buf[self.ptr] = 1 - done.float()

        self.val_buf[self.ptr] = val 
        self.logp_buf[self.ptr] = logp 
        self.goal_buf[self.ptr] = goal 
        self.ptr += 1

    def get(self):
        if self.ptr == self.max_size:
            self.start_idx, self.ptr = 0, 0
        
        advs = (self.adv_buf - self.adv_buf.mean()) / self.adv_buf.std()

        return self.obs_buf, self.map_buf, self.hmap_buf, \
        self.act_buf, self.logp_buf, advs, self.ret_buf, self.val_buf, self.goal_buf
    
    def clear(self):
        self.start_idx, self.ptr = 0, 0

    def finish_path_0(self, last_val=0):
        if last_val == 0:
            lost_val = self.default_last_val

        path_slice = slice(self.start_idx, self.ptr)
        rews = torch.cat((self.rew_buf[path_slice], last_val), dim=0)
        vals = torch.cat((self.val_buf[path_slice], last_val), dim=0)
        deltas = rews[:-1] + self.gamma*vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = torch.as_tensor(discount_cumsum(deltas.cpu().numpy(),
                                                                   self.gamma*self.lam).copy(),
                                                                   dtype=torch.float32, device=self.device)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        self.start_idx = self.ptr 
    
    def finish_buffer(self, last_val):
        self.val_buf[self.ptr] = last_val 
        deltas = self.rew_buf + self.gamma*self.remain_buf*self.val_buf[1:] - self.val_buf[:-1]
        return_i = last_val 

        adv_i = 0
        for i in reversed(range(self.max_size)):
            return_i = self.rew_buf[i] + self.gamma * self.remain_buf[i]*return_i 
            self.ret_buf[i] = return_i 

            adv_i = deltas[i] + self.gamma*self.lam*self.remain_buf[i]*adv_i
            self.adv_buf[i] = adv_i 
    