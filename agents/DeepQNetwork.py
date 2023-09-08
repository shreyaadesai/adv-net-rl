# checkout https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DDQN
# The only changes I made were regarding the network architecture (not CNN here)

import os
import torch as T
import torch.cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir):
        super(DeepQNetwork, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        # you may want to play around with this and forward()
        self.fc1 = nn.Linear(input_dims[0], 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # you may want to play around with this
    def forward(self, state):
        flat1 = F.relu(self.fc1(state))
        flat2 = F.relu(self.fc2(flat1))
        actions = self.fc3(flat2)
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file, map_location=torch.device('cpu')))


class DeepRNNNetwork(DeepQNetwork):
    def __init__(self, lr, n_actions, name, input_dims, chkpt_dir, hid_size=64):
        super(DeepRNNNetwork, self).__init__(lr, n_actions, name, input_dims, chkpt_dir)

        self.n_layers = 2
        self.hidden_dim = hid_size
        self.gru = nn.GRU(input_dims[0], hidden_size=self.hidden_dim, num_layers=2, batch_first=True, device=device)
        self.fc3 = nn.Linear(self.hidden_dim, n_actions, device=device)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, state, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(state.shape[0])
        out, h1 = self.gru(state, hidden)
        actions = self.fc3(F.relu(out[:, -1]))
        return actions

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden

