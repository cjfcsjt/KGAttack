import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor1(nn.Module):
    def __init__(self, emb_size, x_dim=50, state_dim=50, hidden_dim=50, layer_num=1):
        super(Actor1, self).__init__()
        self.rnn = nn.GRU(x_dim, state_dim, layer_num, batch_first = True)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+x_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim,1)
    
    def forward(self, x, l):
        '''
        :param x: encode history [N*L*D]; l: current state length Int
            N: batch size, L: seq length, D: embedding size, K: action set length
        :return: v: action score [N*K]
        '''
        out, h = self.rnn(x)
        h = h.permute(1,0,2) #[N*1*D]
        y = x[:,:l,:] #[N*l*D] 
        x = F.relu(self.fc1(h))
        x = x.repeat(1,y.shape[1],1)
        state_cat_action = torch.cat((x,y),dim=2)
        action_score = self.fc3(F.relu(self.fc2(state_cat_action))).squeeze(dim=2) #[N*K]
        action_prob = F.softmax(action_score, dim=1)
        
        return action_score, action_prob

class Actor2(nn.Module):
    def __init__(self, emb_size, x_dim = 50, state_dim=50, hidden_dim=50, layer_num=1):
        super(Actor2, self).__init__()
        # candi_num = 200
        # emb_size = 50
        # x_dim = 50
        # self.candi_num = candi_num
        self.rnn = nn.GRU(x_dim,state_dim,layer_num,batch_first=True)
        self.fc1 = nn.Linear(state_dim+x_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+emb_size, hidden_dim)   #hidden_dim + emb_size
        self.fc3 = nn.Linear(hidden_dim,1)
    
    def forward(self, x, y, a1):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*E],
            a1: seleted item a1_idx[N*1]
            N: batch size, L: seq length, D: embedding size, K: action set length
        :return: v: action score [N*K]
        """
        
        out, h = self.rnn(x)
        h = h.permute(1,0,2) #[N*1*D]
        test = x[torch.arange(x.shape[0]),a1,:].unsqueeze(1)
        # print(x[0,a1[0],:])
        # print(test[0])
        h = torch.cat((test,h),dim=2) #a1是index [N*1*(D+x_D)]
        x = F.relu(self.fc1(h))
        x = x.repeat(1,y.shape[1],1) # [N*K*D]
        state_cat_action = torch.cat((x,y),dim=2)
        action_score = self.fc3(F.relu(self.fc2(state_cat_action))).squeeze(dim=2) #[N*K]
        action_prob = F.softmax(action_score, dim=1)
        
        return action_score, action_prob


class Actor(nn.Module):
    def __init__(self, emb_size, x_dim = 50, state_dim=50, hidden_dim=50, layer_num=1):
        super(Actor, self).__init__()
        # candi_num = 200
        # emb_size = 50
        # x_dim = 50
        # self.candi_num = candi_num
        self.rnn = nn.GRU(x_dim,state_dim,layer_num,batch_first=True)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim+emb_size, hidden_dim)   #hidden_dim + emb_size
        self.fc3 = nn.Linear(hidden_dim,1)
    
    def forward(self, x, y):
        """
        :param x: encode history [N*L*D]; y: action embedding [N*K*D], 
            N: batch size, L: seq length, D: embedding size, K: action set length
        :return: v: action score [N*K]
        """
        
        out, h = self.rnn(x)
        h = h.permute(1,0,2) #[N*1*D]
        x = F.relu(self.fc1(h))
        x = x.repeat(1,y.shape[1],1)
        state_cat_action = torch.cat((x,y),dim=2)
        action_score = self.fc3(F.relu(self.fc2(state_cat_action))).squeeze(dim=2) #[N*K]
        action_prob = F.softmax(action_score, dim=1)
        
        return action_score, action_prob

class Critic(nn.Module):
    def __init__(self, emb_size, x_dim = 50, state_dim=50, hidden_dim=50, layer_num=1):
        super(Critic, self).__init__()
        self.rnn = nn.GRU(x_dim,state_dim,layer_num,batch_first=True)
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        
        out, h = self.rnn(x)
        h = h.permute(1,0,2) #[N*1*D]
        x = F.relu(self.fc1(h))
        #v(s)
        value = self.out(F.relu(self.fc2(x))).squeeze(dim=2) #[N*1*1]

        return value
        
        
