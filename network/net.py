import torch
from torch import nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, state_dim, max_action, min_log_std=-10, max_log_std=2):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, max_action)
        self.log_std_head = nn.Linear(256, max_action)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mu = self.mu_head(x)
        mu = F.tanh(mu)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.min_log_std, self.max_log_std)
        # 检查mu和log_std是否有NaN值
        if torch.isnan(mu).any() or torch.isnan(log_std).any():
            # 保存下模型
            torch.save(self.state_dict(), "actor_with_nan.pth")
            # print("*"*50)
            # print("Warning: Detected NaN values in mu or log_std. Resetting to zeros.")
            # print("*"*50)
            # mu[torch.isnan(mu)] = 0
            # log_std[torch.isnan(log_std)] = -0.1  
        return mu, log_std


class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class Q(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Q, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.q_head = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.q_head(x)
        return x
    

class Discriminator(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, s, a):
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        x = F.sigmoid(x)
        return x


class DiscriminatorSCA(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscriminatorSCA, self).__init__()
        self.fc1 = nn.Linear(state_dim + 2 + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, 1)

    def forward(self, s, r_e, r_c, a):
        x = torch.cat((s, r_e, r_c, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        x = F.sigmoid(x)
        return x


class PredictNextState(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PredictNextState, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, state_dim)

    def forward(self, s, a):
        x = torch.cat((s, a), -1) # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.head(x)
        return x


class CNNEncoder(nn.Module):
    def __init__(self, channel):
        super(CNNEncoder, self).__init__()
        
        self.conv1 = nn.Conv2d(channel, 16, kernel_size=3, stride=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, stride=2, padding=1)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool(x)
        return x.view(x.size(0), -1)
    
    def get_output_dimensions(self, input_size=(1, 9, 80, 80)):
   
        dummy_input = torch.randn(*input_size)
        
        with torch.no_grad(): 
            output = self.forward(dummy_input)
        
        model_output_shape = output.shape[1:]
        return model_output_shape[0]