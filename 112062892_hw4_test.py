from osim.env import L2M2019Env
import matplotlib.pyplot as plt
#-------------Agent--------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque, namedtuple
import random
# Device configuration
from torch.distributions import Normal
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.priorities = np.zeros((capacity), dtype=np.float32)
    def memory_reset(self):
        self.buffer=[]
        self.priorities=np.zeros((self.capacity,),dtype=np.float32)
    def push(self, batch_obs,batch_action, next_batch_reward, batch_next_state,if_done,td_error):
        self.buffer.append((batch_obs, batch_action, next_batch_reward, batch_next_state,if_done))
        self.priorities[len(self.buffer)-1] = td_error
    def sample(self,batch_size):
        if self.priorities[:len(self.buffer)].sum()>0:
            probabilities = self.priorities[:len(self.buffer)] / self.priorities[:len(self.buffer)].sum()
        else:
            probabilities=self.priorities[:len(self.buffer)]+(1/len(self.buffer))
        selected_indexs = np.random.choice(len(self.buffer),size=batch_size,p=probabilities)
        #selected_indexs = np.random.choice(len(self.buffer),size=batch_size)
        #observations, actions, rewards, next_states = self.buffer[selected_indexs]


        observations, actions, rewards, next_states ,if_done= zip(*[self.buffer[i] for i in selected_indexs])
        #print(type(observations))

        # Since each observation is (1, 36), we concatenate them along the first axis after converting to numpy arrays
        observations = np.concatenate([obs for obs in observations], axis=0)
        actions = np.concatenate([act for act in actions], axis=0)
        rewards = np.concatenate([reward for reward in rewards], axis=0)
        next_states = np.concatenate([nxts for nxts in next_states], axis=0)
        if_done = np.concatenate([[[ifd]]for ifd in np.array(if_done)], axis=0)
        #print(if_done.shape)
        return torch.from_numpy(observations), torch.from_numpy(actions), torch.from_numpy(rewards), torch.from_numpy(next_states),torch.from_numpy(if_done),selected_indexs
    def update_td_error(self,td_errors,indexs):
        #print(td_errors.shape)
        for i, index in enumerate(indexs):
            self.priorities[index] = td_errors[i]
class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dense1 = nn.Linear(339, 700)
        self.dense2 = nn.Linear(700, 500)
        self.dense3 = nn.Linear(500, 22)
        #self.dense4=nn.Linear(256,22)
        self.log_dense3=nn.Linear(339,256)
        self.log_dense4=nn.Linear(256,22)



    def forward(self, x):
        mean=torch.relu(self.dense1(x))
        mean=self.dense2(mean)
        action=self.dense3(mean)
        action = torch.clamp(action, min=0, max=1) 
        #action=torch.relu(self.dense4(mean))



        log_std=torch.relu(self.log_dense3(x))
        log_std=torch.relu(self.log_dense4(log_std))
        log_std = torch.clamp(log_std, min=-20, max=2)  
        return action,log_std
    def sample(self, obs):

        if obs.dim()<2:
            obs.unsqueeze(0)
        obs=obs.float().to(self.device)
        action,_=self.forward(obs)
        return action
    def prop_sample(self,obs):
        if obs.dim()<2:
            obs.unsqueeze(0)
        obs=obs.float().to(self.device)
        mean,log_std=self.forward(obs)
        std = 0.01*log_std.exp()
        normal_dist = torch.distributions.Normal(mean.float(), std.float())  
        action = normal_dist.sample()
        log_pi = normal_dist.log_prob(action)
        if(log_pi.dim()<2):

            log_pi=log_pi.unsqueeze(0) 
        #print("log pi shape",log_pi.shape)

        log_pi = log_pi.mean(dim=1, keepdim=True)
        #print("log pi shape",log_pi.shape)
        return action,log_std

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dense1=nn.Linear(339,700)
        self.dense2=nn.Linear(700,500)
        self.dense3=nn.Linear(500,22)
        self.dense4=nn.Linear(22,1)

    def forward(self, obs):
        if obs.dim()<2:
            obs.unsqueeze(0)
        obs=obs.float().to(self.device)

        x=torch.relu(self.dense1(obs))
        x=self.dense2(x)
        x=self.dense3(x)
        x=self.dense4(x)
        return x

class Q_network(nn.Module):
    def __init__(self):
        super(Q_network, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dense1=nn.Linear(339,800)
        self.dense2=nn.Linear(822,500)
        self.dense3=nn.Linear(500,1)

    def forward(self, obs, x3):
        if obs.dim()<2:
            obs=obs.unsqueeze(0)
        if x3.dim()<2:
            x3=x3.unsqueeze(0)
        x3=x3.float().to(self.device)
        obs=obs.float().to(self.device)
        obs=torch.relu(self.dense1(obs))
        x=torch.cat((obs,x3),dim=1)
        #x=torch.relu(self.dense1(x))
        x=self.dense2(x)
        x=self.dense3(x)
        return x

class Agent:
    def __init__(self):
        # -----------learning parameter--------------------------------
        self.env=L2M2019Env(difficulty=2,visualize=False)
        self.stack_size=1
        self.skip=4
        self.lr=0.0005
        self.batch_size=256
        self.gamma=0.99 #target network weight
        self.alpha=0.01 #entropy weight, exploration
        self.tou=0.01 # soft update ratio
        self.total_episode=800000
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #-----------all network and optimize parameter-------------
        self.policy_network = Actor().float().to(self.device)
        self.value_network = Critic().float().to(self.device)
        self.target_network = Critic().float().to(self.device)
        self.q1_network=Q_network().float().to(self.device)
        self.q2_network=Q_network().float().to(self.device)
        self.policy_optim = optim.Adam(self.policy_network.parameters(), lr=self.lr)
        self.value_optim = optim.Adam(self.value_network.parameters(), lr=self.lr)
        self.target_optim = optim.Adam(self.target_network.parameters(), lr=self.lr)
        self.q1_optim=optim.Adam(self.q1_network.parameters(),lr=self.lr)
        self.q2_optim=optim.Adam(self.q2_network.parameters(),lr=self.lr)
        #--------------memeory------------------------------------------
        self.memory=ReplayMemory(200000)
    
    def process_observation(self,observation):
        obs=list((list(observation.values())[0]).flatten())
        for i in range(1,4):
            values= list(observation.values())[i]
            if isinstance(values,dict):
                for v in list(values.values()):
                    if isinstance(v,float):
                        obs.append(v)
                    if isinstance(v,dict):
                        for val in list(v.values()):
                            obs.append(val)
                    if isinstance(v,list):
                        for val in v:
                            obs.append(val)        
        return torch.from_numpy(np.array(obs))
    def act(self, observation):
        obs=self.process_observation(observation)

        action=self.policy_network.sample(obs)
        return action.squeeze().detach().cpu().numpy()
    def training_act(self,observation,if_random):
        if(if_random):
            return self.env.action_space.sample()
        obs=self.process_observation(observation)

        action,_=self.policy_network.prop_sample(obs)
        return action.squeeze().detach().cpu().numpy()
    def compute_td_error(self,rewards,batch_obs,batch_acts):
        rewards=rewards.to(self.device)

        with torch.no_grad():
            q1_output = self.q1_network(batch_obs, batch_acts)
            q2_output = self.q2_network(batch_obs, batch_acts)
            if (torch.sum(q1_output)<torch.sum(q2_output)):
                min_q_output=q1_output
            else:
                min_q_output=q2_output
            #print("rewards shape",rewards.shape)
            if rewards.shape== torch.Size([256]):
                rewards=rewards.unsqueeze(1)
            #print("rewards shape",rewards.shape)
            td_error=((rewards*10+self.gamma*self.target_network(batch_obs))-min_q_output)**2
            #print(td_error.shape)
        return td_error

    def update_parameter(self):
        states,actions,rewards,next_states,if_dones,select_indexs=self.memory.sample(self.batch_size)
        rewards=rewards.to(self.device)
        if_dones=if_dones.float().to(self.device)
        if rewards.shape== torch.Size([256]):
            rewards=rewards.unsqueeze(1)
        #-----------update_q1,q2_network()-----------
        with torch.no_grad():  # Ensure no gradients are computed for the target calculations
            target_values = 10*rewards + self.gamma *self.target_network(next_states)*(1-if_dones)
        #print(target_values.shape)
        q1_network_loss=(1/self.batch_size)*torch.sum((target_values-self.q1_network(states,actions)*(1-if_dones)**2))
        q1_network_loss.backward()
        self.q1_optim.step()
        q2_network_loss=(1/self.batch_size)*torch.sum((target_values-self.q2_network(states,actions)*(1-if_dones))**2)
        self.q2_optim.zero_grad()
        q2_network_loss.backward()
        self.q2_optim.step()

        #-----------update_value_network()----------
        with torch.no_grad():  # Ensure no gradients are computed for the target calculations
            actions,log_pi=self.policy_network.prop_sample(states)
            q1_output = self.q1_network(states, actions)
            q2_output = self.q2_network(states, actions)
            min_q_output = torch.min(q1_output, q2_output)
            log_pi=torch.mean(log_pi,dim=1).unsqueeze(1)
            Ut = min_q_output-self.alpha*log_pi

        value_pred=self.value_network(states)
        value_loss=(1/self.batch_size)*torch.sum(((value_pred-Ut*(1-if_dones)))**2)

        self.value_optim.zero_grad()
        value_loss.backward()
        self.value_optim.step()
        #---------update_policy_network()-------------
        actions,log_pi=self.policy_network.prop_sample(states)

        log_pi=torch.mean(log_pi,dim=1).unsqueeze(1)


        with torch.no_grad():   
            q1_output = self.q1_network(states, actions)
            q2_output = self.q2_network(states, actions)
            min_q_output = torch.min(q1_output, q2_output)
        policy_loss=(-1/self.batch_size)*torch.sum(min_q_output-self.alpha*log_pi)
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        #------update td error------------------------

        td_error=self.compute_td_error(rewards,states,actions)
        #print(td_error.shape)
        self.memory.update_td_error(td_error,select_indexs)

    def update_target_network(self):
        with torch.no_grad():  # Ensure computations are done without tracking gradients
            for target_param, local_param in zip(self.target_network.parameters(), self.value_network.parameters()):
                updated_weight = self.tou * local_param.data + (1 - self.tou) * target_param.data
                target_param.data.copy_(updated_weight)
        
    def update_hyper(self):
        #self.alpha = max(0.1, self.alpha+(1-0.1)/10000)   #slowly down (exploration)
        #self.gamma=min(0.99, self.gamma+(0.99-0.5) / 1000) #target network importance
        pass
    def save_policy(self):
        torch.save(self.policy_network.state_dict(), '112061588_hw4_data')
        torch.save(self.value_network.state_dict(), 'value_network_data_1')
        torch.save(self.target_network.state_dict(),'target_network_1')
        torch.save(self.q1_network.state_dict(),'q1_network_data_1')  
        torch.save(self.q2_network.state_dict(),'q2_network_data_1')     
    def learn(self):
        # Implementation of learning from the replay buffer
        obs=self.env.reset()
        obs,batch_obs,batch_acts, batch_rtgs,td_error,if_done = self.skip_frame(obs,if_random=False)
        total_reward=0
        i=0
        total_reward=0
        total_td_error=0
        max_reward=0
        for t in range(self.total_episode):
            obs = self.env.reset()  # Reset the environment if done
            ep_reward=0
            while(True):
                i+=1
                if_done=False
                if_random=False
                #if(len(self.memory.buffer)<self.batch_size):
                 #   if_random=True
                obs,next_batch_obs,batch_acts, batch_rtgs,td_error,if_done = self.skip_frame(obs,if_random)
                ep_reward+=torch.sum(batch_rtgs)
                total_td_error+=td_error
                self.memory.push(batch_obs,batch_acts,batch_rtgs,next_batch_obs,if_done,td_error)
                batch_obs=next_batch_obs
                #self.update_hyper()
                if(i%10==0):
                    self.update_parameter()
                    self.update_target_network()
                if (ep_reward>=max_reward or i%30==0):
                    self.update_target_network()
                    self.save_policy()
                    max_reward=ep_reward
                if (len(self.memory.buffer)>20000):
                    self.memory.memory_reset()
                if if_done:
                    break
            total_reward+=ep_reward
            if t%1==0:
                print("---------------------every 10 episode result------------------- ")
                print(t," episode ,mean reward:",float(total_reward))
                print("td error:",total_td_error,"\n")
                total_reward=0
                total_td_error=0
    def load_policy(self):
        path_to_policy: str = '112061588_hw4_data'
        path_to_value: str = 'target_network_1'
        path_to_q1: str = 'q1_network_data_1'
        path_to_q2: str = 'q2_network_data_1'
        path_to_target:str='target_network_1'
        agent.policy_network.load_state_dict(torch.load(path_to_policy, map_location=self.device))
        agent.value_network.load_state_dict(torch.load(path_to_value, map_location=self.device))
        agent.q1_network.load_state_dict(torch.load(path_to_q1, map_location=self.device))
        agent.q2_network.load_state_dict(torch.load(path_to_q2, map_location=self.device))
        agent.target_network.load_state_dict(torch.load(path_to_target, map_location=self.device))
    def skip_frame(self,obs,if_random):
        action = self.training_act(obs,if_random)
        if_done=False
        rewards=0
        for i in range(self.skip):
            obs, reward, done, _ = self.env.step(action)
            rewards+=reward
            if done:
                if_done=True
                break
        batch_obs=self.process_observation(obs).unsqueeze(0)
        batch_acts= torch.tensor(np.array([action]))
        batch_rtgs=torch.tensor(np.array([rewards]))
        td_error=float(torch.sum(self.compute_td_error(batch_rtgs,batch_obs,batch_acts)))
        return obs,batch_obs,batch_acts, batch_rtgs,td_error,float(if_done)

#-----------start train---------------------------------------

# agent=Agent()
# agent.load_policy()
# print("---------------start learning----------------------------------")
# print("Using cuda:", torch.cuda.is_available())
# agent.learn()

#--------------start test----------------------------------------

env = L2M2019Env(difficulty=2,visualize=True)
observation = env.reset()
agent=Agent()
agent.load_policy()

rewards=[]
for i in range(10):
    rew=0
    observation=env.reset()
    while(True):

        action=agent.act(observation)
        
        observation, reward, done, info = env.step(action)
        rew+=reward
        if(done):
            break
    rewards.append(rew)

print ("average reward=",sum(rewards)/len(rewards))
# import matplotlib.pyplot as plt
# plt.figure(1)
# plt.plot(range(len(rewards)),rewards)
# plt.show()
