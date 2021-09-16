import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random 

# slight hack to get aroud the hyphen 
import importlib
hf = importlib.import_module("hopfield-layers.modules")
#from hopfield_layers.modules import HopfieldLayer

class View(nn.Module):
    def __init__(self, shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        x = x.view(*self.shape)
        return x

class DebugPrint(nn.Module):
    def __init__(self):
        super(DebugPrint, self).__init__()

    def forward(self, x):
        print(x.shape)
        return x

# TODO could make everything inhert from an abstract base class
class QNetwork(nn.Module):
    """
    Simple DQN with a target network
    """
    def __init__(self, env, frames=3, lr: float = 3e-4):
        super(QNetwork, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2)

        self.network = nn.Sequential(
            #Scale(1/255),
            nn.Conv2d(frames, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32,(2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n)
        )
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = lr)

    def forward(self, x, device):
        x = np.swapaxes(x,1,3)

        x = torch.FloatTensor(x).to(device)
        # TODO this does not work atm
        # return this in a list to allow for quick and dirty
        # interoperability between algorithms in the main loop
        q = self.network(x)
        return [q, 1]

    def post_step(self,obs, action, reward, next_obs, done):
        # vanilla q learning doesn't have any special post-step updating
        pass

    def update(self,rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy):
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
        #TODO assert that on_policy must be false for DQN)
        with torch.no_grad():
            target_max = torch.max(target_net.forward(s_next_obses, device)[0], dim=1)[0]
            td_target = torch.FloatTensor(s_rewards).to(device) + gamma * target_max * (1 - torch.FloatTensor(s_dones).to(device))

        old_val = self.forward(s_obs, device)[0].gather(1, torch.LongTensor(s_actions).view(-1,1).to(device)).squeeze()
        loss = self.loss_fn(td_target, old_val)
            
        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", loss, global_step)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)
        self.optimizer.step()

    def compute_extra_stats(self,writer,env,episode_reward, global_step, prod):
        pass


class SFNet(nn.Module):
    """
    Deep successor feature net
    """
    def __init__(self, env, frames = 3, lr: float = 3e-4, phi_dim = 16, feature_loss = "l_stp1", action_condition = True):
        super(SFNet, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2)
        self.phi_dim = phi_dim
        self.feature_loss = feature_loss

        # discrete only for now
        self.n_actions = env.action_space.n
        self.action_condition = action_condition

        self.conv_fwd = nn.Sequential(
            nn.Conv2d(frames, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32,(2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU()
        )
        self.conv_phi = nn.Sequential(
            #Scale(1/255),
            nn.Conv2d(frames, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32,(2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.phi_dim),
            nn.ReLU()
        )

        if feature_loss == "l_st" or feature_loss == "l_stp1":
            
            _dim = self.phi_dim + self.n_actions if self.action_condition else self.phi_dim
            self.conv_inv_phi = nn.Sequential(
            nn.Linear(_dim, 512),
            nn.ReLU(),
            nn.Linear(512,self.linear_embedding_size),
            nn.ReLU(),
            View((-1, 64, 1, 1)),
            nn.ConvTranspose2d(64,32, (2,2)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, (2,2)),
            nn.ReLU(),
            #3x3 - 6x6
            nn.ConvTranspose2d(16,16,(2,2), stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,frames, (2,2)),
            )

        # need to output a matrix phi_dim X num actions
        self.SF = nn.Sequential(nn.Linear(512, 256), 
                                nn.ReLU(),
                                nn.Linear(256, self.phi_dim*self.n_actions))

        #self.w = nn.Linear(self.phi_dim, 1)
        self.r_pred = nn.Linear(self.phi_dim, 1)
        self.w = torch.rand([self.phi_dim,1], requires_grad = True)
        self.loss_fn = nn.MSELoss()
        self.optimizer_feature = torch.optim.Adam([*self.conv_phi.parameters(), *self.conv_inv_phi.parameters(), *self.r_pred.parameters()],lr = lr)
        self.optimizer_w = torch.optim.Adam([self.w],lr=2*lr)
        self.optimizer_psi = torch.optim.Adam([*self.conv_fwd.parameters(),*self.SF.parameters()],lr=lr)
 
    def forward(self, x, device):
        x = np.swapaxes(x,1,3)

        x = torch.FloatTensor(x).to(device)
        z = self.conv_fwd(x)

        # psi: B x A x d_phi
        psi = self.SF(z)

        psi = psi.reshape(-1, self.n_actions, self.phi_dim)
        _w = self.w.unsqueeze(0).repeat(psi.shape[0], 1,1).to(device)
        q = torch.bmm(psi,_w).squeeze(2)
        return q,psi

    def reconstruct_input(self, x, device, actions=None):
        x = np.swapaxes(x,1,3)
        x = torch.FloatTensor(x).to(device)
        phi = self.conv_phi(x)
        r = self.r_pred(phi)
        if actions is not None:
            actions = torch.LongTensor(actions).to(device)
            one_hot_actions = F.one_hot(actions, num_classes=self.n_actions)
        if self.action_condition:
            _phi = torch.cat([phi, one_hot_actions], dim=1)
            x_reconstructed = self.conv_inv_phi(_phi).reshape(*x.shape)
        else: 
            x_reconstructed = self.conv_inv_phi(phi).reshape(*x.shape)

        return x_reconstructed, phi, r

    def forward_phi(self, x, device):
        x = np.swapaxes(x,1,3)
        x = torch.FloatTensor(x).to(device)
        phi = self.conv_phi(x)
        return phi

    def post_step(self,obs,action, reward, next_obs, done ):
        pass
    
    def update_phi_encoder(self, rb, target_net, writer, device, gamma,global_step, batch_size, max_grad_norm, on_policy):

        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
        obs_prediction, phi, pred_r = self.reconstruct_input(s_obs, device, s_actions)

        if self.feature_loss == "l_st":
            s_obs = np.swapaxes(s_obs, 1, 3)
            s_obs = torch.FloatTensor(s_obs).to(device)
            reconstruct_loss = self.loss_fn(s_obs.flatten(), obs_prediction.flatten())

        elif self.feature_loss == "l_stp1":
            s_next_obses = np.swapaxes(s_next_obses, 1, 3)
            s_next_obses = torch.FloatTensor(s_next_obses).to(device)
            reconstruct_loss = self.loss_fn(s_next_obses, obs_prediction)
        else:
            reconstruct_loss = 0

        
        reward_loss = self.loss_fn(pred_r.squeeze(1), torch.FloatTensor(s_rewards).to(device))
        regulariser = phi.pow(2).mean()

        reconstruct_loss = reconstruct_loss +reward_loss + 0.1*regulariser
        self.optimizer_feature.zero_grad()
        reconstruct_loss.backward()
        #nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)
        self.optimizer_feature.step()
        
        if global_step % 100 == 0:
            writer.add_scalar("losses/restrt_loss", reconstruct_loss, global_step)

    def update_w(self, rb, target_net, writer, device, gamma,global_step, batch_size, max_grad_norm, on_policy):

        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
        phi = self.forward_phi(s_obs, device).detach()
        expected_reward = torch.matmul(phi, self.w.to(device))
        reward_loss = self.loss_fn(expected_reward.squeeze(1), torch.FloatTensor(s_rewards).to(device))
        
        self.optimizer_w.zero_grad()
        reward_loss.backward()
        self.optimizer_w.step()

        if global_step % 100 == 0:
            writer.add_scalar("losses/reward_loss", reward_loss, global_step)

    def update_SF(self, rb, target_net, writer, device,gamma, global_step, batch_size, max_grad_norm, on_policy):

        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
        with torch.no_grad():
            q_next, target_psi = target_net.forward(s_next_obses, device)
            a_max = torch.max(q_next, dim=1)[1]
            #a_max = a_max.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
            #target_psi = target_psi.gather(1, a_max)
            phi = self.forward_phi(s_obs, device)
            target_psi = phi + torch.FloatTensor(s_dones).unsqueeze(1).to(device) * gamma * target_psi[range(s_obs.shape[0]), a_max.reshape(-1)]
            #target_psi = phi + torch.FloatTensor(s_dones).to(device).unsqueeze(1).repeat(1, self.phi_dim) * gamma * target_psi.squeeze(1)  

        _ , psi = self.forward(s_obs, device)

        s_actions = torch.LongTensor(s_actions).to(device)
        #psi_idx = s_actions.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
        #psi = psi.gather(1, psi_idx).squeeze(1)
        psi = psi[range(s_obs.shape[0]), s_actions.reshape(-1)]
        psi_loss = self.loss_fn(psi, target_psi)

        self.optimizer_psi.zero_grad()
        psi_loss.backward()
        self.optimizer_psi.step()

        if global_step % 100 == 0:
            writer.add_scalar("losses/psi_loss", psi_loss, global_step)

    def update(self,rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy):

        #if global_step < 100000:
            #for _ in range(5):
        self.update_phi_encoder(rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy)
        self.update_SF(rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy)

        #else: 
        self.update_w(rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy)
        #nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)
        #self.optimizer_psi.step()
         
        #if global_step % 100 == 0:
        #    writer.add_histogram("losses/psi", old_psi[0], global_step)
        #    writer.add_histogram("losses/phi", old_phi[0], global_step)
        #    writer.add_histogram("losses/w", self.w.weight.data, global_step)

    def compute_extra_stats(self,writer,env,episode_reward, global_step, prod):
        pass

class SFNetOnlineReward(nn.Module):
    """
    SFNet that does not compute w from the replay buffer, but rather
    a running estimate 
    """
    def __init__(self, env, frames = 3, lr: float = 3e-4, phi_dim = 32):
        super(SFNetOnlineReward, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2)
        self.phi_dim = phi_dim

        # discrete only for now
        self.n_actions = env.action_space.n

        self.conv = nn.Sequential(
            #Scale(1/255),
            nn.Conv2d(frames, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32,(2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU())

        self.phi = nn.Linear(512, self.phi_dim)

        # need to output a matrix phi_dim X num actions
        self.SF = nn.ModuleList([nn.Sequential(nn.Linear(self.phi_dim,self.phi_dim // 2),\
                                                nn.ReLU(),\
                                                nn.Linear(self.phi_dim //2, self.phi_dim)) \
                                            for _ in range(self.n_actions)])
         
        self.w = nn.Parameter(torch.rand(self.phi_dim, 1))

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = lr)

        self.phi_cache = []
        self.reward_cache = []
 
    def forward(self, x, device):
        x = np.swapaxes(x,1,3)

        x = torch.FloatTensor(x).to(device)
        z = self.conv(x)
        phi = self.phi(z)
        # psi: B x A x d_phi
        psi = torch.cat([self.SF[i](phi).unsqueeze(1) for i in range(self.n_actions)] ,dim=1)
        _w = self.w.unsqueeze(0).repeat(psi.shape[0],1,1)
        q = torch.bmm(psi,_w).squeeze(2)
        self.last_phi = phi.detach()

        return [q,phi,psi]

    def post_step(self,obs,action, reward, next_obs, done):
        self.phi_cache.append(self.last_phi)
        self.reward_cache.append(reward)

    def update(self,rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy):
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
#        print(s_n)
        with torch.no_grad():
            if on_policy:
                pass # TODO make sure rb is storing next action as well if on-policy
            else:
                q, _, target_psi = target_net.forward(s_next_obses, device)

                # argmax value = actions to index psi with
                a_max = torch.max(q, dim=1)[1]
                # gather magic
                a_max = a_max.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
                target_psi = target_psi.gather(1, a_max)

               # eval using current w  
                # zero indexing the output just returns the logits
        
        old_q ,old_phi, old_psi = self.forward(s_obs, device)
        s_actions = torch.LongTensor(s_actions).to(device)
        old_psi_idx = s_actions.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
        old_psi = old_psi.gather(1, old_psi_idx)

        # TODO decide if grad needs to be stopped in various places 
        psi_td_target = old_phi.unsqueeze(1) + gamma * target_psi * (1 - torch.FloatTensor(s_dones).to(device))

        #TD loss for SF
        td_loss = self.loss_fn(psi_td_target, old_psi)

        # supervised loss with phi w = r 
        _w_detached = self.w.unsqueeze(0).repeat(old_phi.shape[0],1,1).detach()
        expected_reward = torch.bmm(old_phi.unsqueeze(1),_w_detached)
        reward_loss = self.loss_fn(torch.FloatTensor(s_rewards).to(device), expected_reward.flatten())

        total_loss = td_loss + reward_loss

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", td_loss, global_step)
            writer.add_scalar("losses/reward_loss", reward_loss, global_step)

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)
        self.optimizer.step()
        
        # update w using cache
        phi_cache = torch.cat(self.phi_cache)
        reward_cache = torch.FloatTensor(self.reward_cache).to(device)
        
        _w = self.w.unsqueeze(0).repeat(phi_cache.shape[0],1,1)
        expected_reward_cache = torch.bmm(phi_cache.unsqueeze(1), _w) 
        reward_loss_cache = self.loss_fn(expected_reward_cache.flatten(), reward_cache)

        self.optimizer.zero_grad()
        reward_loss_cache.backward()
        self.optimizer.step()
        
        self.phi_cache = []
        self.reward_cache = []

    def compute_extra_stats(self,writer,env,episode_reward, global_step, prod):

        if prod:
            # TODO include other metrics and metric selection 

            metric = "norm_by_max_positive"
            # calulate the maximum possible reward from objects
            max_r_obj = env.possible_object_rewards()
            if metric == "norm_by_max_positive":

                # floor to get rid of the reward for reaching the goal 
                score =  np.floor(episode_reward) - max_r_obj / (max_r_obj + 0.00001)
                writer.add_scalar("charts/normalised_score", score, global_step)

class SFMiniBatch(SFNet):
    """
    SF that learns online rather than using a replay buffer
    """
    def __init__(self, env, frames = 3, lr: float = 3e-4, phi_dim = 32, feature_loss = "l_st", rollout_length = 16):
        super(SFMiniBatch, self).__init__(env, frames, lr, phi_dim, feature_loss)
        self.rollout_length = rollout_length


    def update(self,rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy):
        if rb.ready:
            s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
#        print(s_n)
            old_q ,old_phi, old_psi = self.forward(s_obs, device)
            old_phi_td = old_phi.clone().detach().unsqueeze(1)
            s_actions = torch.LongTensor(s_actions).to(device)
            with torch.no_grad():
                if on_policy:
                    pass # TODO make sure rb is storing next action as well if on-policy
                else:
                    q, _, target_psi = target_net.forward(s_next_obses, device)
                    # argmax value = actions to index psi with
                    a_max = torch.max(q, dim=1)[1]
                    # gather magic
                    a_max = a_max.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
                    target_psi = target_psi.gather(1, a_max)

                    dones = torch.FloatTensor(s_dones).unsqueeze(1).repeat(1, self.phi_dim).to(device)
                    
                    psi_td_target = target_psi.squeeze(1) * dones
                    psi_td_target = old_phi_td.squeeze(1) + gamma * psi_td_target
            old_psi_idx = s_actions.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
            old_psi = old_psi.gather(1, old_psi_idx).squeeze(1)




            #TD loss for SF
            td_loss = self.loss_fn(psi_td_target, old_psi)
            """
            self.optimizer.zero_grad()
            td_loss.backward()
            
            nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)
            self.optimizer.step()
            """

            # supervised loss with phi w = r 
            _w = self.w.unsqueeze(0).repeat(old_phi.shape[0],1,1)
            expected_reward = torch.bmm(old_phi.unsqueeze(1),_w)
            reward_loss = self.loss_fn(torch.FloatTensor(s_rewards).to(device), expected_reward.flatten())
            
            if self.feature_loss == "l_st":
                obs_prediction = self.conv_inv(old_phi)
                s_obs = np.swapaxes(s_obs, 1, 3)
                s_obs = torch.FloatTensor(s_obs).to(device)
                reconstruct_loss = self.loss_fn(s_obs.flatten(), obs_prediction.flatten())

            elif self.feature_loss == "l_st1":
                obs_prediction = self.conv_inv(old_phi)
                s_next_obses = np.swapaxes(s_next_obses, 1, 3)
                s_next_obses = torch.FloatTensor(s_next_obses).to(device)
                reconstruct_loss = self.loss_fn(s_next_obses, obs_prediction)
            else:
                reconstruct_loss = 0

            #total_loss = td_loss + reward_loss + reconstruct_loss


            # TODO decide if grad needs to be stopped in various places 
            feature_loss = reward_loss + reconstruct_loss
            total_loss = feature_loss + td_loss
             
            self.optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)

            self.optimizer.step()
            
            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", td_loss, global_step)
                writer.add_scalar("losses/restrt_loss", reconstruct_loss, global_step)
                writer.add_scalar("losses/reward_loss", reward_loss, global_step)
            rb.flush()

        else: 
            pass

class SFHopNet(nn.Module):
    """
    Deep successor feature net with Hopfield layer
    """
    def __init__(self, env, frames = 3, lr: float = 3e-4, phi_dim = 32, n_patterns = 25):
        super(SFHopNet, self).__init__()
        n = env.observation_space.shape[0]
        m = env.observation_space.shape[1]
        
        self.linear_embedding_size = 64* ((n-1)//2 -2) * ((m-1)//2 - 2)
        self.phi_dim = phi_dim
        self.n_patterns= n_patterns

        # discrete only for now
        self.n_actions = env.action_space.n

        self.conv = nn.Sequential(
            #Scale(1/255),
            nn.Conv2d(frames, 16, (2,2)),
            nn.ReLU(),
            nn.MaxPool2d((2,2)),
            nn.Conv2d(16, 32,(2,2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2,2)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self.linear_embedding_size, 512),
            nn.ReLU())

        self.phi_candidate = nn.Linear(512, self.phi_dim)

        self.hop = hf.HopfieldLayer(input_size = self.phi_dim,quantity=self.n_patterns)

        # need to output a matrix phi_dim X num actions
        self.SF = nn.ModuleList([nn.Sequential(nn.Linear(self.phi_dim,self.phi_dim // 2),\
                                                nn.ReLU(),\
                                                nn.Linear(self.phi_dim //2, self.phi_dim)) \
                                            for _ in range(self.n_actions)])
         
        self.w = nn.Parameter(torch.rand(self.phi_dim, 1))

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(),lr = lr)
 
    def forward(self, x, device):
        x = np.swapaxes(x,1,3)

        x = torch.FloatTensor(x).to(device)
        z = self.conv(x)
        phi_can = self.phi_candidate(z)
        phi = self.hop(phi_can.unsqueeze(1)).squeeze(1)
        # psi: B x A x d_phi
        psi = torch.cat([self.SF[i](phi).unsqueeze(1) for i in range(self.n_actions)] ,dim=1)
        _w = self.w.unsqueeze(0).repeat(psi.shape[0],1,1)
        q = torch.bmm(psi,_w).squeeze(2)
        return [q,phi,psi]

    def post_step(self,obs,action, reward, next_obs, done ):
        pass

    def update(self,rb, target_net, writer, device, gamma, global_step, batch_size, max_grad_norm, on_policy):
        s_obs, s_actions, s_rewards, s_next_obses, s_dones = rb.sample(batch_size)
#        print(s_n)
        with torch.no_grad():
            if on_policy:
                pass # TODO make sure rb is storing next action as well if on-policy
            else:
                q, _, target_psi = target_net.forward(s_next_obses, device)

                # argmax value = actions to index psi with
                a_max = torch.max(q, dim=1)[1]
                # gather magic
                a_max = a_max.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
                target_psi = target_psi.gather(1, a_max)

               # eval using current w  
                # zero indexing the output just returns the logits
        
        old_q ,old_phi, old_psi = self.forward(s_obs, device)
        s_actions = torch.LongTensor(s_actions).to(device)
        old_psi_idx = s_actions.repeat(1, self.phi_dim).view(-1, 1, self.phi_dim)
        old_psi = old_psi.gather(1, old_psi_idx)

        # TODO decide if grad needs to be stopped in various places 
        psi_td_target = old_phi.unsqueeze(1) + gamma * target_psi * (1 - torch.FloatTensor(s_dones).to(device))

        #TD loss for SF
        td_loss = self.loss_fn(psi_td_target, old_psi)

        # supervised loss with phi w = r 
        _w = self.w.unsqueeze(0).repeat(old_phi.shape[0],1,1)
        expected_reward = torch.bmm(old_phi.unsqueeze(1),_w)
        reward_loss = self.loss_fn(torch.FloatTensor(s_rewards).to(device), expected_reward.flatten())

        total_loss = td_loss + reward_loss

        if global_step % 100 == 0:
            writer.add_scalar("losses/td_loss", td_loss, global_step)
            writer.add_scalar("losses/reward_loss", reward_loss, global_step)

        self.optimizer.zero_grad()
        total_loss.backward()

        nn.utils.clip_grad_norm_(list(self.parameters()), max_grad_norm)
        self.optimizer.step()

    def compute_extra_stats(self,writer,env,episode_reward, global_step, prod):

        if prod:
            # TODO include other metrics and metric selection 

            metric = "norm_by_max_positive"
            # calulate the maximum possible reward from objects
            max_r_obj = env.possible_object_rewards()
            if metric == "norm_by_max_positive":

                # floor to get rid of the reward for reaching the goal 
                score =  np.floor(episode_reward) - max_r_obj / (max_r_obj + 0.00001)
                writer.add_scalar("charts/normalised_score", score, global_step)


