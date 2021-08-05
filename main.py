"""
The MLP architecture was the same as the one
adopted with DQN, except that in this case the output of the network was a matrix Ψ˜
i ∈ R
9×12
representing ψ˜
i
(s, a) ∈ R
12 for each a ∈ A. 
"""

def main():
    """
    Main contains the env-agent loop.
    Adapted from cleanrl https://github.com/vwxyzjn/cleanrl
    """ 

    # parse args

    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    
    from torch.utils.tensorboard import SummaryWriter

    import argparse
    from distutils.util import strtobool
    import collections
    import numpy as np
    import gym
    from gym.wrappers import TimeLimit, Monitor
    from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete, Space
    import time
    import random
    import os
    
    torch.autograd.set_detect_anomaly(True)
    from utils import create_env, create_net, create_replay_buffer, set_seeds, linear_schedule, change_task

    parser = argparse.ArgumentParser(description="multi task RL")
    # Common arguments
    parser.add_argument('--exp_name', type=str, default=os.path.basename(__file__).rstrip(".py"),
                        help='the name of this experiment')
    parser.add_argument("--alg", type=str, default="DQN")
    parser.add_argument('--gym_id', type=str, default="MiniGrid-MTEnvFourRooms-v0",
                        help='the id of the gym environment')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=2,
                        help='seed of the experiment')
    parser.add_argument('--total_timesteps', type=int, default=50000000,
                        help='total timesteps of the experiments')
    parser.add_argument('--torch_deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True,
                        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--prod_mode', type=lambda x:bool(strtobool(x)), default=False, nargs='?', const=True,
                        help='run the script in production mode and use wandb to log outputs')
    parser.add_argument("--wandb_project_name",  type=str, default=None, help="the wandb's project name")

    parser.add_argument('--wandb_entity', type=str, default=None,
                        help="the entity (team) of wandb's project")
    
    # Algorithm specific arguments
    parser.add_argument('--buffer_size', type=int, default=1000000,
                         help='the replay memory buffer size')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='the discount factor gamma')
    parser.add_argument('--target_network_frequency', type=int, default=1000,
                        help="the timesteps it takes to update the target network")
    parser.add_argument('--max_grad_norm', type=float, default=0.5,
                        help='the maximum norm for the gradient clipping')
    parser.add_argument('--batch_size', type=int, default=32,
                        help="the batch size of sample from the reply memory")
    parser.add_argument('--start_e', type=float, default=1.,
                        help="the starting epsilon for exploration")
    parser.add_argument('--end_e', type=float, default=0.02,
                        help="the ending epsilon for exploration")
    parser.add_argument('--exploration_fraction', type=float, default=0.10,
                        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument('--learning_starts', type=int, default=80000,
                        help="timestep to start learning")
    parser.add_argument('--train_frequency', type=int, default=4,
                        help="the frequency of training")
    parser.add_argument('--mu_net_coeff', type=float, default=3,
                        help="coefficent used for determining the init of mu net")
    parser.add_argument('--fully_observable', type=lambda x:bool(strtobool(x)), default=False,help="use full obs wrapper or not")
    parser.add_argument('--task_frequency', type=int, default=20000, help="how many transitions before changing tasks")
    parser.add_argument('--on_policy', type=lambda x:bool(strtobool(x)),default=False, help="use on or off policy evaluation")
    parser.add_argument('--struct_task', type=str, default="default",help="which structural task to use")
    args = parser.parse_args()


    device = torch.device('cuda' if torch.cuda.is_available() and args.cuda else 'cpu')
    experiment_name = f"{args.gym_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    writer = SummaryWriter(f"runs/{experiment_name}")
    writer.add_text('hyperparameters', "|param|value|\n|-|-|\n%s" % (
        '\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()])))

    if args.prod_mode:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity, sync_tensorboard=True, config=vars(args), name=experiment_name, monitor_gym=True, save_code=True)

    writer = SummaryWriter(f"/tmp/{experiment_name}")


    env = create_env(args.gym_id, args.struct_task)
    net = create_net(args.alg, env,args.learning_rate)
    target_net = create_net(args.alg, env, args.learning_rate)
    
    net.to(device)
    target_net.to(device)

    rb = create_replay_buffer(args.buffer_size,args.alg, args.gym_id, args.fully_observable)

    set_seeds(env,args.seed, args.torch_deterministic)

    obs = env.reset()
    episode_reward = 0

    for global_step in range(args.total_timesteps):
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction*args.total_timesteps, global_step)

        obs = np.array(obs)
        logits = net.forward(obs.reshape((1,)+obs.shape), device)[0]

        if random.random() < epsilon:
            action = env.action_space.sample() 
        else:
            action = torch.argmax(logits, dim=1).tolist()[0][0]

        next_obs, reward, done, info = env.step(action)
        
        episode_reward += reward

        rb.put((obs, action, reward, next_obs, done))

        net.post_step(obs, action, reward, next_obs, done)

        if global_step % args.train_frequency == 0 and global_step >= args.learning_starts:
            net.update(rb, target_net,writer,device,args.gamma, global_step, args.batch_size, args.max_grad_norm, args.on_policy)
        
        if global_step % args.target_network_frequency == 0:
            target_net.load_state_dict(net.state_dict())

        obs = next_obs

        if global_step % args.task_frequency == 0:
            env = change_task(env, writer, args.struct_task)

        if done:
            # important to note that because `EpisodicLifeEnv` wrapper is applied,
            # the real episode reward is actually the sum of episode reward of 5 lives
            print(f"global_step={global_step}, episode_reward={episode_reward}")
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)
            writer.add_scalar("charts/epsilon", epsilon, global_step)

            net.compute_extra_stats(writer, env, episode_reward, global_step, args.prod_mode) 

            obs, episode_reward = env.reset(), 0

    env.close()
    writer.close()

if __name__ == "__main__":
    main()
