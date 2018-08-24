import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import sys
import os
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers

def mysoftmax(a):
    a=np.exp(a-np.max(a))
    return a/max(np.sum(a),1e-9)

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
    # Environment
    parser.add_argument("--scenario", type=str, default="simple", help="name of the scenario script")
    parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")
    parser.add_argument("--num-adversaries", type=int, default=0, help="number of adversaries")
    parser.add_argument("--good-policy", type=str, default="maddpg", help="policy for good agents")
    parser.add_argument("--adv-policy", type=str, default="maddpg", help="policy of adversaries")
    # Core training parameters
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=1024, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=64, help="number of units in the mlp")
    parser.add_argument("--adv-eps", type=float, default=1e-3, help="adversarial training rate")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1000, help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")
    # Evaluation
    parser.add_argument("--restore", action="store_true", default=False)
    parser.add_argument("--display", action="store_true", default=False)
    parser.add_argument("--benchmark", action="store_true", default=False)
    parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
    parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")
    return parser.parse_args()

def mlp_model(input, num_outputs, scope, reuse=False, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out

def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def get_trainers(env, num_adversaries, obs_shape_n, arglist, policy_name, adversarial = False):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(num_adversaries):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name=='ddpg', policy_name, adversarial))
    for i in range(num_adversaries, env.n):
        trainers.append(trainer(
            "agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
            policy_name=='ddpg', policy_name, adversarial))
    return trainers


def train(arglist):
    with U.single_threaded_session():
        # Create environment
        num_env = 3
        env = [make_env(arglist.scenario, arglist, arglist.benchmark) for i in range(num_env)]

        # Create agent trainers
        obs_shape_n = [[env[0].observation_space[i].shape for i in range(env[0].n)] for idx in range(num_env)]
        num_adversaries = min(env[0].n, arglist.num_adversaries)
        names = ['ddpg', 'maddpg', 'mmmaddpg']
        noise_on = [False, False, True]
        assert(len(names) == num_env)
        assert(len(noise_on) == num_env)
        trainers = [get_trainers(env[i], num_adversaries, obs_shape_n[i], arglist, names[i], noise_on[i]) for i in range(num_env)]
        for idx in range(num_env):
            print("PolicyName: {}\n       Env: {}\n   Trainer: {}".format(names[idx], env[idx], [trainer.debuginfo() for trainer in trainers[idx]]))
        # Initialize
        U.initialize()

        # Load previous results, if necessary
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.display or arglist.restore or arglist.benchmark:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)

        episode_rewards = [[0.0] for _ in range(num_env)]  # sum of rewards for all agents
        agent_rewards = [[[0.0] for _ in range(env[0].n)] for _ in range(num_env)] # individual agent reward
        final_ep_rewards = [[] for _ in range(num_env)]  # sum of rewards for training curve
        final_ep_ag_rewards = [[] for _ in range(num_env)] # agent rewards for training curve
        agent_info = [[[[]]] for _ in range(num_env)] # placeholder for benchmarking info
        saver = tf.train.Saver()
        obs_n = [env[i].reset() for i in range(num_env)]
        episode_step = [0 for _ in range(num_env)]
        train_step = [0 for _ in range(num_env)]
        t_start = time.time()

        print('Starting iterations...')
        while True:
            for turn in range(num_env):
                while True:
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(trainers[turn],obs_n[turn])]
                    # environment step
                    new_obs_n, rew_n, done_n, info_n = env[turn].step([mysoftmax(elem) for elem in action_n])
                    episode_step[turn] += 1
                    done = all(done_n)
                    terminal = (episode_step[turn] >= arglist.max_episode_len)
                    # collect experience
                    for i, agent in enumerate(trainers[turn]):
                        agent.experience(obs_n[turn][i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                    obs_n[turn] = new_obs_n

                    for i, rew in enumerate(rew_n):
                        episode_rewards[turn][-1] += rew
                        agent_rewards[turn][i][-1] += rew

                    if done or terminal:
                        obs_n[turn] = env[turn].reset()
                        episode_step[turn] = 0
                        episode_rewards[turn].append(0)
                        for a in agent_rewards[turn]:
                            a.append(0)
                        agent_info[turn].append([[]])

                    # increment global step counter
                    train_step[turn] += 1

                    # for benchmarking learned policies
                    if arglist.benchmark:
                        for i, info in enumerate(info_n):
                            agent_info[turn][-1][i].append(info_n['n'])
                        if train_step[turn] > arglist.benchmark_iters and (done or terminal):
                            file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                            print('Finished benchmarking, now saving...')
                            with open(file_name, 'wb') as fp:
                                pickle.dump(agent_info[:-1], fp)
                            break
                        continue

                    # for displaying learned policies
                    if arglist.display:
                        time.sleep(0.1)
                        env[turn].render()
                        continue

                    # update all trainers, if not in display or benchmark mode
                    loss = None
                    for agent in trainers[turn]:
                        agent.preupdate()
                    for agent in trainers[turn]:
                        loss = agent.update(trainers[turn], train_step[turn])

                    # save model, display training output
                    if terminal and (len(episode_rewards[turn]) % arglist.save_rate == 0):
                        # print statement depends on whether or not there are adversaries
                        if num_adversaries == 0:
                            print("policy: {:8s}, steps: {}, episodes: {}, mean episode reward: {:4.3f}, time: {}".format(names[turn],
                                train_step[turn], len(episode_rewards[turn]), np.mean(episode_rewards[turn][-arglist.save_rate:]), round(time.time()-t_start, 3)))
                        else:
                            print("policy: {:8s}, steps: {}, episodes: {}, mean episode reward: {:4.3f}, agent episode reward: {}, time: {}".format(names[turn],
                                train_step[turn], len(episode_rewards[turn]), np.mean(episode_rewards[turn][-arglist.save_rate:]),
                                [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards[turn]], round(time.time()-t_start, 3)))
                        t_start = time.time()
                        # Keep track of final episode reward
                        final_ep_rewards[turn].append(np.mean(episode_rewards[turn][-arglist.save_rate:]))
                        for rew in agent_rewards[turn]:
                            final_ep_ag_rewards[turn].append(np.mean(rew[-arglist.save_rate:]))
                        break
            U.save_state(arglist.save_dir, global_step = len(episode_rewards[0]), saver=saver)
            # saves final episode reward for plotting training curve later
            if len(episode_rewards[0]) > arglist.num_episodes:
                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                if not os.path.exists(os.path.dirname(rew_file_name)):
                    try:
                        os.makedirs(os.path.dirname(rew_file_name))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise
                with open(rew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_rewards, fp)
                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                with open(agrew_file_name, 'wb') as fp:
                    pickle.dump(final_ep_ag_rewards, fp)
                print('...Finished total of {} episodes.'.format(len(episode_rewards[0])))
                break

if __name__ == '__main__':
    arglist = parse_args()
    print(arglist)
    train(arglist)
