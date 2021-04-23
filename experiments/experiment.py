import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import time
from maddpg.common.tf_util import allow_growth


class Experiment(object):
    def __init__(self, environment, trainer, args=None):
        print("PID: ", str(os.getpid()))

        self.args = self.parser().parse_args() if args is None else args

        self.trainer = trainer
        self.environment = environment()
        self.environment_info = {}


        #################
        # Training type #
        #################

        tf.config.set_visible_devices([], 'GPU')
        # allow_growth()
        # self.args.max_episode_len = 2 # Fast debug

        ###################################
        # Specify input and output spaces #
        ###################################

        self.trainers = [
            self.trainer(
                "agent_%d" % i,
                self.environment.n,
                self.environment.observation_space[i].shape,
                self.environment.action_space[i],
                i,
                self.args
            )
            for i in range(self.environment.n)
        ]

        #############################
        # Specify saving parameters #
        #############################

        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(os.path.dirname(self.args.logs_dir), exist_ok=True)

        nets = {}
        for i, trainer in enumerate(self.trainers):
            for k, v in trainer.model_save_dict().items():
                nets[k + str(i)] = v

        _saver = tf.train.Checkpoint(**nets, step=tf.Variable(0))
        saver = tf.train.CheckpointManager(_saver, self.args.save_dir, max_to_keep=3)

        # Load previous results, if necessary
        if self.args.load_dir == "":
            self.args.load_dir = self.args.save_dir
            loader = saver
        else:
            loader = tf.train.CheckpointManager(_saver, self.args.load_dir, max_to_keep=3)

        if self.args.display or self.args.restore or self.args.benchmark:
            print('Loading previous state...')
            loader.restore_or_initialize()

        ##################################
        # Specify tensorboard parameters #
        ##################################

        logs_writer = tf.summary.create_file_writer(os.path.join(self.args.logs_dir, datetime_str))
        history = np.zeros((6, len(self.trainers)))

        episode_rewards = [0.0]  # sum of rewards for all agents

        #########################
        # Specify replay buffer #
        #########################

        self.replay_buffer_n = [self.init_buffer() for _ in range(self.environment.n)]
        self.max_replay_buffer_len = self.args.batch_size * self.args.max_episode_len
        self.replay_sample_index = [None for _ in range(self.environment.n)]

        ##########################################
        # Initialise environment and observation #
        ##########################################

        self.init_loop()

        obs_n = self.environment.reset()
        episode_step = 0
        tf_train_step = _saver.step

        self.reset_loop()

        print('Starting iterations...')
        t_start = time.time()
        p_start = time.time()

        while True:

            #################
            # OBSERVE & ACT #
            #################

            # get action
            action_n = self.collect_action(obs_n)

            # environment step
            new_obs_n, rew_n, done_n, info_n = self.environment.step(action_n)
            episode_step += 1

            done = all(done_n)
            terminal = (episode_step >= self.args.max_episode_len)
            # collect experience
            self.collect_experience(obs_n, action_n, rew_n, new_obs_n, done_n, terminal)
            obs_n = new_obs_n

            episode_rewards[-1] += rew_n[0]

            if done or terminal:
                obs_n = self.environment.reset()
                episode_step = 0
                episode_rewards.append(0)
                self.reset_loop()

            # increment global step counter
            tf_train_step.assign_add(1)
            train_step = tf_train_step.numpy()

            # for displaying learned policies
            if self.args.display:
                time.sleep(0.1)
                self.environment.render()
                continue

            #########
            # TRAIN #
            #########

            loss = None
            if len(self.replay_buffer_n[0]) >= self.max_replay_buffer_len and train_step % 100 == 0:
                for i, agent in enumerate(self.trainers):
                    exp = self.train_experience(self.replay_buffer_n[i])
                    loss = agent.update(self.trainers, exp)
                    history[:, i] = loss

            if loss is not None:
                metrics = np.mean(history, axis=-1)
                with logs_writer.as_default():
                    tf.summary.scalar("q_loss", metrics[0], step=train_step)
                    tf.summary.scalar("p_loss", metrics[1], step=train_step)
                    tf.summary.scalar("target_q", metrics[2], step=train_step)
                    tf.summary.scalar("reward", metrics[3], step=train_step)
                    tf.summary.scalar("target_q_next", metrics[4], step=train_step)
                    tf.summary.scalar("target_q_std", metrics[5], step=train_step)
                    tf.summary.scalar("mean_reward", np.mean(episode_rewards[-self.args.save_rate:]), step=train_step)
                    tf.summary.scalar("time", round(time.time() - p_start, 3), step=train_step)
                    p_start = time.time()
                logs_writer.flush()

            # save model, display training output
            if terminal and (len(episode_rewards) % self.args.save_rate == 0):
                saver.save()

                # print statement depends on whether or not there are adversaries
                print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                    train_step, len(episode_rewards), np.mean(episode_rewards[-self.args.save_rate:]),
                    round(time.time() - t_start, 3)))

                t_start = time.time()
                episode_rewards = episode_rewards[-self.args.save_rate:]


    @staticmethod
    def parser():
        parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")
        # Environment
        parser.add_argument("--scenario", type=str, default="simple_spread_random", help="name of the scenario script")
        parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
        parser.add_argument("--num-episodes", type=int, default=60000, help="number of episodes")

        # Core training parameters
        parser.add_argument("--lr", type=float, default=1e-2, help="learning rate for Adam optimizer")
        parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
        parser.add_argument("--batch-size", type=int, default=1024,
                            help="number of episodes to optimize at the same time")
        # Checkpointing
        parser.add_argument("--exp-name", type=str, default=None, help="name of the experiment")
        parser.add_argument("--save-dir", type=str, default="/tmp/policy/",
                            help="directory in which training state and model should be saved")
        parser.add_argument("--save-rate", type=int, default=1000,
                            help="save model once every time this many episodes are completed")
        parser.add_argument("--load-dir", type=str, default="",
                            help="directory in which training state and model are loaded")
        # Evaluation
        parser.add_argument("--restore", action="store_true", default=False)
        parser.add_argument("--display", action="store_true", default=False)
        parser.add_argument("--benchmark", action="store_true", default=False)
        parser.add_argument("--benchmark-iters", type=int, default=100000,
                            help="number of iterations run for benchmarking")
        parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/",
                            help="directory where benchmark data is saved")
        parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                            help="directory where plot data is saved")
        parser.add_argument("--logs-dir", type=str, default="../../logs_dir/",
                            help="directory where logs for tensorboard are saved")
        return parser

    def init_buffer(self):
        raise NotImplemented()

    def init_loop(self):
        raise NotImplemented()

    def reset_loop(self):
        raise NotImplemented()

    def collect_action(self, obs_n):
        raise NotImplemented()

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal):
        raise NotImplemented()

    def train_experience(self, buffer):
        raise NotImplemented()
