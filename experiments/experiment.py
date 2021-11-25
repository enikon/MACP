import argparse
import os
from datetime import datetime

import numpy as np
import tensorflow as tf

import time
from maddpg.common.tf_util import allow_growth
from experiments.environmenter import scenario_environment

import json

ignore_noise_metrics = False
manual_curriculum = False


class Experiment(object):
    def __init__(self, environment, trainer, name, args=None):
        print("PID: ", str(os.getpid()))
        self.name = name

        ######################
        # Reading parameters #
        ######################

        self.args, _ = self.parser().parse_known_args()

        t_args = argparse.Namespace()

        if os.path.exists('default'):
            with open('default', 'rt') as f:
                t_args.__dict__.update(json.load(f))
                print('Default file loaded')

        if self.args.config is not None and os.path.exists(self.args.config):
            with open(self.args.config, 'rt') as f:
                t_args.__dict__.update(json.load(f))
                print('Config file loaded')

        if args is not None:
            t_args.__dict__.update(args)
            print('Used enforced arguments')

        self.args, _ = self.parser().parse_known_args(namespace=t_args)

        #####################
        # Initialise fields #
        #####################

        self.trainer = trainer
        self.environment = environment()
        self.environment_info = {}

        #################
        # Training type #
        #################

        if self.args.GPU:
            allow_growth()
        else:
            tf.config.set_visible_devices([], 'GPU')

        #self.args.max_episode_len = 2 # Fast debug

        self.noise_adapting = None
        self.noise_adapting_value = tf.constant(0., dtype=tf.float32)
        self.noise_adapting_value_iterator = tf.constant(0., dtype=tf.float32)
        self.noise_adapting_speed = tf.constant(0., dtype=tf.float32)
        self.noise_adapting_max = tf.constant(1., dtype=tf.float32)

    def init(self):

        #############################
        # Specify saving parameters #
        #############################

        self.saving_enabled = not self.args.display

        # Create experiment name
        datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
        experiment_full_name =\
            ('' if self.args.exp_name == '' else self.args.exp_name+'_') + \
            ('train_' if not self.args.evaluate else 'evalC_') + \
            (str(self.name)+'_') + \
            datetime_str
        experiment_direction = os.path.join(self.args.save_dir, experiment_full_name, '')

        # Save config file
        if self.saving_enabled:
            os.makedirs(os.path.dirname(experiment_direction), exist_ok=True)
            with open(os.path.join(experiment_direction, 'config'), 'wt') as f:
                json.dump(vars(self.args), f, indent=4)

        # Define networks save dictionary
        nets = {}
        for i, trainer in enumerate(self.trainers):
            for k, v in trainer.model_save_dict().items():
                nets[k + str(i)] = v

        # Define checkpoints and directories
        _saver = tf.train.Checkpoint(**nets, step=tf.Variable(0))
        _best_saver = tf.train.Checkpoint(**nets, step=_saver.step)

        save_last_dir = os.path.join(experiment_direction, 'last')
        save_best_dir = os.path.join(experiment_direction, 'best')
        save_logs_dir = os.path.join(experiment_direction, 'logs', '')
        save_profile_dir = os.path.join(experiment_direction, 'profile', '')

        saver = tf.train.CheckpointManager(_saver, save_last_dir, max_to_keep=3)
        best_saver = tf.train.CheckpointManager(_best_saver, save_best_dir, max_to_keep=3, checkpoint_name='best_ckpt')

        if self.saving_enabled:
            os.makedirs(os.path.dirname(save_logs_dir), exist_ok=True)

        # Load previous results, if necessary
        if self.args.load_dir == "":
            self.args.load_dir = save_best_dir if self.args.restore_best else save_last_dir
            loader = best_saver if self.args.restore_best else saver
        else:
            loader = tf.train.CheckpointManager(
                _best_saver if self.args.restore_best else _saver,
                os.path.join(
                    self.args.load_dir,
                    'best' if self.args.restore_best else 'last'
                ), max_to_keep=3)

        if self.args.restore or self.args.restore_best:
            print('Loading previous state...')
            test = loader.restore_or_initialize()
            if test is None:
                print('Wrong path!!!')
                return

        best_value = float('-inf')

        ##################################
        # Specify tensorboard parameters #
        ##################################

        if self.saving_enabled:
            logs_writer = tf.summary.create_file_writer(save_logs_dir)

        history = np.zeros((6, len(self.trainers)))

        n_agents = self.environment.n

        episode_info = np.full((2, self.args.logs_range_collect), np.nan)  # sum of rewards for all agents
        episode_rewards = np.full((1, self.args.max_episode_len), np.nan)
        episode_metrics = np.full((2, 2, n_agents, self.args.logs_range_collect), np.nan)
        benchmark_metrics = np.full((5, self.args.logs_range_collect), np.nan)

        episode_info[0, 0] = 0
        episode_info[1, 0] = 0
        episode_metrics[:, :, :, 0] = np.zeros((2, 2, n_agents))

        #########################
        # Specify replay buffer #
        #########################

        self.replay_buffer_n = [self.init_buffer() for _ in range(len(self.trainers))]
        self.min_experiences_number = self.args.batch_size * self.args.max_episode_len
        self.replay_sample_index = [None for _ in range(len(self.trainers))]

        ##########################################
        # Initialise environment and observation #
        ##########################################

        self.init_loop()

        obs_n = self.environment.reset()
        episode_step = 0
        episode_number = 0
        update_number = 0

        patience = self.args.decaying_lr_patience
        patience_iterator = 0
        max_rew = -100000
        using_decay = self.args.decaying_lr is not None

        tf_train_step = _saver.step
        train_step = tf_train_step.numpy()

        self.reset_loop()

        print('Starting iterations...')
        t_start = time.time()
        p_start = time.time()
        f_start = time.time()
        d_start = time.time()

        running = True

        # Profiling if enabled
        not_graphed=True
        profiling = 0
        if self.args.profile:
            from tensorflow.python.profiler import profiler_v2 as profiler
            profiler.warmup()

        if self.args.noise_mask == 'all':
            self.args.noise_mask_value = 0
        elif self.args.noise_mask == 'none':
            self.args.noise_mask_value = n_agents

        mask = tf.constant(
            np.concatenate([
                np.zeros(shape=(self.args.noise_mask_value, 1)),
                np.ones(shape=(n_agents - self.args.noise_mask_value, 1))
            ], axis=0),
            dtype=tf.float32
        )

        if self.args.noise_mask == 'random':
            mask = tf.random.shuffle(mask)
        elif self.args.noise_mask == 'fixed':
            pass

        while running:

            #################
            # OBSERVE & ACT #
            #################

            # get action
            action_n = self.collect_action(obs_n, mask)

            # environment step
            new_obs_n, rew_n, done_n, info_n = self.environment.step(action_n)
            done = all(done_n)

            terminal = episode_step+1 >= self.args.max_episode_len

            # collect experience
            if not self.args.evaluate:
                self.collect_experience(obs_n, action_n, np.array(rew_n).astype(np.float32), new_obs_n,
                                        np.array(done_n).astype(np.float32), terminal, mask)

            if episode_number % self.args.metrics_rate_collect == 0 and self.saving_enabled:
                metrics_i = int((episode_number / self.args.metrics_rate_collect) % self.args.logs_range_collect)
                if episode_step == 0:
                    if not ignore_noise_metrics:
                        episode_metrics[:, :, :, metrics_i] = self.collect_metrics(obs_n, mask)
                    benchmark_metrics[:, metrics_i] = np.array(self.environment._get_benchmark())
                else:
                    if not ignore_noise_metrics:
                        episode_metrics[:, :, :, metrics_i] += self.collect_metrics(obs_n, mask)
                    #rew, collision, occupied, max
                    benchmarks = self.environment._get_benchmark()
                    benchmark_metrics[0, metrics_i] += benchmarks[0]
                    benchmark_metrics[1, metrics_i] += benchmarks[1]
                    benchmark_metrics[2, metrics_i] = max(benchmark_metrics[2, metrics_i], benchmarks[2])
                    benchmark_metrics[3, metrics_i] = min(benchmark_metrics[3, metrics_i], benchmarks[3])
                    benchmark_metrics[4, metrics_i] = max(benchmark_metrics[4, metrics_i], benchmarks[4])
                # [send, recv] [signal, noise] [a1,a2,a3]

            obs_n = new_obs_n
            episode_info[0, episode_number % self.args.logs_range_collect] += rew_n[0]
            episode_rewards[0, episode_step % self.args.max_episode_len] = rew_n[0]

            episode_step += 1

            if done or terminal:
                ###############################
                # EVALUATE END OF THE EPISODE #
                ###############################

                episode_info[1, episode_number % self.args.logs_range_collect] = episode_step
                episode_number += 1

                self.noise_adapting_value_iterator += self.noise_adapting_speed
                self.noise_adapting_value = tf.clip_by_value(self.noise_adapting_value_iterator, 0, self.noise_adapting_max)

                # LR Decay
                if using_decay:
                    candidate_rew = np.nanmean(episode_info[0])
                    if candidate_rew > max_rew:
                        max_rew = candidate_rew
                        patience_iterator = 0
                    else:
                        patience_iterator += 1
                    if patience_iterator > patience:
                        patience_iterator = 0
                        c_lr = -1
                        for t in self.trainers:
                            for n in t.get_networks():
                                new_lr = n.optimizer.learning_rate * self.args.decaying_lr
                                c_lr = new_lr if self.args.decaying_lr_limit is None else max(new_lr, self.args.decaying_lr_limit)
                                n.optimizer.learning_rate.assign(c_lr)
                        print('New learning rate:'+str(c_lr))

                # Printing
                if episode_number % self.args.logs_rate_collect == 0 and self.saving_enabled:

                    ep_rew = np.nanmean(episode_info[0])
                    ep_len = np.nanmean(episode_info[1])
                    ep_last_rew = np.nansum(episode_rewards[0])
                    ep_time = time.time() - t_start

                    with logs_writer.as_default():
                        # Mean of the last collect_rate episodes, collected on every step
                        tf.summary.scalar("01_general/episodes_reward", ep_rew, step=episode_number)
                        tf.summary.scalar("01_general/last_episode_reward", ep_last_rew, step=episode_number)
                        tf.summary.scalar("01_general/episodes_length", ep_len, step=episode_number)

                        tf.summary.scalar("01_general/environment_time", round(ep_time, 3), step=episode_number)

                        tf.summary.scalar("09_extra/episodes_reward_in_steps", ep_rew, step=train_step)
                        tf.summary.scalar("09_extra/episodes_length_in_steps", ep_len, step=train_step)

                    t_start = time.time()

                if not self.args.no_console and episode_number % self.args.logs_rate_display == 0:
                    ed_time = time.time() - d_start
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step+1, episode_number, np.nanmean(episode_info[0]), round(ed_time, 3)))
                    d_start = time.time()

                if not self.args.evaluate and episode_number % self.args.save_rate == 0 and self.saving_enabled:
                    saver.save()
                    if episode_number >= self.args.skip_best_for_episodes:
                        best_value_candidate = np.nanmean(episode_info[0])
                        if best_value < best_value_candidate:
                            best_value = best_value_candidate
                            best_saver.save()
                            print("Saved best:\n\t steps: {}, episodes: {}, mean episode reward: {}".format(
                                train_step+1, episode_number, best_value_candidate))

                if episode_number % self.args.logs_rate_collect == 0 and self.saving_enabled:
                    if not ignore_noise_metrics:
                        __noise = np.nanmean(episode_metrics / self.args.max_episode_len, axis=(-1, -2))

                        t__noise, t__noise_special = None, None
                        metrics_i = int(
                            ((episode_number - 1) / self.args.metrics_rate_collect) % self.args.logs_range_collect)

                        if self.args.noise_mask == 'all':
                            t__noise = np.nanmean(episode_metrics[:, :, :, metrics_i] / self.args.max_episode_len, axis=-1)
                        elif not self.args.noise_mask == 'none':
                            t__noise_all = episode_metrics[:, :, :, metrics_i] / self.args.max_episode_len
                            t__noise         = np.nanmean(t__noise_all[:, :, mask[:, 0].numpy() == 0], axis=-1)
                            t__noise_special = np.nanmean(t__noise_all[:, :, mask[:, 0].numpy() == 1], axis=-1)

                    t__bench = np.nanmean(benchmark_metrics, axis=-1)
                    with logs_writer.as_default():
                        tf.summary.scalar("03_metrics/rew", t__bench[0], step=episode_number)
                        tf.summary.scalar("03_metrics/collisions", t__bench[1], step=episode_number)
                        tf.summary.scalar("03_metrics/occupied_landmarks", t__bench[2], step=episode_number)
                        tf.summary.scalar("03_metrics/farthest_distance", t__bench[3], step=episode_number)
                        tf.summary.scalar("03_metrics/completion", t__bench[4], step=episode_number)

                        # #TODO HACK COMPLETION GROWTH
                        if manual_curriculum:
                            if t__bench[4] > 0.975:
                                running = False
                                print(":: :: "+str(t__bench[4]))
                            else:
                                print("<> " + str(t__bench[4]))

                    if not ignore_noise_metrics:
                        if not self.args.noise_mask == 'none':
                            with logs_writer.as_default():
                                # Mean of the last collect_rate episodes, collected on every step
                                tf.summary.scalar("04_metrics/send_noise_power", t__noise[0, 1], step=episode_number)
                                tf.summary.scalar("04_metrics/send_signal_power", t__noise[0, 0], step=episode_number)
                                tf.summary.scalar("04_metrics/recv_noise_power", t__noise[1, 1], step=episode_number)
                                tf.summary.scalar("04_metrics/recv_signal_power", t__noise[1, 0], step=episode_number)
                                tf.summary.scalar("04_metrics/send_snr", t__noise[0, 0]/t__noise[0, 1], step=episode_number)
                                tf.summary.scalar("04_metrics/recv_snr", t__noise[1, 0]/t__noise[1, 1], step=episode_number)

                        if not self.args.noise_mask == 'all':
                            with logs_writer.as_default():
                                # Mean of the last collect_rate episodes, collected on every step
                                tf.summary.scalar("04_metrics/special_send_noise_power", t__noise_special[0, 1], step=episode_number)
                                tf.summary.scalar("04_metrics/special_send_signal_power", t__noise_special[0, 0], step=episode_number)
                                tf.summary.scalar("04_metrics/special_recv_noise_power", t__noise_special[1, 1], step=episode_number)
                                tf.summary.scalar("04_metrics/special_recv_signal_power", t__noise_special[1, 0], step=episode_number)
                                tf.summary.scalar("04_metrics/special_send_snr", t__noise_special[0, 0]/t__noise_special[0, 1], step=episode_number)
                                tf.summary.scalar("04_metrics/special_recv_snr", t__noise_special[1, 0]/t__noise_special[1, 1], step=episode_number)

                if not running or (self.args.num_episodes is not None and episode_number >= self.args.num_episodes and self.saving_enabled):
                    if not self.args.evaluate:
                        saver.save()
                    print("Training completed:\n\t steps: {}, episodes: {}, time: {}".format(
                        train_step+1, episode_number, round(time.time() - f_start, 3)))
                    running = False
                # End printing

                if self.args.noise_mask == 'random':
                    mask = tf.random.shuffle(mask)
                elif self.args.noise_mask == 'fixed':
                    pass

                obs_n = self.environment.reset()
                self.reset_loop()
                episode_info[0, episode_number % self.args.logs_range_collect] = 0

                #END IF TERMINAL

            # for displaying learned policies
            if self.args.display:
                time.sleep(self.args.render_time)
                self.environment.render()

                if not self.args.no_console:
                    print('Step: {0}/{1}, Reward: {2}'.format(
                        episode_step,
                        self.args.max_episode_len,
                        rew_n[0]))

                    if  episode_step == self.args.max_episode_len :
                        print('---------------------------')
                        print('Reward: {0}, Avg reward: {1}'.format(
                            np.nansum(episode_rewards[0]),
                            np.nanmean(episode_info[0])))
                        print('---------------------------')

            # increment global step counter
            tf_train_step.assign_add(1)
            train_step += 1

            if done or terminal:
                episode_step = 0

            #########
            # TRAIN #
            #########
            loss = None

            # TODO set max buffer size and inital buffer size as params
            if len(self.replay_buffer_n[0]) >= self.min_experiences_number \
                    and train_step % self.args.steps_per_train == 0:
                # Fill replay buffer

                update_number += 1

                for i, agent in enumerate(self.trainers):
                    exp = self.train_experience(self.replay_buffer_n[i])
                    if self.args.graph and not_graphed:
                        tf.summary.trace_on(graph=True)
                        loss = agent.update(self.trainers, exp)
                        with logs_writer.as_default():
                            tf.summary.trace_export(name='train_func', step=0)
                        not_graphed = False
                    else:
                        loss = agent.update(self.trainers, exp)
                    if loss is not None:
                        history[:, i] = loss

                if train_step % self.args.logs_rate_collect == 0 and self.saving_enabled:
                    metrics = np.mean(history, axis=-1)
                    with logs_writer.as_default():
                        tf.summary.scalar("02_train/q_loss", metrics[0], step=train_step)
                        tf.summary.scalar("02_train/p_loss", metrics[1], step=train_step)
                        tf.summary.scalar("02_train/target_q", metrics[2], step=train_step)
                        tf.summary.scalar("02_train/target_q_next", metrics[4], step=train_step)
                        tf.summary.scalar("02_train/target_q_std", metrics[5], step=train_step)
                        tf.summary.scalar("01_general/train_time", round(time.time() - p_start, 3), step=train_step)
                        p_start = time.time()
                    # logs_writer.flush()

            if self.args.profile:
                if episode_number >= 3000 and profiling == 0:
                    profiling = 1
                    profiler.start(logdir=save_profile_dir)

                if episode_number >= 3100 and profiling == 1:
                    profiling = 2
                    profiler.stop()

    @staticmethod
    def parser():
        parser = argparse.ArgumentParser("Reinforcement Learning experiments for multiagent environments")

        # Computing
        parser.add_argument("--GPU", action="store_true", default=False)
        parser.add_argument("--profile", action="store_true", default=False)
        parser.add_argument("--graph", action="store_true", default=False)

        # Environment
        parser.add_argument("--scenario", type=str, default="simple_spread_random", help="name of the scenario script")
        parser.add_argument("--max-episode-len", type=int, default=25, help="maximum episode length")
        parser.add_argument("--num-episodes", type=int, default=None, help="number of episodes")

        # Core training parameters
        parser.add_argument("--lr", type=float, default=1e-3, help="learning rate for Adam optimizer")
        parser.add_argument("--decaying-lr", type=float, default=None, help="how much decrease learning rate after patience is exceeded, None means no decay")
        parser.add_argument("--decaying-lr-patience", type=int, default=2000, help="how many episodes averaged to logs rangewith may be lower than best before lr decay")
        parser.add_argument("--decaying-lr-limit", type=float, default=None, help="what is the minimum learning rate using decay")

        parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
        parser.add_argument("--batch-size", type=int, default=512, help="number of episodes to optimize at the same time")
        parser.add_argument("--steps-per-train", type=int, default=400, help="number of environment steps after which one step of training is performed")

        # Checkpointing
        parser.add_argument("--config", type=str, default=None, help="path to configuration file")
        parser.add_argument("--exp-name", type=str, default="", help="name of the experiment")
        parser.add_argument("--save-dir", type=str, default="/tmp/policy/", help="directory in which training state and model should be saved")
        parser.add_argument("--save-rate", type=int, default=5000, help="save model once every time this many episodes are completed")
        parser.add_argument("--load-dir", type=str, default="", help="directory in which training state and model are loaded")

        # Evaluation
        parser.add_argument("--restore", action="store_true", default=False)
        parser.add_argument("--restore-best", action="store_true", default=False)

        parser.add_argument("--skip-best-for-episodes", type=int, default=2000)

        parser.add_argument("--display", action="store_true", default=False)
        parser.add_argument("--render-time", type=float, default=0.1)
        parser.add_argument("--evaluate", action="store_true", default=False)

        # Noise
        parser.add_argument("--noise-mask", type=str, default="all")
        parser.add_argument("--noise-mask-value", type=int, default=0, help="how many agents will not be influenced by the noise")

        # Inactive
        # parser.add_argument("--benchmark", action="store_true", default=False)
        # parser.add_argument("--benchmark-iters", type=int, default=100000, help="number of iterations run for benchmarking")
        # parser.add_argument("--benchmark-dir", type=str, default="./benchmark_files/", help="directory where benchmark data is saved")
        # parser.add_argument("--plots-dir", type=str, default="./learning_curves/", help="directory where plot data is saved")

        # Logs
        #parser.add_argument("--logs-dir", type=str, default="../logs_dir/", help="directory where logs for tensorboard are saved")
        parser.add_argument("--logs-rate-display", type=int, default=1000, help="how often log will be displayed")

        parser.add_argument("--logs-rate-collect", type=int, default=1000, help="how often logs will be collected")
        parser.add_argument("--metrics-rate-collect", type=int, default=2000, help="how often logs will be collected")
        parser.add_argument("--logs-range-collect", type=int, default=1000, help="how many instances will be used for counting averages")
        parser.add_argument("--no-console", action="store_true", default=False)

        return parser

    def get_env(self):
        return scenario_environment(self.args.scenario)

    def get_trainers(self):
        raise NotImplemented()

    def init_buffer(self):
        raise NotImplemented()

    def init_loop(self):
        raise NotImplemented()

    def reset_loop(self):
        raise NotImplemented()

    def collect_action(self, obs_n, mask):
        raise NotImplemented()

    def collect_metrics(self, obs_n, mask):
        raise NotImplemented()

    def collect_experience(self, obs_n, action_n, rew_n, new_obs_n, done_n, terminal, mask):
        raise NotImplemented()

    def train_experience(self, buffer):
        raise NotImplemented()
