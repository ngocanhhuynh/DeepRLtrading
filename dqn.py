#adapt with modifications from https://github.com/88d52bdba0366127fffca9dfa93895/dqn/blob/master/dqn.py

import numpy as np
import pandas as pd
#import pandas_datareader.data as web
import datetime
from hmmlearn import hmm

import warnings
warnings.filterwarnings("ignore")

import os
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import tensorflow as tf
from collections import deque
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Convolution2D, Flatten, Dense, Dropout

#from keras import backend as K
#K.set_image_dim_ordering('th')

ENV_NAME = 'stockMarket'  # Environment name
FRAME_WIDTH = 50 - 1 + 2  # Resized frame width
FRAME_HEIGHT = 9  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
EXPLORATION_STEPS = 1000000  # Number of steps over which the initial value of epsilon is linearly annealed to its final value
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
INITIAL_REPLAY_SIZE = 20000 # Number of steps to populate the replay memory before training starts
NUM_REPLAY_MEMORY = 400000 # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
TARGET_UPDATE_INTERVAL = 10000  # The frequency with which the target network is updated
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 1e-3  #0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
MIN_GRAD = 0.01  # Constant added to the squared gradient in the denominator of the RMSProp update
SAVE_INTERVAL = 200000  # The frequency with which the network is saved
NO_OP_STEPS = 10  # Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
LOAD_NETWORK = True
TRAIN = False
SAVE_NETWORK_PATH = 'train/bmp/saved_networks/' + ENV_NAME
SAVE_SUMMARY_PATH = 'train/bmp/summary/' + ENV_NAME
NUM_EPISODES_AT_TEST = 1  # Number of episodes the agent plays at test time

class MarketEnv(gym.Env):

    def __init__(self, symbol = "bmp", fr_date = "2017-11-01", to_date = "2018-08-30", horizon = 90, max_loss = 1.):
        self.symbol = symbol
        self.fr_date = fr_date   
        self.to_date = to_date
        self.horizon = horizon
        self.max_loss = max_loss

        # Parameters used for summary
        self.total_reward = 0
        self.duration = 0

        self.actions = [0.2, 0.5, 0.7]                                         #percentage of risky assets
        self.nr_actions = len(self.actions)
        self.nr_hid_states = 3
        self.action_space = spaces.Discrete(len(self.actions))
        self.obs_space = spaces.Discrete(50 * self.nr_hid_states * len(self.actions))

        self.hidden_states, self.quotes = self._hmm()
        
    def _discretise(self, x):
        out = np.ones(x.shape)
        nr_of_points = 50
        start = np.min(x, axis=0)
        stop = np.max(x, axis=0) 
        if len(x.shape)==1:
            bins_array = np.linspace(start, stop, nr_of_points)
            out = np.digitize(x, bins_array)
        elif len(x.shape)>1:
            for i in range(x.shape[1]):
                bins_array = np.linspace(start[i], stop[i], nr_of_points)
                out[:,i] = np.digitize(x[:,i], bins_array)  
        return out
    
    def _crawl(self, fr_date, to_date, VN = True):                             #'yyyy-mm-dd'  #(year, month, day)
        if not VN:
            # get quotes from yahoo finance
            quotes = web.DataReader(self.symbol, "yahoo", datetime.date(*fr_date), datetime.date(*to_date))
        
        else:
            link = 'https://raw.githubusercontent.com/88d52bdba0366127fffca9dfa93895/vnstock-data/master/symbols/'
            df = pd.read_csv(link + self.symbol + '.csv')
            df = df[(df['date'] >= fr_date) & (df['date'] <= to_date)]
            df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'})
            df = df.set_index('date')
            quotes = df.sort_index(ascending=True)

        quotes['logPrice'] = quotes['Close'].map(lambda x: np.log(x))
        quotes['logReturn']= quotes['logPrice'].diff(1)
        quotes['logReturn_idx']= self._discretise(quotes['logReturn'])

        fracChange = (quotes['Close'] - quotes['Open'])/quotes['Open']
        fracHigh = (quotes['High'] - quotes['Open'])/quotes['Open']
        fracLow = (quotes['Open'] - quotes['Low'])/quotes['Open']
        hmm_inp = self._discretise(np.column_stack([fracChange, fracHigh, fracLow])) 

        return hmm_inp, quotes

    def _hmm(self):
        """
        =========================================================
        Hidden Markov Model with Gaussian emissions on stock data
        =========================================================
        """
        np.random.seed(0)

        n_components=self.nr_hid_states
        n_mix=4
        min_covar=0.001
        startprob_prior=1.0
        transmat_prior=1.0
        weights_prior=1.0
        means_prior=0.0
        means_weight=0.0
        covars_prior=None
        covars_weight=None
        algorithm='viterbi'
        covariance_type='diag'
        random_state=None
        n_iter=15
        tol=0.01
        verbose=False
        params='stmcw'
        init_params='stmcw'

        #model = hmm.GMMHMM(n_components=n_components, n_mix=n_mix, n_iter=n_iter, verbose=verbose)       
        model = hmm.GaussianHMM(n_components=n_components, n_iter=n_iter, verbose=verbose)      

        # Crawl training data
        hmm_train, _ = self._crawl(fr_date="2014-11-01", to_date="2017-11-01")
        model.fit(hmm_train)

        # Crawl testing data
        hmm_test, quotes = self._crawl(fr_date=self.fr_date, to_date=self.to_date)
        hidden_states = model.predict(hmm_test)

        return hidden_states, quotes

    def _step(self, pre_action: int, date: int, action: int):
        #hidden_states, quotes = self._hmm()
        self.duration += 1
        
        ret_idx = self.quotes['logReturn_idx'][date+1]
        hid_state_idx = self.hidden_states[date+1]
        
        #self.obs = ret_idx * self.nr_hid_states * self.nr_actions + \
                   #hid_state_idx * self.nr_actions + \
                   #pre_action
            
        col_idx = hid_state_idx * self.nr_actions + \
                  pre_action
        
        obs = np.zeros((50 - 1 + 2, self.nr_hid_states * self.nr_actions))
        obs[ret_idx, col_idx] = 1
        self.obs = obs
    
        self.reward = self.quotes['logReturn'][date+1] * self.actions[action]
        self.total_reward += self.reward

        if self.total_reward <= - self.max_loss or self.duration >= self.horizon:
            self.terminal = True

        self.info = {'date': self.quotes.index[date+1]}

        return self.obs, self.reward, self.terminal, self.info

    def _reset(self):
        self.obs = np.zeros((50 - 1 + 2, self.nr_hid_states * self.nr_actions))
        self.reward = 0
        self.total_reward = 0
        self.duration = 0
        self.terminal = False

        return self.obs

    def _render(self, mode='human', close=False):
        pass

    def _seed(self, seed = None):
        pass

    '''
    def _close(self):
        pass

    def _configure(self):
        pass
    '''

class Agent():
    def __init__(self, nr_actions):
        self.nr_actions = nr_actions
        self.epsilon = INITIAL_EPSILON
        self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
        self.t = 0

        # Parameters used for summary
        self.total_reward = 0
        self.total_q_max = 0
        self.total_loss = 0
        self.duration = 0
        self.episode = 0

        # Create replay memory
        self.replay_memory = deque()

        # Create q network
        self.s, self.q_values, q_network = self.build_network()
        q_network_weights = q_network.trainable_weights

        # Create target network
        self.st, self.target_q_values, target_network = self.build_network()
        target_network_weights = target_network.trainable_weights

        # Define target network update operation
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) for i in range(len(target_network_weights))]

        # Define loss and gradient update operation
        self.a, self.y, self.loss, self.grads_update = self.build_training_op(q_network_weights)
        self.sess = tf.InteractiveSession()
        self.saver = tf.train.Saver(q_network_weights)
        self.summary_placeholders, self.update_ops, self.summary_op = self.setup_summary()
        self.summary_writer = tf.summary.FileWriter(SAVE_SUMMARY_PATH, self.sess.graph)

        if not os.path.exists(SAVE_NETWORK_PATH):
            os.makedirs(SAVE_NETWORK_PATH)

        self.sess.run(tf.global_variables_initializer())

        # Load network
        if LOAD_NETWORK:
            self.load_network()

        # Initialize target network
        self.sess.run(self.update_target_network)

    def build_network(self):
        model = Sequential()
        model.add(Convolution2D(32, 2, 2, subsample=(1, 1), activation='relu', input_shape=(STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT)))
        model.add(Convolution2D(64, 2, 2, subsample=(1, 1), activation='relu'))
        model.add(Convolution2D(64, 2, 2, subsample=(1, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.nr_actions))

        s = tf.placeholder(tf.float32, [None, STATE_LENGTH, FRAME_WIDTH, FRAME_HEIGHT])
        q_values = model(s)

        return s, q_values, model

    def build_training_op(self, q_network_weights):
        a = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])

        # Convert action to one hot vector
        a_one_hot = tf.one_hot(a, self.nr_actions, 1.0, 0.0)
        q_value = tf.reduce_sum(tf.multiply(self.q_values, a_one_hot), reduction_indices=1)

        # Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
        error = tf.abs(y - q_value)
        quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
        linear_part = error - quadratic_part
        loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        grads_update = optimizer.minimize(loss, var_list=q_network_weights)

        return a, y, loss, grads_update

    def get_initial_state(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        processed_observation = np.uint8(resize(processed_observation, (FRAME_WIDTH, FRAME_HEIGHT)))
        state = [observation for _ in range(STATE_LENGTH)]
        return np.stack(state, axis=0)

    def get_action(self, state):
        if self.epsilon >= random.random() or self.t < INITIAL_REPLAY_SIZE:
            action = random.randrange(self.nr_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        # Anneal epsilon linearly over time
        if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
            self.epsilon -= self.epsilon_step

        return action

    def run(self, state, action, reward, terminal, observation):
        next_state = np.append(state[1:, :, :], [observation], axis=0)
        
        # Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
        reward = np.clip(reward, -1, 1)

        # Store transition in replay memory
        self.replay_memory.append((state, action, reward, next_state, terminal))
        if len(self.replay_memory) > NUM_REPLAY_MEMORY:
            self.replay_memory.popleft()

        if self.t >= INITIAL_REPLAY_SIZE:
            # Train network
            if self.t % TRAIN_INTERVAL == 0:
                self.train_network()

            # Update target network
            if self.t % TARGET_UPDATE_INTERVAL == 0:
                self.sess.run(self.update_target_network)

            # Save network
            if self.t % SAVE_INTERVAL == 0:
                save_path = self.saver.save(self.sess, SAVE_NETWORK_PATH + '/' + ENV_NAME, global_step=self.t)
                print('Successfully saved: ' + save_path)

        self.total_reward += reward
        self.total_q_max += np.max(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))
        self.duration += 1

        if terminal:
            # Write summary
            if self.t >= INITIAL_REPLAY_SIZE:
                stats = [self.total_reward, self.total_q_max / float(self.duration),
                        self.duration, self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL))]
                for i in range(len(stats)):
                    self.sess.run(self.update_ops[i], feed_dict={
                        self.summary_placeholders[i]: float(stats[i])
                    })
                summary_str = self.sess.run(self.summary_op)
                self.summary_writer.add_summary(summary_str, self.episode + 1)

            # Debug
            if self.t < INITIAL_REPLAY_SIZE:
                mode = 'random'
            elif INITIAL_REPLAY_SIZE <= self.t < INITIAL_REPLAY_SIZE + EXPLORATION_STEPS:
                mode = 'explore'
            else:
                mode = 'exploit'
            print('EPISODE: {0:6d} / TIMESTEP: {1:8d} / DURATION: {2:5d} / EPSILON: {3:.5f} / TOTAL_REWARD: {4:3.6f} / AVG_MAX_Q: {5:2.4f} / AVG_LOSS: {6:.5f} / MODE: {7}'.format(
                self.episode + 1, self.t, self.duration, self.epsilon,
                self.total_reward, self.total_q_max / float(self.duration),
                self.total_loss / (float(self.duration) / float(TRAIN_INTERVAL)), mode))

            self.total_reward = 0
            self.total_q_max = 0
            self.total_loss = 0
            self.duration = 0
            self.episode += 1

        self.t += 1

        return next_state

    def train_network(self):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        terminal_batch = []
        y_batch = []

        # Sample random minibatch of transition from replay memory
        minibatch = random.sample(self.replay_memory, BATCH_SIZE)
        for data in minibatch:
            state_batch.append(data[0])
            action_batch.append(data[1])
            reward_batch.append(data[2])
            next_state_batch.append(data[3])
            terminal_batch.append(data[4])

        # Convert True to 1, False to 0
        terminal_batch = np.array(terminal_batch) + 0

        target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch))})
        y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

        loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
            self.s: np.float32(np.array(state_batch)),
            self.a: action_batch,
            self.y: y_batch
        })

        self.total_loss += loss

    def setup_summary(self):
        episode_total_reward = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/TotalReward/Episode', episode_total_reward)
        episode_avg_max_q = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/AverageMaxQ/Episode', episode_avg_max_q)
        episode_duration = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/Duration/Episode', episode_duration)
        episode_avg_loss = tf.Variable(0.)
        tf.summary.scalar(ENV_NAME + '/AverageLoss/Episode', episode_avg_loss)
        summary_vars = [episode_total_reward, episode_avg_max_q, episode_duration, episode_avg_loss]
        summary_placeholders = [tf.placeholder(tf.float32) for _ in range(len(summary_vars))]
        update_ops = [summary_vars[i].assign(summary_placeholders[i]) for i in range(len(summary_vars))]
        summary_op = tf.summary.merge_all()
        return summary_placeholders, update_ops, summary_op

    def load_network(self):
        checkpoint = tf.train.get_checkpoint_state(SAVE_NETWORK_PATH)
        if checkpoint and checkpoint.model_checkpoint_path:
            self.saver.restore(self.sess, checkpoint.model_checkpoint_path)
            print('Successfully loaded: ' + checkpoint.model_checkpoint_path)
        else:
            print('Training new network...')

    def get_action_at_test(self, state):
        if random.random() <= 0.05:
            action = random.randrange(self.nr_actions)
        else:
            action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        # action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state)]}))

        self.t += 1

        return action


def preprocess(observation, last_observation):
    processed_observation = np.maximum(observation, last_observation)
    processed_observation = np.uint8(resize(processed_observation, (FRAME_WIDTH, FRAME_HEIGHT)))
    return np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))

def plot_results(symbol, date, price, ret, total_ret, action):
    fig = plt.figure(figsize=(12,10))
    fig.suptitle('Test Trading Performance on stock %s\
\n from %s to %s' % (symbol.upper(), date[0], date[-1]))

    ax1 = plt.subplot(411)
    plt.plot(date, price)
    plt.grid()
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.set_ylabel('close_price')

    ax2 = plt.subplot(412, sharex=ax1)
    plt.plot(date, ret)
    plt.grid()
    plt.setp(ax2.get_xticklabels(), visible=False)
    ax2.set_ylabel('log_return')

    ax3 = plt.subplot(413, sharex=ax1)
    plt.plot(date, total_ret)
    plt.grid()
    plt.setp(ax3.get_xticklabels(), visible=False)
    ax3.set_ylabel('total_reward')

    ax4 = plt.subplot(414, sharex=ax1)
    plt.plot(date, action)
    plt.grid()
    ax4.set_ylabel('action')

    plt.xticks(fontsize=8, rotation=90)
    plt.show()

def main():
    env = MarketEnv()
    agent = Agent(nr_actions=env.action_space.n)

    if TRAIN:  # Train mode
        for e in range(NUM_EPISODES):
            terminal = False
            observation = env._reset()
        
            for date in range(random.randint(1, NO_OP_STEPS)):
                if date == 0: previous_action = random.randrange(env.action_space.n)  
                last_observation = observation       
                action = random.randrange(env.action_space.n)
                observation, _, _, _ = env._step(previous_action, date, action)  # random action
                previous_action = action

            state = agent.get_initial_state(observation, last_observation)
            date = 0 
            env.duration = 0
            env.total_reward = 0
            
            while not terminal:
                date += 1
                #last_observation = observation
                action = agent.get_action(state)
                observation, reward, terminal, _ = env._step(previous_action, date, action)
                previous_action = action
                #processed_observation = preprocess(observation, last_observation)
                state = agent.run(state, action, reward, terminal, observation)

        print('Finish training!')
        print('Run `tensorboard --logdir=train/summary/` to see the results.')
        

    else:  # Test mode
        #env.monitor.start(ENV_NAME + '-test')
        for e in range(NUM_EPISODES_AT_TEST):
            daily_ret = {}
            daily_total_ret = {}
            daily_act = {}
            terminal = False
            observation = env._reset()
              
            for date in range(env.horizon, random.randint(env.horizon+1, env.horizon+NO_OP_STEPS)):
                if date == env.horizon: previous_action = random.randrange(env.action_space.n)  
                last_observation = observation       
                action = random.randrange(env.action_space.n)
                observation, _, _, _ = env._step(previous_action, date, action)  # random action
                previous_action = action

            state = agent.get_initial_state(observation, last_observation)
            date = env.horizon
            env.duration = 0
            env.total_reward = 0

            while not terminal:
                date += 1
                #last_observation = observation
                action = agent.get_action_at_test(state)
                observation, reward, terminal, info = env._step(previous_action, date, action)

                daily_ret[info['date']] = reward
                daily_total_ret[info['date']] = env.total_reward
                daily_act[info['date']] = action

                state = np.append(state[1:, :, :], [observation], axis=0)
                previous_action = action
                #processed_observation = preprocess(observation, last_observation)
                state = np.append(state[1:, :, :], [observation], axis=0)

            #visualise
            keys, ret = zip(*daily_ret.items())
            keys, total_ret = zip(*daily_total_ret.items())
            keys, act = zip(*daily_act.items())
            price = env.quotes.loc[keys,['Close']]

            benchmark = np.log(price.iloc[-1] / price.iloc[0])[0]
            difference = np.abs((env.total_reward - benchmark)/benchmark)
            
            print('SYMBOL:{0} / FR_DATE: {1} / TO_DATE:{2} / DURATION:{3:5d} / TOTAL_REWARD:{4:3.6f} / BENCHMARK:{5:3.6f}'.format(
                env.symbol, keys[0], keys[-1], env.duration, env.total_reward, benchmark))

            # print('{0} & {1} & {2} & {3:5d} & {4:3.6f} & {5:3.6f} & {6:3.6f}\\'.format(
                # env.symbol.upper(), keys[0], keys[-1], env.duration, env.total_reward, benchmark, difference))

            plot_results(env.symbol, keys, price, ret, total_ret, act)

        #env.monitor.close()
    
          
if __name__ == '__main__':
    main()