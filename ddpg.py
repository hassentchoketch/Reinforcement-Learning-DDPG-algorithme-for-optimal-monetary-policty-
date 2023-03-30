import numpy as np
import os
from utils import *
from plots import *


class DDPG:
    """ A class for training a reinforcement learning agent using the Deep Deterministic Policy Gradient (DDPG) algorithm.
   
     Attributes:
     ----------
     - env: The environment for the agent to interact with.
     - data: The dataset from with we initialize the state.
     - n_episodes: The number of episodes to train the agent for
     - agent: The reinforcement learning agent to train
     - print_every: The interval at which to print progress updates during training
     - results_folder: The folder to save training results in"""
   
    def __init__(self, env, data, n_episodes, agent, print_every, results_folder):
        """Initializes a DDPG instance with the given parameters and calls the `train` method to start training the agent."""
        self.env = env
        self.data = data
        self.n_episodes = n_episodes
        self.agent = agent
        self.print_every = print_every
        self.results_folder = results_folder

        self.train()

    def train(self):
        """Trains the reinforcement learning agent using the DDPG algorithm. ."""
        scores = []
        saved_agents_scores = []
        self.saved_agent = []
        self.ss_max = None
        # loop over number of episods 
        for self.i_episode in range(1, self.n_episodes + 1):
            done = False
            state = self.env.reset()
            self.agent.reset()
            self.score = 0
            self.steps = 0
            while not done:
                action = self.agent.act(state[:2], add_noise=True)               
                next_state, reward, done, _ = self.env.step(action)
                self.agent.step(state[:2], action, reward, next_state[:2], done)
                state = next_state
                self.score += reward.squeeze(0)
                self.steps += 1
                if done:
                    break
            print(f"\rEpisode{self.i_episode}\tAverage Score:{self.score}\tSteps:{self.steps}\tmean Score:{self.score/self.steps}",end="")  # np.mean(scores_deque)
            scores.append(self.score)
            
            # save all agents during training that fulfill the following criteria : steps > 1 and (self.score / self.steps) > -4.
            if self.steps > 1 and (self.score / self.steps) > -4:
                self._steady_state_calculation()
                saved_agents_scores.append(self.score)
                
        # select the agent with the best steady state reward    
        self._optimal_agent()
        
        # print scours of all agens   
        plot_scores(scores, self.results_folder,'scores')
        
        # print scours of saved agens
        plot_scores(saved_agents_scores, self.results_folder,'saved_agents__scores')

    def _steady_state_calculation(self):
        
        states = get_states(self.data)
        actions = agent_policy(states, self.agent)
        self.inf_, self.gap_ = contrfactual_simulation(states, actions)
        self._ss_reward()
        self._save_best_agents()

    def _ss_reward(self):
        inf_div = []
        gap_div = []
        for p, g in zip(self.inf_, self.gap_):
            inf_div.append((p - self.env.goal_π) ** 2)
            gap_div.append((g) ** 2)
        self.inf_loss = np.mean(inf_div)
        self.gap_loss = np.mean(gap_div)
        self.ss_reward = 0.5 * (self.inf_loss) + 0.5 * (self.gap_loss)

    def _save_best_agents(self):
        if self.ss_max is None or self.ss_reward < self.ss_max:
            self.ss_max = self.ss_reward
            print(
                f"\rEpisode{self.i_episode}\tAverage Score:{self.score}\tSteps:{self.steps}\ inf_loss:{self.inf_loss}\gap_loss:{self.gap_loss}\ss_reward: {self.ss_reward}"
            )
            save_agent(self.agent, self.results_folder, self.i_episode)
            self.saved_agent.append(self.i_episode)

    def _optimal_agent(self):
        for i in self.saved_agent:
            if i == max(self.saved_agent):
                os.rename(
                    f"{self.results_folder}/checkpoint_actor_{i}.pth",
                    f"{self.results_folder}/checkpoint_actor_.pth",
                )
                os.rename(
                    f"{self.results_folder}/checkpoint_critic_{i}.pth",
                    f"{self.results_folder}/checkpoint_critic_.pth",
                )
            else:
                os.remove(f"{self.results_folder}/checkpoint_actor_{i}.pth")
                os.remove(f"{self.results_folder}/checkpoint_critic_{i}.pth")
