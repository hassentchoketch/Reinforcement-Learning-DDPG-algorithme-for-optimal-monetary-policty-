import sys
sys.path.append(r'c:\users\hassen\miniconda3\envs\r_learning\lib\site-packages')

import numpy as np
import gym 
from gym.utils import seeding


class Artificial_Economy_Env(gym.Env):
    ''' 
    
    # Description
    -------------
    
    - This environment class refers to a model that predicts the behavior of the economy excluding the central bank. 
    - This model uses two key variables, inflation and the output gap, to approximate the state of the economy.
    - The model is based on the New Keynesian framework developed by Rotemberg and Woodford in 1997.
    - The environment is used to simulate how the economy will respond to changes in monetary policy .
     
     
       - π :   inflation rate
       - gap:  GDP gap
       - i :   policy rate( nominal interest rate)
       - π_p:  inflation rate in (t-1)
       - gap_p: GDP gap in (t-1)
       
       - goal_π:   inflation target 
       - goal_gap: GDP gap target
       
       
    ### Action Space
    ----------------
    
    The action is a `ndarray` with shape `(1,)` representing the policy rate applied from the central bank to stabilize the ecomomy.
    
    | Num |   Action    |   Min  |   Max  |
    |-----|-------------|--------|--------|
    | 0   | policy rate | min(i) | max(i) |
    
    
    ### Observation Space
    ---------------------
    
    The observation is a `ndarray` with shape `(5,)` representing the model variables.
    
    | Num | Observation |     Min    |     Max    |
    |-----|-------------|------------|------------|
    | 0   |       π     | min(π)     | max(π)     |
    | 1   |       gap   | min(gap)   | max(gap)   |
    | 2   |       i     | min(i)     | max(i)     |
    | 3   |       π_p   | min(π_p)   | max(π_p)   |
    | 4   |       gap_p | min(gap_p) | max(gap_p) |

    ### Rewards
    -----------
    
    The reward function is defined as:
    
    r = -(omega_π*(π(t+1)-goal_π)**2 + omega_y*(gap(t+1)-goal_gap)**2)
    
    while omega_π = omega_y = 0.5.
    The maximum reward is zero ( the economy is reache its steady state level )
    
    
    ### Starting State
    ------------------
    
    The starting state is randomly drawing π0 gap0 i0 π(-1) gap(-1) from the data.
    
    ### Arguments
    --------------
    data : DataFrame contain historical data used to initialize state.
    
    '''
    def __init__(self, data):
        
        # Initialize the environment with default values
        self.__version__ = "0.0.1"
        self.criteria = 0.3
        self.goal_π = 2
        self.goal_gap = 0

        # set initial observations
        self.inf , self.gdp_gap, self.int = (data["inf"],data["GDP_gap"],data["ffr"])

        self._max_episode_steps = None
        self._current_step = 0
        

        self.seed()
        self.reset()

    def seed(self, seed=None):
        # Seeds the random number generator used in the environment
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """
        Resets the environment to a new initial state.
    
        Returns:
        --------
        state : ndarray
        """
        
        # Sets the environment to its initial state
        index = np.random.choice(len(self.inf))
        self.π   = self.inf[index]
        self.gap = self.gdp_gap[index]
        self.i   = self.int[index]
        self.π_p = self.inf[index - 1]
        self.gap_p = self.gdp_gap[index - 1]
        
        # Returns the current state of the environment
        self.state = np.array([self.π , self.gap, self.i, self.π_p, self.gap_p],dtype = np.float64)
        self._current_step = 0
        return self.state

    def step(self, action):
        """
        Perform a single time step of the environment given the specified action.

        Args:
        action (float): The action to take in the environment.

        Returns:
        observation (ndarray): An observation of the environment's current state.
        reward (float): The reward obtained from the action.
        done (bool): Whether the episode has ended.
        """
        
        # Executes the specified action in the environment
        self.π,self.gap,self.i_p,self.π_p,self.gap_p = self.state
        self._execute_action(action)
        self._transition_to_next_state()
        self._done_()
        self._get_reward()
        self.state = np.array([self.π1, self.gap1, self.i, self.π, self.gap],dtype = np.float64)
        
        # Returns the observation, reward, and done status after taking the specified action
        return self.state, self.reward, self.done, {}

    def _execute_action(self, action):
        """ Setting the value of action within the range of intrest rate """
        self.i  = np.clip(action, min(self.int), max(self.int))[0]
        
    def _transition_to_next_state(self):
        """Executes the specified action """
        eps_gap = np.random.normal(0, np.sqrt(0.2108))
        eps_pi = np.random.normal(0, np.sqrt(0.0330))

        self.gap1 = (
            0.3834
            + 0.9084 * self.gap
            - 0.1437 * self.π
            + 0.2726 * self.i
            - 0.2896 * self.i_p
            + eps_gap
        )
        self.π1 = (
            0.1035
            - 0.0655 * self.gap1
            + 0.1970 * self.gap
            - 0.1121 * self.gap_p
            + 1.297 * self.π
            - 0.3116 * self.π_p
            - 0.0122 * self.i
            + eps_pi
        )

    def _done_(self):
        """
        Check if the episode is done based on the current state and maximum number of steps.
        
        Returns:
        --------
        done (bool): True if the episode is done, False otherwise.
        
        """
        
        if self._max_episode_steps is not None:
            if self._current_step <= self._max_episode_steps:
                self.done = bool(
                    np.abs(self.π1 - self.goal_π) < self.criteria
                    and np.abs(self.gap1 - self.goal_gap) < self.criteria
                )
                self._current_step += 1
            else:
                self.done = True
        else:
            self.done = bool(
                np.abs(self.π1 - self.goal_π) < self.criteria
                and np.abs(self.gap1 - self.goal_gap) < self.criteria
            )

    def _get_reward(self):
        """Calculates the reward for the current state of the environment based on the deviations from the target values.

        Returns:
        float: The reward for the current state.
        """
        
        # Deviations from target
        π_div = (self.π1 - self.goal_π) ** 2
        g_div = (self.gap1 - self.goal_gap) ** 2

        # Punishe deviations from target that exceed 2 percentage points
        reward_pi = -(10 * π_div) if π_div > 4 else 0
        reward_g = -(10 * g_div) if g_div > 4 else 0

        # Total reward
        reward = -0.5 * π_div - 0.5 * g_div

        self.reward = reward + reward_g + reward_pi
