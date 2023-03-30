from environnement import Artificial_Economy_Env
from Agent import Agent
from ddpg import DDPG
from plots import *
from utils import *
import torch

torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
        file_path = '__main__'
        path = "DDPG_US_DATA_.csv"
        results_folder = create_subdirectory(file_path, "output")
        data = load_data(path)
        
        env = Artificial_Economy_Env(data)
        env.seed(2)
        env._max_episode_steps = 50
        
        
        # Agent paramaters
        state_size = 2
        action_size = 1
        random_seed = 2
        
        # ddpg parameters
        n_episodes = 500
        print_every = 100

        
        agent = Agent(state_size, action_size, random_seed, device)


        DDPG(env, data, n_episodes, agent, print_every, results_folder)



        load_agent(agent, results_folder)
        plot_contrfactual_series(data, agent, results_folder, i="(train_noise)", add_noise=True)
        
           
if __name__ == '__main__':
    main()