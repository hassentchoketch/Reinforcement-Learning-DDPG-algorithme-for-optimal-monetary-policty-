from environnement import Artificial_Economy_Env
from Agent import Agent
from ddpg import DDPG
from plots import *
from utils import *
import torch
import statsmodels.api as sm


torch.set_default_dtype(torch.float64)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
        file_path = '__main__'
        path = "DDPG_US_DATA_.csv"
        results_folder = create_subdirectory(file_path, "output")
        data = load_data(path)
        # print(data.head())
        # print(len(data))
        data = data.drop(['Unnamed: 6','GDP', 'GDP_P'],axis = 1)
        # print(data.head())
        # print(len(data))
        for column in data.columns:
                data[column + '_1'] = data[column].shift(1)
                data[column + '_2'] = data[column].shift(2)
        data = data[2:]
        

        # Create an instance of MinMaxScaler
        # scaler = MinMaxScaler()
        # data = scaler.fit_transform(data_0)
        # Convert the scaled data back to a DataFrame
        # data = pd.DataFrame(data, columns=data_0.columns)
        # print(data.head())
        # print(len(data))
        
        
        X_inf = data[['GDP_gap','GDP_gap_1','GDP_gap_2','inf_1','inf_2','ffr_1']]
        X_inf = sm.add_constant(X_inf)
        y_inf = data['inf']
        inf_model = sm.OLS(y_inf, X_inf).fit()
        # print(inf_model.params)
        # print(inf_model.resid)
        # print(len(inf_model.resid))
        X_gap = data[['GDP_gap_1','inf_1','ffr_1','ffr_2']]
        X_gap = sm.add_constant(X_gap)
        y_gap = data['GDP_gap']
        gap_model = sm.OLS(y_gap, X_gap).fit()
        # print(gap_model.params)
        # print(gap_model.resid)
        # print(len(gap_model.resid))

        criteria =0.3
        env = Artificial_Economy_Env(data,gap_model,inf_model,criteria)
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

        DDPG(env, data, n_episodes, agent, print_every, results_folder,gap_model,inf_model)

        # load_agent(agent, results_folder)
        
        plot_contrfactual_series(data, agent, gap_model, inf_model, results_folder, i="(train_noise)", add_noise=True)
        
        plot_contrfactual_series(data, agent, gap_model, inf_model, results_folder, i="(train_noise)", add_noise=False)
if __name__ == '__main__':
    main()