import pandas as pd
import numpy as np
import os
import torch
import glob


def create_subdirectory(directory_path: str, subdirectory_name: str) -> str:
    """Create a subdirectory in the given directory.

    If the subdirectory already exists, delete all files in it. If it doesn't
    exist, create it.

    Args:
        directory_path (str): The path of the directory where the
            subdirectory should be created.
        subdirectory_name (str): The name of the subdirectory to create.

    Returns:
        str: The path of the created subdirectory.

    Raises:
        OSError: If there is an error creating the subdirectory.
    """
    # Get the directory of the file path
    script_dir = os.path.dirname(directory_path)

    # Create the path for the subdirectory
    subdirectory_path = os.path.join(script_dir, subdirectory_name)

    # Check if the subdirectory already exists
    if os.path.isdir(subdirectory_path):
        # If it does, delete all files in the subdirectory
        files = glob.glob(subdirectory_path + "/*")
        for f in files:
            os.remove(f)
    else:
        # If it doesn't, create the subdirectory
        os.makedirs(subdirectory_path)

    # Return the path of the created subdirectory
    return subdirectory_path

def load_data(path: str) -> pd.DataFrame:
    """Load data from a CSV file and convert the Date column to a datetime index.

    Args:
        path (str): The path of the CSV file to load.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded data with the Date column
            converted to a datetime index.
    """
    # Load the data from the CSV file Convert the Date column to a datetime index
    data = pd.read_csv(path, parse_dates=["Date"], index_col="Date", infer_datetime_format=True)

    return data

def save_agent(agent: object, results_folder: str, episode_num: int) -> None:
    """Save the trained actor and critic networks of the agent.

    Args:
        agent (object) : The agent whose networks should be saved.
        results_folder (str): The path of the directory where the network weights
            should be saved.
        episode_num (int): The episode number to use in the saved file names.
    """
    # Save the actor network weights to a file
    actor_file = f"{results_folder}/checkpoint_actor_{episode_num}.pth"
    torch.save(agent.actor_local.state_dict(), actor_file)

    # Save the critic network weights to a file
    critic_file = f"{results_folder}/checkpoint_critic_{episode_num}.pth"
    torch.save(agent.critic_local.state_dict(), critic_file)

def load_agent(agent: object, results_folder: str) -> None:
    """Load the saved actor and critic networks of the trained agent.

    Args:
        agent: The agent whose networks should be updated with
            the saved weights.
        results_folder (str): The path of the directory where the network weights are
            saved.

    Returns:
        None
    """
    # Load the actor network weights from the saved file
    actor_file = f"{results_folder}/checkpoint_actor_.pth"
    agent.actor_local.load_state_dict(torch.load(actor_file))

    # Load the critic network weights from the saved file
    critic_file = f"{results_folder}/checkpoint_critic_.pth"
    agent.critic_local.load_state_dict(torch.load(critic_file))

def get_states(data: pd.DataFrame) -> list[tuple[float, float]]:
    """Extract states information from a pandas DataFrame.

    Args:
        data (pd.DataFrame): A DataFrame containing columns "inf" and "GDP_gap".

    Returns:
        A list of state tuples. Each tuple contains two float values: the value of the "inf"
        column and the value of the "GDP_gap" column, in that order.
    """
    states = []
    for pi, g in zip(data["inf"], data["GDP_gap"]):
        state = (pi, g)
        states.append(state)
    return states

def agent_policy(states: list[tuple[float, float]], agent: object) -> list[float]:
    """Determine the agent's recommended actions for a given list of states.

    Args:
        states (List[Tuple[float, float]]): A list of tuples representing the states of the
            environment. 
        agent (object): A reinforcement learning agent object with an "act" method that takes
            a state as input and returns an action.

    Returns:
        A list of float values representing the agent's recommended actions for each state in
        the input list.
    """
    actions = []
    for state in states:
        # Get the agent's recommended action for the current state
        action = agent.act(np.asarray(state), add_noise=True)
        actions.append(action)
    return actions

def contrfactual_simulation(states: list[tuple[float, float]], actions: list[float],gap_model,inf_model)-> list[float]:
    """Simulate a counterfactual scenario using the given states and actions.

    Args:
        states (list): A list of tuples, where each tuple contains the values of inflation and GDP_gap for a given time step.
        actions (list): A list of actions taken by an agent in response to the states.

    Returns:
        A tuple (contrfactual_inf, contrfactual_gap) containing the simulated values of inflation and GDP_gap.
    """
    contrfactual_inf = []
    contrfactual_gap = []

    # Calculate the noise terms for the simulation
    # eps_gap = np.random.normal(0, np.sqrt(0.2108))
    # eps_pi = np.random.normal(0, np.sqrt(0.0330))

    # Simulate the counterfactual scenario for each time step
    for i in range(2, len(states)):
        gap1 = gap_model.predict((1,states[i-1][1],
                                    states[i-1][0],
                                    actions[i-1],
                                    actions[i-2])) + gap_model.resid[i]
       
        inf1 = inf_model.predict((1,gap1,
                                      states[i-1][1],
                                      states[i - 2][1],
                                      states[i-1][0],
                                      states[i-2][0],
                                      actions[i-1])) + inf_model.resid[i]
        # gap1 = (
        #     0.3834
        #     + 0.9084 * states[i][1]
        #     - 0.1437 * states[i][0]
        #     + 0.2726 * actions[i]
        #     - 0.2896 * actions[i - 1]
        #     + eps_gap
        # )
        # inf1 = (
        #     0.1035
        #     - 0.0655 * gap1
        #     + 0.1970 * states[i][1]
        #     - 0.1121 * states[i - 1][1]
        #     + 1.297 * states[i][0]
        #     - 0.3116 * states[i - 1][0]
        #     - 0.0122 * actions[i]
        #     + eps_pi
        # )
        contrfactual_gap.append(gap1)
        contrfactual_inf.append(inf1)

    return contrfactual_inf, contrfactual_gap







def scaling_action(OldMax, OldMin, NewMax, NewMin, OldValue):
    OldRange = OldMax - OldMin
    NewRange = NewMax - NewMin
    new = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    return new

def state_dict(agent):

    print("Model's state_dict:")
    for param_tensor in agent.actor_local.state_dict():
        print(param_tensor, "\t", agent.actor_local.state_dict()[param_tensor])

    for param_tensor in agent.critic_local.state_dict():
        print(param_tensor, "\t", agent.critic_local.state_dict()[param_tensor])
