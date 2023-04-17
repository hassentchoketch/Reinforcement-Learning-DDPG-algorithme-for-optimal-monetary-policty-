import matplotlib.pyplot as plt
import numpy as np
from utils import *


def plot_scores(scores,results_folder,name):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.savefig(f"{results_folder}\{name}.png")
    plt.show()


class plot_contrfactual_series:
    def __init__(self, data, agent, gap_model, inf_model, results_folder , i, add_noise=None):
        self.data = data
        self.agent = agent
        self.results_folder = results_folder
        self.i = i
        self.add_noise = add_noise
        self.gap_model = gap_model
        self.inf_model = inf_model

        self.get_state()
        # print("satat", self.states)
        self.policy()
        # print("action", self.actions)
        self.contrfactual()
        self.contrefactual_plot()

    def get_state(self):
        self.states = []
        for pi, g in zip(self.data["inf"], self.data["GDP_gap"]):
            self.states.append((pi, g))

    def policy(self):
        self.actions = []
        for state in self.states:
            action = self.agent.act(np.asarray(state), add_noise=self.add_noise)
            self.actions.append(action)

    def contrfactual(self):
        self.inf_ = []
        self.gap_ = []
        for i in range(2, len(self.states)):
            gap1 = self.gap_model.predict((1,self.states[i-1][1],
                                    self.states[i-1][0],
                                    self.actions[i-1],
                                    self.actions[i-2])) + self.gap_model.resid[i]
            inf1 = self.inf_model.predict((1,gap1,
                                      self.states[i-1][1],
                                      self.states[i - 2][1],
                                      self.states[i-1][0],
                                      self.states[i-2][0],
                                      self.actions[i-1])) + self.inf_model.resid[i]
            self.gap_.append(gap1)
            self.inf_.append(inf1)

    def contrefactual_plot(self):
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7), dpi=100)
        (l1,) = ax1.plot(
            self.data["ffr"][2:].index,
            np.array(self.actions[2:]).squeeze(-1),
            color="tab:Blue",
            lw=2,
        )
        (l2,) = ax1.plot(
            self.data["ffr"][2:].index,
            self.data["ffr"][2:],
            color="tab:Orange",
            # alpha=0.2,
        )
        ax1.set_ylabel("FFR")
        # -----------------------------------
        (l1,) = ax2.plot(
            self.data["inf"][2:].index,
            np.array(self.inf_).squeeze(-1),
            color="tab:Blue",
            lw=2,
        )
        (l2,) = ax2.plot(
            self.data["inf"][2:].index,
            self.data["inf"][2:],
            color="tab:Orange",
            # alpha=0.2,
        )
        ax2.set_ylabel("Inf")
        ax2.axhline(y=2, color="r", linestyle="-")
        # ---------------------------------------
        (l1,) = ax3.plot(
            self.data["GDP_gap"][2:].index,
            np.array(self.gap_).squeeze(-1),
            color="tab:Blue",
            lw=2,
        )
        (l2,) = ax3.plot(
            self.data["GDP_gap"][2:].index,
            self.data["GDP_gap"][2:],
            color="tab:Orange",
            # alpha=0.2,
        )
        ax3.set_ylabel("Output Gap")
        ax3.axhline(y=0, color="r", linestyle="-")
        # ----------------------------------------------
        plt.legend([l1, l2], ["RL", "Actual"])
        fig.tight_layout()
        plt.savefig(f"{self.results_folder}/counterfactual(best_agent)_{self.i}.png")
        plt.show()
