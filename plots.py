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
    def __init__(self, data, agent, results_folder, i, add_noise=None):
        self.data = data
        self.agent = agent
        self.results_folder = results_folder
        self.i = i
        self.add_noise = add_noise

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
        eps_gap = np.random.normal(0, np.sqrt(0.2108))
        eps_pi = np.random.normal(0, np.sqrt(0.0330))
        for i in range(1, len(self.states)):
            gap1 = (
                0.3834
                + 0.9084 * self.states[i][1]
                - 0.1437 * self.states[i][0]
                + 0.2726 * self.actions[i]
                - 0.2896 * self.actions[i - 1]
                + eps_gap
            )
            inf1 = (
                0.1035
                - 0.0655 * gap1
                + 0.1970 * self.states[i][1]
                - 0.1121 * self.states[i - 1][1]
                + 1.297 * self.states[i][0]
                - 0.3116 * self.states[i - 1][0]
                - 0.0122 * self.actions[i]
                + eps_pi
            )
            self.gap_.append(gap1)
            self.inf_.append(inf1)

    def contrefactual_plot(self):
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 7), dpi=100)
        (l1,) = ax1.plot(
            self.data["ffr"].index,
            np.array(self.actions).squeeze(-1),
            color="tab:Blue",
            lw=2,
        )
        (l2,) = ax1.plot(
            self.data["ffr"].index,
            self.data["ffr"],
            color="tab:Orange",
            # alpha=0.2,
        )
        ax1.set_ylabel("FFR")
        # -----------------------------------
        (l1,) = ax2.plot(
            self.data["inf"][1:].index,
            np.array(self.inf_).squeeze(-1),
            color="tab:Blue",
            lw=2,
        )
        (l2,) = ax2.plot(
            self.data["inf"][1:].index,
            self.data["inf"][1:],
            color="tab:Orange",
            # alpha=0.2,
        )
        ax2.set_ylabel("Inf")
        ax2.axhline(y=2, color="r", linestyle="-")
        # ---------------------------------------
        (l1,) = ax3.plot(
            self.data["GDP_gap"][1:].index,
            np.array(self.gap_).squeeze(-1),
            color="tab:Blue",
            lw=2,
        )
        (l2,) = ax3.plot(
            self.data["GDP_gap"][1:].index,
            self.data["GDP_gap"][1:],
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
