"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import sys
import typing as t

import gymnasium as gym
import numpy as np

from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n  # type: ignore


#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.5, epsilon=0.1, gamma=0.99, legal_actions=list(range(n_actions))
)


def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e4)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()

    for k in range(t_max):
        # Get agent to pick action given state s
        a = agent.get_action(s)

        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        # BEGIN SOLUTION
        total_reward += r
        if done:
            agent.set_qvalue(
                s,
                a,
                agent.get_qvalue(s, a)
                + agent.learning_rate * (r - agent.get_qvalue(s, a)),
            )
            break
        else:
            agent.update(s, a, r, next_s)
            s = next_s
        # END SOLUTION

    return total_reward


rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent, t_max=200))
    if i % 100 == 0:
        m = np.mean(rewards[-100:])
        print("mean reward", m)
        if m > 0:
            break

assert np.mean(rewards[-100:]) > 0.0


def play_exp(env: gym.Env, agent: QLearningAgent, t_max=200):
    frames = []
    s, _ = env.reset()
    for k in range(t_max):
        frames.append(env.render())
        a = agent.get_action(s)
        next_s, r, done, _, _ = env.step(a)
        s = next_s
        if done:
            break
    frames.append(env.render())
    return frames


from gymnasium.utils.save_video import save_video

frames = play_exp(env, agent)
save_video(
    frames=frames,
    video_folder="videos",
    fps=env.metadata["render_fps"],
    name_prefix="qlearning",
)
print("qlearning ok")

#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################


agent = QLearningAgentEpsScheduling(
    learning_rate=0.5, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

assert np.mean(rewards[-100:]) > 0.0

# TODO: créer des vidéos de l'agent en action
frames = play_exp(env, agent)
save_video(
    frames=frames,
    video_folder="videos",
    fps=env.metadata["render_fps"],
    name_prefix="qlearning-eps-scheduling",
)
print("qlearning epsilon scheduling ok")


####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.5, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
for i in range(1000):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        print("mean reward", np.mean(rewards[-100:]))

frames = play_exp(env, agent)
save_video(
    frames=frames,
    video_folder="videos",
    fps=env.metadata["render_fps"],
    name_prefix="sarsa",
)
print("sarsa ok")
