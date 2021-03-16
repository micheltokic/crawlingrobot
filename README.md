# gym-crawlingrobot

Crawling Robot 2D Simulation with physics behavior.

# Used librarys with Python 3.7

- Pygame for GUI
- Pymunk for Physics


# Installation

git clone https://gitlab.lrz.de/arl-ws2021/gym-crawlingrobot.git

pip install -e gym-crawlingrobot

## Example for usage
``` Python
import gym
import gym_crawlingrobot

# crawlingrobot discrete environment for Reinforcement Learning algorithm e.g. Q-Learning
env = gym.make('crawlingrobot-discrete-v1', rotation_angles=5, goal_distance=2500, window_size=(1500, 800))
env.reset()
...

# crawlingrobot continuous environment for Reinforcement Learning algorithm e.g. PPO2
env = gym.make('crawlingrobot-continuous-v1', goal_distance=2500, window_size=(1500, 800), render_intermediate_steps=False)


```

For Quick Start you can find a Python-notebook under the folder `example`.
