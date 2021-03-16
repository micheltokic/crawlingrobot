import gym
from gym import spaces
import numpy as np
from gym_crawlingrobot.envs.crawlingRobot import CrawlingRobotDiscrete

class CrawlingRobotDiscreteEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, rotation_angles=5, goal_distance=1300, window_size=(1400, 650), render_intermediate_steps=False, plot_steps_per_episode=True):
        super(CrawlingRobotDiscreteEnv, self).__init__()
        self.robot = CrawlingRobotDiscrete(rotation_angles, goal_distance, window_size, render_intermediate_steps, plot_steps_per_episode)
        self.robot.mode = 3

        # observation space contains all of the env`s data to be observed by the agent
        self.action_space = spaces.Discrete(4)
        size = self.robot.ROTATION_ANGLES
        low = np.array([0, 0])
        high = np.array([size-1, size-1])
        self.observation_space = spaces.Box(low, high, dtype=np.int64)
    
    def step(self, action):
        assert self.action_space.contains(action)
        self.robot.action(action)
        obs = self.robot.get_observation()
        reward = self.robot.get_distance()
        done = self.robot.is_done()
        return obs, reward, done, {}
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.robot.reset()
        return self.robot.get_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.robot.render()