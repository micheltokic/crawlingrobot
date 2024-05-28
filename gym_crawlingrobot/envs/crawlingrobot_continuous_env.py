import gymnasium as gym
from gymnasium import spaces
import numpy as np
from gym_crawlingrobot.envs.crawlingRobot import CrawlingRobotContinuous
class CrawlingRobotContinuousEnv(gym.Env):
    """Custom Environment that follows gym interface"""
    metadata = {'render.modes': ['human']}

    def __init__(self, goal_distance=1300, window_size=(1400, 650), render_intermediate_steps=False, plot_steps_per_episode=True):
        super(CrawlingRobotContinuousEnv, self).__init__()
        self.robot = CrawlingRobotContinuous(goal_distance, window_size, render_intermediate_steps, plot_steps_per_episode)
        self.robot.mode = 1

        # observation space contains all of the env`s data to be observed by the agent
        self.action_space = spaces.Box(low=np.array([-1, -1]), high=np.array([1, 1]), dtype=np.float32)
        self.observation_space = spaces.Box(np.array([-60.0, -60.0]), np.array([60.0, 60.0]), dtype=np.float32)

    def step(self, action):
        assert self.action_space.contains(action)
        self.robot.action(action*60)
        obs = self.robot.get_observation()
        reward = self.robot.get_distance()
        done = self.robot.is_done()
        #print (obs)
        return np.array(obs), reward, done, done, {}

    def reset(self, seed=None, options={}):
        # Reset the state of the environment to an initial state
        self.robot.reset()
        #print (self.robot.get_observation())
        return np.array(self.robot.get_observation()), {}

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        self.robot.render()