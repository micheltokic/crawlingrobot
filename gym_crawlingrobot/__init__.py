from gym.envs.registration import register

register(id='crawlingrobot-discrete-v1', entry_point='gym_crawlingrobot.envs:CrawlingRobotDiscreteEnv')
register(id='crawlingrobot-continuous-v1', entry_point='gym_crawlingrobot.envs:CrawlingRobotContinuousEnv')