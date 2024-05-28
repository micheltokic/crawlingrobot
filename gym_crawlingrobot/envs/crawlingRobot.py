import sys, random
import pygame
import pymunk
import pymunk.pygame_util
import math
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import os

random.seed(1)  # make the simulation the same each time, easier to debug

class CrawlingRobot:
    PHYSICS_TIME_STEP = 1 / 200  # physics simulation timestep, smaller is better, has to divide ACTION_TIME,
    ACTION_TIME = 1 / 5  # time an action takes in seconds
    TOTAL_ROTATION_ANGLE = 60   # full angle of possible rotation in deg
    rate = ((math.pi / 180) * 15) / ACTION_TIME
    mode = 0

    def __init__(self, goal_distance=1400, window_size=(1500, 800), render_intermediate_steps=False, plot_steps_per_episode=False):
        pygame.init()
        self.window_size = window_size
        self.goal = goal_distance
        self.level_size = ((abs(self.goal)+self.window_size[0])*2, 600)
        self.base_offset = self.level_size[0] / 2
        self.current_timesteps = 0
        self.current_steps = 0
        self.previous_rewards = collections.deque([0]*100, 100)
        self.episode_time_results = []
        self.episode_steps_results = []
        self.done = False
        self.render_intermediate_steps = render_intermediate_steps
        self.is_render_initialized = False
        self.pymunk_layer = None
        self.plot_steps_per_episode = plot_steps_per_episode

        self._init_physics()
        self._init_robot_body()

    def _init_render(self):
        self._init_episode_results_plot()
        self._init_rewards_plot()
        pygame.display.set_caption("Crawling Robot")
        self.clock = pygame.time.Clock()

        self.screen = pygame.display.set_mode(self.window_size)
        self.pymunk_layer = pygame.Surface(self.level_size)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.pymunk_layer)
        
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.flag = pygame.image.load(dir_path + '/res/flag.png')

        self.camera = [-self.base_offset, -100]
        self.update_camera()

    def _init_episode_results_plot(self):
        self.fig = plt.figure()
        self.fig.set_size_inches(self.window_size[0] / self.fig.dpi, 4)
        self.ax = self.fig.add_subplot(211)
        self.ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax.set_xlabel("episodes")
        if self.plot_steps_per_episode:
            self.ax.set_ylabel("steps")
        else:
            self.ax.set_ylabel("seconds")
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.graph = self.plot_episodes(self.ax, self.episode_time_results)
    
    def _init_rewards_plot(self):
        self.fig_rewards = plt.figure()
        self.fig_rewards.set_size_inches(self.window_size[0] / self.fig_rewards.dpi, 4) 
        self.ax_rewards = self.fig_rewards.add_subplot(211)
        self.ax_rewards.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax_rewards.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
        self.ax_rewards.set_xlabel("step")
        self.ax_rewards.set_ylabel("reward")
        self.canvas_rewards = agg.FigureCanvasAgg(self.fig_rewards)
        self.graph_rewards = self.plot_rewards()

    def _init_physics(self):
        self.space = pymunk.Space()
        self.space.gravity = (0.0, 9000.0)

        #ground
        ground = pymunk.Segment(self.space.static_body, (0, 400), (self.level_size[0], 400), 4)
        ground.friction = 10
        self.space.add(ground)

        # walls
        wall_left = pymunk.Segment(self.space.static_body, (5, 400), (5, 200), 5)
        wall_left.friction = 10
        self.space.add(wall_left)

        wall_right = pymunk.Segment(self.space.static_body, (self.level_size[0]-5, 400), (self.level_size[0]-5, 200), 5)
        wall_right.friction = 10
        self.space.add(wall_right)

    def _init_robot_body(self):
        # robot body
        body_size = (120, 60)
        self.robot_body = pymunk.Body(10, 1000)
        self.robot_body.position = (self.base_offset+190, 340)
        robot = pymunk.Poly.create_box(self.robot_body, body_size, 1)
        robot.friction = 1
        robot.mass = 8
        self.space.add(self.robot_body, robot)

        # wheel
        wheel_size = 12
        wheel_body = pymunk.Body(1, 10)
        wheel_body.position = (self.base_offset+150, 350)
        wheel = pymunk.Circle(wheel_body, wheel_size)
        wheel.friction = 2
        wheel.mass = 0.5
        self.space.add(wheel_body, wheel)

        # wheel robot bodyjoint
        wheel_joint = pymunk.PivotJoint(wheel_body, self.robot_body, (0,0), (-55, 25))
        self.space.add(wheel_joint)

        # upper arm
        self.ua_body = pymunk.Body(10, 1000)
        upper_arm_point1 = (self.base_offset+250, 310)
        upper_arm_point2 = (self.base_offset+310, 230)
        self.upper_arm = pymunk.Segment(self.ua_body, upper_arm_point1, upper_arm_point2, 8)
        self.upper_arm.mass = 1
        self.upper_arm.friction = 10
        self.space.add(self.ua_body, self.upper_arm)

        # lower arm
        la_body = pymunk.Body(10, 1000)
        lower_arm_point2 = (self.base_offset+370, 280)
        self.lower_arm = pymunk.Segment(la_body, upper_arm_point2, lower_arm_point2, 8)
        self.lower_arm.mass = 1
        self.lower_arm.friction = 10
        self.space.add(la_body, self.lower_arm)

        # joint robot + upper arm
        joint_robot = pymunk.PivotJoint(self.ua_body, self.robot_body, upper_arm_point1, (60, -30))
        self.space.add(joint_robot)

        # joint upper arm + lower arm
        joint = pymunk.PivotJoint(self.ua_body, la_body, upper_arm_point2, upper_arm_point2)
        self.space.add(joint)

        # motor robot
        self.motor_robot = pymunk.SimpleMotor(self.robot_body, la_body, 0)
        self.space.add(self.motor_robot)

        # motor arm
        self.motor_arm = pymunk.SimpleMotor(self.ua_body, la_body, 0)
        self.space.add(self.motor_arm)

        # filter (to avoid collision)
        shape_filter = pymunk.ShapeFilter(group=1)
        self.upper_arm.filter = shape_filter
        self.lower_arm.filter = shape_filter
        wheel.filter = shape_filter
        robot.filter = shape_filter

        self.last_position = self.robot_body.position.x

        # fix initial physic body clipping
        for i in range(100):
                        robot.space.step(0.01)

    def action(self, target_angles: [float, float]):
        rate = CrawlingRobot.rate
        self.current_steps += 1
        target_angles = np.clip(target_angles, 0, self.TOTAL_ROTATION_ANGLE)     # cap target angle to valid value between 0 and TOTAL_ROTATION_ANGLE

        angle_upper = self.get_arm_angle(self.upper_arm)
        angle_lower = self.get_arm_angle(self.lower_arm)
        target_angle_upper = target_angles[0]
        target_angle_lower = target_angles[1]

        self.motor_robot.rate = 0
        self.motor_arm.rate = 0
        upper_real_rate = 0
        lower_real_rate = 0

        if angle_upper < target_angle_upper:
            self.motor_robot.rate -= rate
            upper_real_rate = -rate
        elif angle_upper > target_angle_upper:
            self.motor_robot.rate += rate
            upper_real_rate = rate
   
        if angle_lower < target_angle_lower:
            self.motor_arm.rate -= rate
            self.motor_robot.rate -= rate
            lower_real_rate = -rate
        elif angle_lower > target_angle_lower:
            self.motor_arm.rate += rate
            self.motor_robot.rate += rate
            lower_real_rate = rate

        while True:
            angle_upper = self.get_arm_angle(self.upper_arm)
            angle_lower = self.get_arm_angle(self.lower_arm)

            if upper_real_rate != 0:
                if upper_real_rate < 0 and angle_upper > target_angle_upper:
                    self.motor_robot.rate += rate
                    upper_real_rate = 0
                elif upper_real_rate > 0 and angle_upper < target_angle_upper:
                    self.motor_robot.rate -= rate
                    upper_real_rate = 0
            if lower_real_rate != 0:
                if lower_real_rate < 0 and angle_lower > target_angle_lower:
                    self.motor_arm.rate += rate
                    self.motor_robot.rate += rate
                    lower_real_rate = 0
                elif lower_real_rate > 0 and angle_lower < target_angle_lower:
                    self.motor_arm.rate -= rate
                    self.motor_robot.rate -= rate
                    lower_real_rate = 0

            if self.motor_robot.rate == 0 and self.motor_arm.rate == 0:
                break 

            self.space.step(CrawlingRobot.PHYSICS_TIME_STEP)
            self.current_timesteps += 1

            if(self.render_intermediate_steps and self.is_render_initialized):
                self.render()

        if self.is_render_initialized:
            self.update_camera()

        if self.is_render_initialized and self.render_intermediate_steps:
            self.graph_rewards = self.plot_rewards()
        
        angle = self.to_deg(self.robot_body.angle)
        if angle < -90 or angle > 90:
            self.reset()


        if not self.done and self.check_if_past_goal():
            self.done = True
            self.episode_time_results.append(self.current_timesteps * self.PHYSICS_TIME_STEP)
            self.episode_steps_results.append(self.current_steps)
            self.current_timesteps = 0
            self.current_steps = 0
            if self.is_render_initialized:
                if self.plot_steps_per_episode:
                    episode_plot_data = self.episode_steps_results
                else:
                    episode_plot_data = self.episode_time_results
                self.graph = self.plot_episodes(self.ax, episode_plot_data)
                self.graph_rewards = self.plot_rewards()

    def check_if_past_goal(self):
        if self.goal > 0:
            return (self.robot_body.position.x-self.base_offset) >= self.goal
        else:
            return (self.robot_body.position.x-self.base_offset) <= self.goal

    def get_current_timestep(self):
        return np.sum(self.episode_time_results) + (self.current_timesteps * self.PHYSICS_TIME_STEP)

    def reset_current_timestep(self):
        self.episode_time_results = []
        self.current_timesteps = 0

    def get_observation(self):
        raise NotImplementedError('Not implemented in CrawlingRobot class! Use CrawlingRobotDiscrete or CrawlingRobotContinuous instead')

    def get_arm_angle(self, arm):
        ref_angle = self.robot_body.angle if arm == self.upper_arm else self.upper_arm.body.angle
        arm_angle = self.to_deg(arm.body.angle - ref_angle)
        if arm_angle > 0:
            arm_angle %= 360
        return arm_angle

    def to_deg(self, rad):
        return rad * (180 / math.pi)

    def get_distance(self):
        distance = self.robot_body.position.x - self.last_position
        self.last_position = self.robot_body.position.x
        
        # if goal is to the left, reward has to be negated
        if self.goal < 0:
            distance = -distance
        self.previous_rewards.append(distance)
        return distance

    def is_done(self):
        return self.done

    def update_camera(self):
        self.camera[0] = -(self.base_offset + np.floor((self.robot_body.position[0]-self.base_offset+100) / self.window_size[0]) * self.window_size[0])

    def reset(self):
        # reinitialize physics-objects to reset, lazy
        self.done = False
        self._init_physics()
        self._init_robot_body()

    def render(self):
        if not self.is_render_initialized:
            self.is_render_initialized = True
            self._init_render()

        self.screen.fill((255,255,255))

        font = pygame.font.SysFont('Verdana', int(self.window_size[0] / 90), 1)
        text1, text2 = "", ""
        if self.mode == 1:
            text1 = "Press 'Space' to control the simulation speed"
        elif self.mode == 2:
            text1 = "Use WASD or Arrow Keys to control the robot's arms"
            text2 = "Press 'R' to reset"
        elif self.mode == 3:
            text1 = "Press 'Space' to control the simulation speed"
            text2 = ""#"Press 'E' to change the learn rate"

        self.pymunk_layer.fill((255,255,255))
        self.space.debug_draw(self.draw_options)
        self.screen.blit(self.pymunk_layer, self.camera)

        self.screen.blit(self.flag, (self.base_offset + self.goal + self.camera[0], 180+self.camera[1]))
        control_text1 = font.render(text1, True, (0, 0, 0))
        self.screen.blit(control_text1, (10, 5))
        if self.mode > 1:
            control_text2 = font.render(text2, True, (0, 0, 0))
            self.screen.blit(control_text2, (10, 25))

        self.screen.blit(self.graph_rewards, (10, 410+self.camera[1]))
        self.screen.blit(self.graph, (10, 640+self.camera[1]))

        pygame.display.flip()
        #self.clock.tick(60

    def plot_episodes(self, ax, data):
        ax.plot(data, ls="-", marker="o", color='k')
        ax.set_xlim(left=0, right=max(10, len(data)-1))
        ax.set_ylim(bottom=0, top=max(data, default=1)*1.1)
        ax.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.2)
        #self.fig.tight_layout()
        self.canvas.draw()
        renderer = self.canvas.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas.get_width_height()

        return pygame.image.fromstring(raw_data, size, "RGB")
    
    def plot_rewards(self):
        colors = []
        for r in self.previous_rewards:
            c = "black"
            if r > 0:
                c = "green"
            elif r < 0:
                c = "red"
            colors.append(c)

        prev_rs = self.previous_rewards
        num_rs = len(prev_rs)

        plt.cla()
        self.ax_rewards.set_xlabel("step")
        self.ax_rewards.set_ylabel("reward")
        self.ax_rewards.scatter(list(range(-num_rs,0)), prev_rs, c=colors)
        self.ax_rewards.plot(list(range(-num_rs,0)), prev_rs, color="black")
        self.ax_rewards.set_xlim(left=-num_rs, right=0)
        self.ax_rewards.set_ylim(bottom=-120, top=120)
        self.ax_rewards.grid(visible=True, which='major', color='#666666', linestyle='-', alpha=0.2)
        #self.fig_rewards.tight_layout()
        self.canvas_rewards.draw()
        renderer = self.canvas_rewards.get_renderer()
        raw_data = renderer.tostring_rgb()
        size = self.canvas_rewards.get_width_height()

        return pygame.image.fromstring(raw_data, size, "RGB")

class CrawlingRobotContinuous(CrawlingRobot):
    def __init__(self, goal_distance=1400, window_size=(1500, 800), render_intermediate_steps=False, plot_steps_per_episode=False):
        window_size = (max(window_size[0], 1000), max(window_size[1], 800))
        super(CrawlingRobotContinuous, self).__init__(goal_distance, window_size, render_intermediate_steps, plot_steps_per_episode)

    def get_observation(self):
        ua_data = self.get_arm_angle(self.upper_arm)
        la_data = self.get_arm_angle(self.lower_arm)
        return [ua_data, la_data]

class CrawlingRobotDiscrete(CrawlingRobot):
    ROTATION_ANGLES = 4  # number of different rotation settings for each arm (influences Env + Q-Learning!)
    ROTATION_ANGLE_STEP = CrawlingRobot.TOTAL_ROTATION_ANGLE / ROTATION_ANGLES  # degrees of rotation per action (vorher 360/24=15)

    def __init__(self, rotation_angles=5, goal_distance=1400, window_size=(1500, 800), render_intermediate_steps=False, plot_steps_per_episode=False):
        window_size = (max(window_size[0], 1000), max(window_size[1], 800))
        super(CrawlingRobotDiscrete, self).__init__(goal_distance, window_size, render_intermediate_steps, plot_steps_per_episode)
        CrawlingRobotDiscrete.ROTATION_ANGLES = rotation_angles

    def action(self, action:int):
        rounded_upper = self.get_rounded_arm_angle(self.upper_arm)
        rounded_lower = self.get_rounded_arm_angle(self.lower_arm)

        target_angle_upper = rounded_upper
        target_angle_lower = rounded_lower

        # 0 up, 1 right, 2 down, 3 left
        if action == 0 and rounded_upper > 0:
            target_angle_upper = rounded_upper - self.ROTATION_ANGLE_STEP   
        elif action == 1 and rounded_lower > 0:
            target_angle_lower = rounded_lower - self.ROTATION_ANGLE_STEP 
        elif action == 2 and rounded_upper < CrawlingRobot.TOTAL_ROTATION_ANGLE:
            target_angle_upper = rounded_upper + self.ROTATION_ANGLE_STEP 
        elif action == 3 and rounded_lower < CrawlingRobot.TOTAL_ROTATION_ANGLE:
            target_angle_lower = rounded_lower + self.ROTATION_ANGLE_STEP 

        if target_angle_upper == rounded_upper and target_angle_lower == rounded_lower:
            # skip action, if already in correct discrete target state. maybe remove this, but results are worse
            return
        super().action([target_angle_upper, target_angle_lower])

    def get_observation(self):
        ua_data = max(min(round((self.get_arm_angle(self.upper_arm) / self.ROTATION_ANGLE_STEP)), self.ROTATION_ANGLES - 1), 0)
        la_data = max(min(round((self.get_arm_angle(self.lower_arm) / self.ROTATION_ANGLE_STEP)), self.ROTATION_ANGLES - 1), 0)
        return [ua_data, la_data]

    def get_rounded_arm_angle(self, arm):
        return round(self.get_arm_angle(arm) / self.ROTATION_ANGLE_STEP) * self.ROTATION_ANGLE_STEP

def run():
    robot = CrawlingRobotDiscrete(window_size=[1400, 800], goal_distance=1400, render_intermediate_steps=True)
    robot.mode = 2
    pygame.init()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    sys.exit(0)
                elif event.key == pygame.K_UP or event.key == pygame.K_w:
                    robot.action(0)
                elif event.key == pygame.K_RIGHT or event.key == pygame.K_d:
                    robot.action(1)
                elif event.key == pygame.K_DOWN or event.key == pygame.K_s:
                    robot.action(2)
                elif event.key == pygame.K_LEFT or event.key == pygame.K_a:
                    robot.action(3)
                elif event.key == pygame.K_r:
                    robot.reset()
                    robot.action(3)
                robot.get_distance()    # to update rewards plot

        robot.render()
        robot.clock.tick(60)

        if robot.robot_body.position.x-robot.base_offset >= robot.goal:
            robot.reset()