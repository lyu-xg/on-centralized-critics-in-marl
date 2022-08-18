#!/usr/bin/python

import gym
import numpy as np
import IPython

from gym import spaces
from .box_pushing_core import Agent, Box

DIRECTION = [(0, 1), (1, 0), (0, -1), (-1, 0)]
ACTIONS = ["Move_Forward", "Turn_L", "Turn_R", "Stay"]


class BoxPushing(gym.Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 50}

    def __init__(
        self, grid_dim=(4, 4), terminate_step=300, random_init=False, *args, **kwargs
    ):

        self.n_agent = 2

        # "move forward, turn left, turn right, stay"
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.MultiBinary(5)
        # self.state_space = 

        self.xlen, self.ylen = grid_dim

        self.random_init = random_init

        self.createAgents()
        self.createBoxes()

        self.terminate_step = terminate_step
        self.pushing_big_box = False

        self.viewer = None

        self.single_small_box = 0.0
        self.both_small_box = 0.0
        self.big_box = 0.0

    @property
    def obs_size(self):
        return [self.observation_space.n] * self.n_agent

    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]

    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)

    @property
    def action_spaces(self):
        return [self.action_space] * 2

    def get_s1_index(self):
        state_list = self._getobs()
        obs1 = state_list[0].reshape((1, 5))
        obs2 = state_list[1].reshape((1, 5))
        return (
            obs1[0, 0] * 16
            + obs1[0, 1] * 8
            + obs1[0, 2] * 4
            + obs1[0, 3] * 2
            + obs1[0, 4]
        )

    def get_s2_index(self):
        state_list = self._getobs()
        obs1 = state_list[0].reshape((1, 5))
        obs2 = state_list[1].reshape((1, 5))
        return (
            obs2[0, 0] * 16
            + obs2[0, 1] * 8
            + obs2[0, 2] * 4
            + obs2[0, 3] * 2
            + obs2[0, 4]
        )

    def createAgents(self):
        if self.random_init:
            init_ori = np.random.randint(4, size=2)
            init_xs = np.random.randint(8, size=2) + 0.5
            init_ys = np.random.randint(3, size=2) + 0.5
            A0 = Agent(0, init_xs[0], init_ys[0], init_ori[0])
            A1 = Agent(1, init_xs[1], init_ys[1], init_ori[1])
        else:
            if self.ylen >= 8.0:
                A0 = Agent(0, 1.5, 1.5, 1, (self.xlen, self.ylen))
                A1 = Agent(1, self.ylen - 1.5, 1.5, 3, (self.xlen, self.ylen))

                # A0 = Agent(0, 1.5, 1.5, 1, (self.xlen, self.ylen))
                # A1 = Agent(1, 6.5, 1.5, 3, (self.xlen, self.ylen))
            elif self.ylen == 6.0:
                A0 = Agent(0, 0.5, 1.5, 1, (self.xlen, self.ylen))
                A1 = Agent(1, 5.5, 1.5, 3, (self.xlen, self.ylen))
            else:
                A0 = Agent(0, 0.5, 0.5, 1, (self.xlen, self.ylen))
                A1 = Agent(1, 3.5, 0.5, 3, (self.xlen, self.ylen))

        self.agents = [A0, A1]

    def createBoxes(self):
        if self.ylen >= 8.0:
            SB_0 = Box(0, 1.5, (self.ylen / 2 + 0.5), 1.0, 1.0)
            SB_1 = Box(1, self.ylen - 1.5, (self.ylen / 2 + 0.5), 1.0, 1.0)
            BB_2 = Box(2, self.ylen / 2.0, (self.ylen / 2 + 0.5), 1.0, 2.0)

            # SB_0 = Box(0, 1.5, self.ylen/2+0.5, 1.0, 1.0)
            # SB_1 = Box(1, 6.5, self.ylen/2+0.5, 1.0, 1.0)
            # BB_2 = Box(2, 4.0, self.ylen/2+0.5, 1.0, 2.0)
        elif self.ylen == 6.0:
            SB_0 = Box(0, 0.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            SB_1 = Box(1, 5.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            BB_2 = Box(2, 3.0, self.ylen / 2 + 0.5, 1.0, 2.0)
        else:
            SB_0 = Box(0, 0.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            SB_1 = Box(1, 3.5, self.ylen / 2 + 0.5, 1.0, 1.0)
            BB_2 = Box(2, 2.0, self.ylen / 2 + 0.5, 1.0, 2.0)

        self.boxes = [SB_0, SB_1, BB_2]

    def get_observation(self):
        # state_list = self._getobs()
        # obs1 = state_list[0].reshape((1, 5))
        # obs2 = state_list[1].reshape((1, 5))
        # state = np.hstack((obs1, obs2))
        # return state
        s1, s2 = self._getobs()
        return [s1.reshape(-1), s2.reshape(-1)]

    def reset(self, debug=False):
        self.createAgents()
        self.createBoxes()
        self.t = 0
        self.count_step = 0
        self.pushing_big_box = False

        if debug:
            self.render()

        return self.get_observation()

    def step(self, actions, debug=False):

        rewards = -0.1
        terminate = 0

        cur_actions = actions
        cur_actions_done = [1, 1]
        self.pushing_big_box = False

        self.count_step += 1

        if (
            (actions[0] == 0)
            and (actions[1] == 0)
            and self.agents[0].ori == 0
            and self.agents[1].ori == 0
            and (
                (
                    self.agents[0].xcoord == self.boxes[2].xcoord - 0.5
                    and self.agents[1].xcoord == self.boxes[2].xcoord + 0.5
                    and self.agents[0].ycoord == self.boxes[2].ycoord - 1.0
                    and self.agents[1].ycoord == self.boxes[2].ycoord - 1.0
                )
                or (
                    self.agents[1].xcoord == self.boxes[2].xcoord - 0.5
                    and self.agents[0].xcoord == self.boxes[2].xcoord + 0.5
                    and self.agents[1].ycoord == self.boxes[2].ycoord - 1.0
                    and self.agents[0].ycoord == self.boxes[2].ycoord - 1.0
                )
            )
        ):
            self.pushing_big_box = True

        if not self.pushing_big_box:
            for idx, agent in enumerate(self.agents):
                reward = agent.step(actions[idx], self.boxes)
                rewards += reward
        else:
            for agent in self.agents:
                agent.cur_action = 0
                agent.ycoord += 1.0
            self.boxes[2].ycoord += 1.0

        reward = 0.0
        small_box = 0.0

        for idx, box in enumerate(self.boxes):
            if box.ycoord == self.ylen - 0.5:
                terminate = 1
                reward = reward + 10 if idx < 2 else reward + 100
                if idx == 2:
                    self.big_box += 1.0
                else:
                    small_box += 1.0

        if small_box == 1.0:
            self.single_small_box += 1.0
        elif small_box == 2.0:
            self.both_small_box += 1.0

        rewards += reward

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            print("Agent_0 \t action \t\t{}".format(ACTIONS[self.agents[0].cur_action]))
            print(" ")
            print("Agent_1 \t action \t\t{}".format(ACTIONS[self.agents[1].cur_action]))

        # observations = self._getobs(debug)

        return (
            self.get_observation(),
            rewards,
            bool(terminate) or self.count_step == self.terminate_step,
        )

    def _getobs(self, debug=False):

        if self.t == 0:
            obs = np.zeros(self.observation_space.n)
            obs[2] = 1.0
            self.t = 1
            observations = [obs, obs]
            self.old_observations = observations

            return observations

        if debug:
            print("")
            print("Observations list:")

        observations = []
        for idx, agent in enumerate(self.agents):

            obs = np.zeros(self.observation_space.n)

            # assume empty front
            obs[2] = 1.0

            # observe small box
            for box in self.boxes[0:2]:
                if (
                    box.xcoord == agent.xcoord + DIRECTION[agent.ori][0]
                    and box.ycoord == agent.ycoord + DIRECTION[agent.ori][1]
                ):
                    obs[0] = 1.0
                    obs[2] = 0.0
            # observe large box
            if (
                self.boxes[2].xcoord + 0.5 == agent.xcoord + DIRECTION[agent.ori][0]
                or self.boxes[2].xcoord - 0.5 == agent.xcoord + DIRECTION[agent.ori][0]
            ) and self.boxes[2].ycoord == agent.ycoord + DIRECTION[agent.ori][1]:
                obs[1] = 1.0
                obs[2] = 0.0

            # observe wall
            if (
                agent.xcoord + DIRECTION[agent.ori][0] > self.xlen
                or agent.xcoord + DIRECTION[agent.ori][0] < 0.0
                or agent.ycoord + DIRECTION[agent.ori][1] > self.ylen
                or agent.ycoord + DIRECTION[agent.ori][1] < 0.0
            ):
                obs[3] = 1.0
                obs[2] = 0.0

            # observe agent
            if idx == 0:
                teamate_idx = 1
            else:
                teamate_idx = 0
            if (
                agent.xcoord + DIRECTION[agent.ori][0]
                == self.agents[teamate_idx].xcoord
            ) and (
                agent.ycoord + DIRECTION[agent.ori][1]
                == self.agents[teamate_idx].ycoord
            ):
                obs[4] = 1.0
                obs[2] = 0.0

            if debug:
                print("Agent_" + str(idx) + " \t small_box  \t\t{}".format(obs[0]))
                print("          " + " \t large_box \t\t{}".format(obs[1]))
                print("          " + " \t empty \t\t\t{}".format(obs[2]))
                print("          " + " \t wall \t\t\t{}".format(obs[3]))
                print("          " + " \t teammate \t\t{}".format(obs[4]))
                print("")

            observations.append(obs)

        self.old_observations = observations

        return observations

    def render(self, mode="human"):

        screen_width = self.xlen * 100
        screen_height = self.ylen * 100

        if self.viewer is None:
            import rendering

            self.viewer = rendering.Viewer(screen_width, screen_height)

            # -------------------draw goal
            goal = rendering.FilledPolygon(
                [(-400, -50), (-400, 50), (400, 50), (400, -50)]
            )
            goal.set_color(1.0, 1.0, 0.0)
            goal_trans = rendering.Transform(translation=(400, (self.ylen - 0.5) * 100))
            goal.add_attr(goal_trans)
            self.viewer.add_geom(goal)

            # -------------------draw box
            small_box_0 = rendering.FilledPolygon(
                [(-50, -50), (-50, 50), (50, 50), (50, -50)]
            )
            small_box_0.set_color(0.42, 0.4, 0.4)
            self.small_box_0_trans = rendering.Transform(
                translation=(self.boxes[0].xcoord * 100, self.boxes[0].ycoord * 100)
            )
            small_box_0.add_attr(self.small_box_0_trans)
            self.viewer.add_geom(small_box_0)

            small_box_1 = rendering.FilledPolygon(
                [(-50, -50), (-50, 50), (50, 50), (50, -50)]
            )
            small_box_1.set_color(0.42, 0.4, 0.4)
            self.small_box_1_trans = rendering.Transform(
                translation=(self.boxes[1].xcoord * 100, self.boxes[1].ycoord * 100)
            )
            small_box_1.add_attr(self.small_box_1_trans)
            self.viewer.add_geom(small_box_1)

            big_box_2 = rendering.FilledPolygon(
                [(-100, -50), (-100, 50), (100, 50), (100, -50)]
            )
            big_box_2.set_color(0.0, 0.0, 0.0)
            self.big_box_2_trans = rendering.Transform(
                translation=(self.boxes[2].xcoord * 100, self.boxes[2].ycoord * 100)
            )
            big_box_2.add_attr(self.big_box_2_trans)
            self.viewer.add_geom(big_box_2)

            # -------------------draw agent
            agent_0 = rendering.make_circle(radius=25.0)
            agent_0.set_color(0.0, 153.0 / 255.0, 0.0)
            self.agent_0_trans = rendering.Transform(
                translation=(self.agents[0].xcoord * 100, self.agents[0].ycoord * 100)
            )
            agent_0.add_attr(self.agent_0_trans)
            self.viewer.add_geom(agent_0)

            agent_1 = rendering.make_circle(radius=25.0)
            agent_1.set_color(0.0, 0.0, 153.0 / 255.0)
            self.agent_1_trans = rendering.Transform(
                translation=(self.agents[1].xcoord * 100, self.agents[1].ycoord * 100)
            )
            agent_1.add_attr(self.agent_1_trans)
            self.viewer.add_geom(agent_1)

            # -------------------draw agent sensor
            sensor_0 = rendering.FilledPolygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)]
            )
            sensor_0.set_color(1.0, 0.0, 0.0)
            self.sensor_0_trans = rendering.Transform(
                translation=(
                    self.agents[0].xcoord * 100 + 25 * DIRECTION[self.agents[0].ori][0],
                    self.agents[0].ycoord * 100 + 25 * DIRECTION[self.agents[0].ori][1],
                )
            )
            sensor_0.add_attr(self.sensor_0_trans)
            self.viewer.add_geom(sensor_0)

            sensor_1 = rendering.FilledPolygon(
                [(-10, -10), (-10, 10), (10, 10), (10, -10)]
            )
            sensor_1.set_color(1.0, 0.0, 0.0)
            self.sensor_1_trans = rendering.Transform(
                translation=(
                    self.agents[1].xcoord * 100 + 25 * DIRECTION[self.agents[1].ori][0],
                    self.agents[1].ycoord * 100 + 25 * DIRECTION[self.agents[1].ori][1],
                )
            )
            sensor_1.add_attr(self.sensor_1_trans)
            self.viewer.add_geom(sensor_1)

        self.small_box_0_trans.set_translation(
            self.boxes[0].xcoord * 100, self.boxes[0].ycoord * 100
        )
        self.small_box_1_trans.set_translation(
            self.boxes[1].xcoord * 100, self.boxes[1].ycoord * 100
        )
        self.big_box_2_trans.set_translation(
            self.boxes[2].xcoord * 100, self.boxes[2].ycoord * 100
        )

        self.agent_0_trans.set_translation(
            self.agents[0].xcoord * 100, self.agents[0].ycoord * 100
        )
        self.agent_1_trans.set_translation(
            self.agents[1].xcoord * 100, self.agents[1].ycoord * 100
        )

        # if self.agents[0].cur_action_done or self.agents[0].cur_action.idx <= 4:
        self.sensor_0_trans.set_translation(
            self.agents[0].xcoord * 100 + 25 * DIRECTION[self.agents[0].ori][0],
            self.agents[0].ycoord * 100 + 25 * DIRECTION[self.agents[0].ori][1],
        )
        self.sensor_0_trans.set_rotation(0.0)
        # else:
        #    x = self.agents[0].xcoord*100 + 25*self.agents[0].direct[0]
        #    y = self.agents[0].ycoord*100 + 25*self.agents[0].direct[1]
        #    angle = np.arccos(np.dot(self.agents[0].direct,np.array([1.0,0.0])))
        #    angle = angle * -1.0 if self.agents[0].direct[1] < 0.0 else angle
        #    self.sensor_0_trans.set_translation(x, y)
        #    self.sensor_0_trans.set_rotation(angle)

        # if self.agents[1].cur_action_done or self.agents[1].cur_action.idx <= 4:
        self.sensor_1_trans.set_translation(
            self.agents[1].xcoord * 100 + 25 * DIRECTION[self.agents[1].ori][0],
            self.agents[1].ycoord * 100 + 25 * DIRECTION[self.agents[1].ori][1],
        )
        self.sensor_1_trans.set_rotation(0.0)
        # else:
        #    x = self.agents[1].xcoord*100 + 25*self.agents[1].direct[0]
        #    y = self.agents[1].ycoord*100 + 25*self.agents[1].direct[1]
        #    angle = np.arccos(np.dot(self.agents[1].direct,np.array([1.0,0.0])))
        #    angle = angle * -1.0 if self.agents[1].direct[1] < 0.0 else angle
        #    self.sensor_1_trans.set_translation(x, y)
        #    self.sensor_1_trans.set_rotation(angle)

        return self.viewer.render(return_rgb_array=mode == "rgb_array")
