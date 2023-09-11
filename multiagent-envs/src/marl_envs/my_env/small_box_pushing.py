#!/usr/bin/python

import gym
import numpy as np
import IPython

from gym import spaces
from .box_pushing_core import Agent, Box 

DIRECTION = [(0,1), (1,0), (0,-1), (-1,0)]
ACTIONS = ["Move_Forward", "Turn_L", "Turn_R", "Stay"]

class SmallBoxPushing(gym.Env):
    
    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, grid_dim=(4,4), n_agent=2, terminate_step=100, random_init=False, terminal_reward_only=False, small_box_reward=10, *args, **kwargs):

        assert (n_agent <= 6 and n_agent <= grid_dim[0]), "Too many agents"
        self.n_agent = n_agent
        
        #"move forward, turn left, turn right, stay"
        self.action_space = [spaces.Discrete(4)] * self.n_agent
        self.observation_space = [spaces.MultiBinary(5)] * self.n_agent

        self.xlen, self.ylen = grid_dim

        self.random_init = random_init
        self.terminal_reward_only = terminal_reward_only
        self.small_box_reward = small_box_reward

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
        return [self.observation_space[0].n] * self.n_agent
    
    @property
    def n_action(self):
        return [a.n for a in self.action_spaces]
    
    def action_space_sample(self, i):
        return np.random.randint(self.action_spaces[i].n)
    
    @property
    def action_spaces(self):
        return self.action_space

    def createAgents(self):
        if self.random_init:
            init_ori = np.random.randint(4,size=2)
            init_xs = np.random.randint(self.xlen,size=2) + 0.5
            init_ys = np.random.randint(int(self.ylen/2-1),size=2) + 0.5
            A0 = Agent(0, init_xs[0], init_ys[0], init_ori[0], (self.xlen, self.ylen))
            A1 = Agent(1, init_xs[1], init_ys[1], init_ori[1], (self.xlen, self.ylen))
            A2 = Agent(1, init_xs[1], init_ys[1], init_ori[1], (self.xlen, self.ylen))
        else:
            if self.ylen >= 8.0:
                A0 = Agent(0, 1.5, 1.5, 1, (self.xlen, self.ylen))
                A1 = Agent(1, self.xlen-1.5, 1.5, 3, (self.xlen, self.ylen))
            elif self.ylen == 6.0:
                A0 = Agent(0, 0.5, 1.5, 1, (self.xlen, self.ylen))
                A1 = Agent(1, 5.5, 1.5, 3, (self.xlen, self.ylen))
            else:
                A0 = Agent(0, 0.5, 0.5, 1, (self.xlen, self.ylen))
                A1 = Agent(1, 3.5, 0.5, 3, (self.xlen, self.ylen))

        self.agents = [A0, A1]
        for i in range(self.n_agent-2):
            if self.ylen >= 6.0:
                if i % 2 == 0:
                    A = Agent(i+2, self.xlen/2.0-(i//2*1.0+0.5), 1.5, 1, (self.xlen, self.ylen))
                else:
                    A = Agent(i+2, self.xlen/2.0+(i//2*1.0+0.5), 1.5, 3, (self.xlen, self.ylen))
            else:
                if i % 2 == 0:
                    A = Agent(i+2, self.xlen/2.0-(i//2*1.0+0.5), 0.5, 1, (self.xlen, self.ylen))
                else:
                    A = Agent(i+2, self.xlen/2.0+(i//2*1.0+0.5), 0.5, 3, (self.xlen, self.ylen))
            self.agents.append(A)

    def createBoxes(self):
        if self.ylen >= 8.0:
            SB_0 = Box(0, 1.5, self.ylen/2+0.5, 1.0, 1.0) 
            SB_1 = Box(1, self.ylen-1.5, self.ylen/2+0.5, 1.0, 1.0) 
        elif self.ylen == 6.0:
            SB_0 = Box(0, 0.5, self.ylen/2+0.5, 1.0, 1.0) 
            SB_1 = Box(1, 5.5, self.ylen/2+0.5, 1.0, 1.0) 
        else:
            SB_0 = Box(0, 0.5, self.ylen/2+0.5, 1.0, 1.0) 
            SB_1 = Box(1, 3.5, self.ylen/2+0.5, 1.0, 1.0) 

        self.boxes = [SB_0, SB_1]
        for i in range(self.n_agent-2):
            if i % 2 == 0:
                SB = Box(1, self.xlen/2.0-(i//2*1.0+0.5), self.ylen/2+0.5, 1.0, 1.0) 
            else:
                SB = Box(1, self.xlen/2.0+(i//2*1.0+0.5), self.ylen/2+0.5, 1.0, 1.0) 
            self.boxes.append(SB)
    
    def reset(self, debug=False):
        self.createAgents()
        self.createBoxes()
        self.t = 0
        self.count_step = 0
        self.pushing_big_box = False

        if debug:
            self.render()

        return self._getobs()

    def step(self, actions, debug=False):

        if self.terminal_reward_only:
            rewards = 0.0
        else:
            rewards = 0.0
        terminate = 0

        cur_actions = actions
        cur_actions_done = [1,1]
        self.pushing_big_box = False

        self.count_step += 1

        for idx, agent in enumerate(self.agents):
            reward = agent.step(actions[idx], self.boxes)
            if not self.terminal_reward_only:
                rewards += reward

        # check whether any box is pushed to the goal area
        reward = 0.0
        small_box = 0.0

        for idx, box in enumerate(self.boxes):
            if box.ycoord == self.ylen - 0.5:
                terminate = 1
                reward = reward + self.small_box_reward
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
            for ag in self.agents:
                print("Agent_" + str(ag.idx) + " \t action \t\t{}".format(ACTIONS[ag.cur_action]))
                print(" ")

        observations = self._getobs(debug)

        return cur_actions, observations, [rewards]*self.n_agent, [bool(terminate)]*self.n_agent, cur_actions_done, {}

    def _getobs(self, debug=False):

        if debug:
            print("")
            print("Observations list:")

        observations = []
        for idx, agent in enumerate (self.agents):

            obs = np.zeros(self.observation_space[0].n)

            # assume empty front
            obs[2] = 1.0

            # observe small box
            for box in self.boxes:
                if box.xcoord == agent.xcoord + DIRECTION[agent.ori][0] and \
                        box.ycoord == agent.ycoord + DIRECTION[agent.ori][1]:
                        obs[0] = 1.0
                        obs[2] = 0.0
                        break
            
            # observe wall
            if agent.xcoord + DIRECTION[agent.ori][0] > self.xlen or \
                    agent.xcoord + DIRECTION[agent.ori][0] < 0.0 or \
                    agent.ycoord + DIRECTION[agent.ori][1] > self.ylen or \
                    agent.ycoord + DIRECTION[agent.ori][1] < 0.0:
                        obs[3] = 1.0
                        obs[2] = 0.0
            
            # observe agent
            for teamate_idx in range(self.n_agent):
                if teamate_idx != idx:
                    if (agent.xcoord + DIRECTION[agent.ori][0] == self.agents[teamate_idx].xcoord) and \
                            (agent.ycoord + DIRECTION[agent.ori][1] == self.agents[teamate_idx].ycoord):
                        obs[4] = 1.0
                        obs[2] = 0.0
                        break

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

    def get_state(self):
        positions = []
        for ag in self.agents:
            positions.append(ag.xcoord/self.xlen)
            positions.append(ag.ycoord/self.ylen)
            ori = np.eye(4)
            ori = ori[ag.ori]
            positions += ori.tolist()
        for bx in self.boxes:
            positions.append(bx.xcoord/self.xlen)
            positions.append(bx.ycoord/self.ylen)
        return np.array(positions)

    def get_env_info(self):
        return {'state_shape': len(self.get_state()),
                'obs_shape': self.obs_size[0],
                'n_actions': self.n_action[0],
                'n_agents': self.n_agent,
                'episode_limit': self.terminate_step}

    def get_avail_actions(self):
        return [[1]*n for n in self.n_action]

    def render(self, mode='human'):
        
        screen_width = 8*100
        screen_height = 8*100

        scale = 8 / self.ylen

        agent_size = 30.0
        agent_in_size = 25.0
        agent_clrs = [((0.15,0.65,0.15), (0.0,0.8,0.4)), ((0.15,0.15,0.65), (0.0, 0.4,0.8)), ((0.2,0.0,0.4), (0.5,0.0,1.0)), ((0.12,0.12,0.12), (0.5,0.5,0.5)), ((0,0.4,0.4), (0.0,1.0,1.0)), ((0.4,0.2,0.0), (1.0,0.5,0.0))]

        small_box_size = 85.0
        small_box_clrs = [(0.43,0.28,0.02), (0.67,0.43,0.02)]
        small_box_in_size = 75.0

        if self.viewer is None:
            from marl_envs.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #-------------------draw line-----------------
            for l in range(1, self.ylen):
                line = rendering.Line((0.0, l*100*scale), (screen_width, l*100*scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.ylen):
                line = rendering.Line((l*100*scale, 0.0), (l*100*scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (0.0, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((screen_width, 0.0), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            line = rendering.Line((0.0, screen_width), (screen_width, screen_width))
            line.linewidth.stroke = 60
            line.set_color(0.0, 0.0, 0.0)
            self.viewer.add_geom(line)

            #-------------------draw goal
            goal = rendering.FilledPolygon([(-(screen_width-8)/2.0,(-50+2)*scale), (-(screen_width-8)/2.0,(50-2)*scale), ((screen_width-8)/2.0,(50-2)*scale), ((screen_width-8)/2.0,-(50-2)*scale)])
            goal.set_color(1.0,1.0,0.0)
            goal_trans = rendering.Transform(translation=(screen_width/2.0,(self.ylen-0.5)*100*scale))
            goal.add_attr(goal_trans)
            self.viewer.add_geom(goal)

            #-------------------draw small box
            self.small_box_trans=[]
            for box in self.boxes:
                small_box = rendering.FilledPolygon([(-small_box_size/2.0*scale,-small_box_size/2.0*scale), (-small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,small_box_size/2.0*scale), (small_box_size/2.0*scale,-small_box_size/2.0*scale)])
                small_box.set_color(*small_box_clrs[0])
                self.small_box_trans.append(rendering.Transform(translation=(box.xcoord*100*scale, box.ycoord*100*scale)))
                small_box.add_attr(self.small_box_trans[-1])
                self.viewer.add_geom(small_box)

            self.small_box_in_trans=[]
            for box in self.boxes:
                small_box_in = rendering.FilledPolygon([(-small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale), (-small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,small_box_in_size/2.0*scale), (small_box_in_size/2.0*scale,-small_box_in_size/2.0*scale)])
                small_box_in.set_color(*small_box_clrs[1])
                self.small_box_in_trans.append(rendering.Transform(translation=(box.xcoord*100*scale, box.ycoord*100*scale)))
                small_box_in.add_attr(self.small_box_in_trans[-1])
                self.viewer.add_geom(small_box_in)
            
            #-------------------draw agent
            self.agent_trans = []
            for ag in self.agents:
                agent = rendering.make_circle(radius=agent_size*scale)
                agent.set_color(*agent_clrs[ag.idx][0])
                self.agent_trans.append(rendering.Transform(translation=(ag.xcoord*100*scale, ag.ycoord*100*scale)))
                agent.add_attr(self.agent_trans[-1])
                self.viewer.add_geom(agent)

            self.agent_in_trans = []
            for ag in self.agents:
                agent_in = rendering.make_circle(radius=agent_in_size*scale)
                agent_in.set_color(*agent_clrs[ag.idx][1])
                self.agent_in_trans.append(rendering.Transform(translation=(ag.xcoord*100*scale, ag.ycoord*100*scale)))
                agent_in.add_attr(self.agent_in_trans[-1])
                self.viewer.add_geom(agent_in)
            
            #-------------------draw agent sensor
            sensor_size = 20.0
            sensor_in_size = 14.0
            sensor_clrs = ((0.65,0.15,0.15), (1.0, 0.2,0.2))

            self.sensor_trans = []
            for idx in range(self.n_agent):
                sensor = rendering.FilledPolygon([(-sensor_size/2.0*scale,-sensor_size/2.0*scale), (-sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,sensor_size/2.0*scale), (sensor_size/2.0*scale,-sensor_size/2.0*scale)])
                sensor.set_color(*sensor_clrs[0])
                self.sensor_trans.append(rendering.Transform(translation=(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                                                       self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)))
                sensor.add_attr(self.sensor_trans[-1])
                self.viewer.add_geom(sensor)

            self.sensor_in_trans = []
            for idx in range(self.n_agent):
                sensor_in = rendering.FilledPolygon([(-sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale), (-sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,sensor_in_size/2.0*scale), (sensor_in_size/2.0*scale,-sensor_in_size/2.0*scale)])
                sensor_in.set_color(*sensor_clrs[1])
                self.sensor_in_trans.append(rendering.Transform(translation=(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                                                            self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)))
                sensor_in.add_attr(self.sensor_in_trans[-1])
                self.viewer.add_geom(sensor_in)
            
        for idx, trans in enumerate(self.small_box_trans):
            trans.set_translation(self.boxes[idx].xcoord*100*scale, self.boxes[idx].ycoord*100*scale)
        for idx, trans in enumerate(self.small_box_in_trans):
            trans.set_translation(self.boxes[idx].xcoord*100*scale, self.boxes[idx].ycoord*100*scale)
        
        for idx, trans in enumerate(self.agent_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale, self.agents[idx].ycoord*100*scale)
        for idx, trans in enumerate(self.agent_in_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale, self.agents[idx].ycoord*100*scale)

        for idx, trans in enumerate(self.sensor_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)
            trans.set_rotation(0.0)
        for idx, trans in enumerate(self.sensor_in_trans):
            trans.set_translation(self.agents[idx].xcoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][0]*scale, 
                                self.agents[idx].ycoord*100*scale+(agent_size)*DIRECTION[self.agents[idx].ori][1]*scale)
            trans.set_rotation(0.0)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
