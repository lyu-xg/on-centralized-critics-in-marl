import numpy as np
import IPython
import gym
import random

from numpy.random import randint

NORTH = np.array([0, 1])
SOUTH = np.array([0, -1])
WEST = np.array([-1, 0])
EAST = np.array([1, 0])
STAY = np.array([0, 0])

TRANSLATION_TABLE = [
    # [left, intended_direction, right]
    [WEST,  NORTH, EAST],
    [EAST,  SOUTH, WEST],
    [SOUTH, WEST,  NORTH],
    [NORTH, EAST,  SOUTH],
    [STAY,  STAY,  STAY]
]

DIRECTION = np.array([[0.0, 1.0],
                      [0.0, -1.0],
                      [-1.0, 0.0],
                      [1.0, 0.0],
                      [0.0, 0.0]])

ACTIONS = ["NORTH", "SOUTH", "WEST", "EAST", "STAY"]

class CaptureTarget(gym.Env):

    """target random move around"""

    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second' : 50
            }

    def __init__(self, n_target=1, n_agent=2, grid_dim=(4,4), terminate_step=60,
                 intermediate_r=False, target_flick_prob=0.3, obs_one_hot=False,
                 tgt_avoid_agent=False, tgt_random_move=False, tgt_trans_noise=0.0, agent_trans_noise=0.1):

        # env generic settings
        self.n_target = n_target 
        self.n_agent = n_agent
        self.multi_task = False
        self.intermediate_reward = intermediate_r
        self.terminate_step=terminate_step
        
        # dimensions 
        self.x_len, self.y_len = grid_dim
        self.x_mean, self.y_mean = np.mean(np.arange(self.x_len)), np.mean(np.arange(self.y_len))

        # probabilities
        self.target_flick_prob = target_flick_prob
        self.tgt_avoid_agent=tgt_avoid_agent
        self.tgt_random_move = tgt_random_move
        self.tgt_trans_noise = tgt_trans_noise
        self.agent_trans_noise = agent_trans_noise

        self.n_action = [len(ACTIONS)] * self.n_agent
        self.action_space = gym.spaces.Discrete(len(ACTIONS))

        if obs_one_hot:
            self.obs_size = [self.x_len*self.y_len] * 2  # agent position and target position
            assert False, print("obs one hot not working")
        else:
            self.obs_size = [len(grid_dim) * (1 + self.n_target)] * self.n_agent # agent position and target position

        self.observation_space = gym.spaces.Box(low=-255, high=255, shape=(self.obs_size[0],), dtype=np.float32)
        
        assert  (self.n_agent / self.n_target == 2 or self.n_target==1), print("number of agent and number of target are not matching")

        self.obs_one_hot = obs_one_hot
        self.viewer = None

        self.reset()

    def action_space_sample(self,idx):
        return np.random.randint(self.n_action[idx])

    def action_space_batch_sample(self):
        return np.random.randint(self.n_action[idx], size=self.n_agent)

    def reset(self, debug=False):
        self.step_n = 0
        self.visited = np.zeros(self.n_target)

        # "game state" is really just the positions of all players and targets
        self.target_positions = np.stack(self.rand_target_position(self.n_target))
        self.agent_positions  = np.stack([self.rand_position() for _ in range(self.n_agent)])
        assert self.target_positions.shape == (self.n_target, 2) 

        obs = self.get_obs(debug)
        
        if debug:
            self.render()

        return obs

    def step(self, actions, debug=False):
        self.step_n += 1
        assert len(actions) == self.n_agent

        self.agent_positions = self.move(self.agent_positions, actions, noise=self.agent_trans_noise)

        if self.tgt_random_move:
            target_directions = np.random.randint(len(TRANSLATION_TABLE), size=self.n_target)
        elif self.tgt_avoid_agent:
            target_directions = self.get_tgt_moves()
        else:
            target_directions = [3] * self.n_target

        self.target_positions = self.move(self.target_positions, target_directions, noise=self.tgt_trans_noise)

        won = self.target_captured()

        r = float(won)

        if debug:
            self.render()
            print(" ")
            print("Actions list:")
            print("Target  \t action \t\t{}".format(ACTIONS[target_directions[0]]))
            print(" ")
            print("Agent_0 \t action \t\t{}".format(ACTIONS[actions[0]]))
            print(" ")
            print("Agent_1 \t action \t\t{}".format(ACTIONS[actions[1]]))
        
        terminate = bool(won) or self.step_n > self.terminate_step

        return actions, self.get_obs(debug), [r]*self.n_agent, [terminate]*self.n_agent, [1]*self.n_agent, {}
    
    def get_obs(self, debug=False):
        if self.obs_one_hot:
            agt_pos_obs = self.one_hot_positions(self.agent_positions)
            tgt_pos_obs = self.one_hot_positions(self.target_positions)
        else:
            agt_pos_obs = self.normalize_positions(self.agent_positions)
            tgt_pos_obs = self.normalize_positions(self.target_positions)
            
            tgt_pos_obs = tgt_pos_obs.reshape(1,-1)
            tgt_pos_obs = np.tile(tgt_pos_obs, (self.n_agent, 1))

        tgt_pos_obs = self.flick(tgt_pos_obs, prob=self.target_flick_prob)
        
        if debug:
            print("")
            print("Observations list:")
            for i in range(self.n_agent):
                print("Agent_" + str(i) + " \t self_loc  \t\t{}".format(self.agent_positions[i]))
                print("          " + " \t tgt_loc  \t\t{}".format(self.target_positions[0] if not all(tgt_pos_obs[i]==-1.0) else np.array([-1, -1])))
                print("")

        return [obs for obs in np.concatenate([agt_pos_obs, tgt_pos_obs], axis=1)]

    def get_state(self):
        agt_pos = self.normalize_positions(self.agent_positions)
        tgt_pos = self.normalize_positions(self.target_positions)
        return np.vstack([agt_pos, tgt_pos]).reshape(-1)

    def get_env_info(self):
        return {'state_shape': len(self.get_state()),
                'obs_shape': self.obs_size[0],
                'n_actions': self.n_action[0],
                'n_agents': self.n_agent,
                'episode_limit': self.terminate_step}

    def get_avail_actions(self):
        return [[1]*n for n in self.n_action]

    #################################################################################################
    # Helper functions
    
    def get_tgt_moves(self, single=True):
        assert self.target_positions.shape[0] == 1
        moves = self.wrap_positions(DIRECTION + self.target_positions)
        if single:
            cl_agt_idx = np.linalg.norm(self.agent_positions-self.target_positions, axis=1).argmin()
            h = np.linalg.norm(self.agent_positions[cl_agt_idx]-moves, axis=1)
        else:
            h_0 = np.linalg.norm(self.agent_positions[0]-moves, axis=1)
            h_1 = np.linalg.norm(self.agent_positions[1]-moves, axis=1)
            h = h_0 + h_1
        return np.random.choice(np.where(h == h.max())[0], size=1)
    
    def move(self, positions, directions, noise=0):
        translations = np.stack([self.translation(d, noise=noise) for d in directions])
        positions += translations
        return self.wrap_positions(positions)
    
    def target_captured(self):
        if self.n_target == 1:
            return all(np.array_equal(a, t) for a, t in zip(self.agent_positions, self.respective_target_positions(0)))
        else:
            return all(sum(np.array_equal(a, t) for a, t in zip(self.agent_positions, self.respective_target_positions(i))) == 2 for i in range(self.n_target))

    def respective_target_positions(self, idx):
        if self.multi_task:
            return self.target_positions
        else:
            return (self.target_positions[idx] for _ in range(self.n_agent))
    
    def rand_target_position(self, num):
        return np.array([random.sample(range(self.x_len), num), random.sample(range(self.y_len), num)]).T

    def rand_position(self):
        return np.array([randint(self.x_len), randint(self.y_len)])
    
    @staticmethod
    def translation(direction, noise=0.1):
        return TRANSLATION_TABLE[direction][np.random.choice(3, p=[noise/2, 1-noise, noise/2])]

    def flick(self, N, prob=0.3):
        mask = np.random.random(N.shape[0]*self.n_target).reshape(N.shape[0], -1) < prob
        N_new = N.reshape(self.n_agent, self.n_target, -1)
        N_new[mask] = np.array([[-1,-1]])
        return N_new.reshape(self.n_agent, -1)
    
    def normalize_positions(self, positions):
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([
                X / self.x_mean,
                Y / self.y_mean], axis=1)

    def one_hot_positions(self, positions):
        one_hot_vector = np.zeros((self.n_agent, self.x_len*self.y_len))
        index = positions[:,1] * self.y_len + positions[:,0]
        one_hot_vector[np.arange(self.n_agent), index] = 1
        return one_hot_vector
    
    def wrap_positions(self, positions):
        # fix translations which moved positions out of bound.
        X, Y = np.split(positions, 2, axis=1)
        return np.concatenate([X % self.x_len, Y % self.y_len], axis=1)

    def render(self, mode='human'):

        screen_width = 8 * 100
        screen_height = 8 * 100

        scale = 8 / self.y_len

        agent_size = 40.0
        agent_in_size = 35.0
        agent_clrs = [((0.15,0.15,0.65), (0.0, 0.4,0.8)),((0.15,0.65,0.15), (0.0,0.8,0.4)), ((0.65,0.15,0.15), (1.0, 0.5,0.5)), ((0.8, 0.8, 0.0), (1.0, 1.0,0.4)), ((0.8, 0.0, 0.8), (1.0, 0.4, 1.0)), ((0.0, 0.8, 0.8), (0.4, 1.0,1.0))]

        target_l = 80.0
        target_w = 26.0
        target_in_l = 70.0
        target_in_w = 16.0
        target_clrs = [((0.65,0.15,0.15), (1.0, 0.5,0.5)), ((0.8,0.4,0.0), (1.0, 0.70,0.37)),((0.37,0.37,0.37), (0.75,0.75,0.75))]

        if self.viewer is None:
            from marl_envs.my_env import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            #-------------------draw agents
            self.render_agents = []
            #agent_clrs = [(0.0,153.0/255.0,0.0), (0.0,0.0,153.0/255.0)]
            for i in range(self.n_agent):
                agent = rendering.make_circle(radius=agent_size*scale)
                agent.set_color(*agent_clrs[i][0])
                agent_trans = rendering.Transform(translation=((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale))
                agent.add_attr(agent_trans)
                self.render_agents.append(agent_trans)
                self.viewer.add_geom(agent)

            #-------------------draw agents contours
            for i in range(self.n_agent):
                agent = rendering.make_circle(radius=agent_in_size*scale)
                agent.set_color(*agent_clrs[i][1])
                agent_trans = rendering.Transform(translation=((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale))
                agent.add_attr(agent_trans)
                self.render_agents.append(agent_trans)
                self.viewer.add_geom(agent)

            self.render_target = []
            #-------------------draw target
            for t in range(self.n_target):
                tgt_l = rendering.FilledPolygon([(-target_w/2.0*scale,-target_l/2.0*scale), (-target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,-target_l/2.0*scale)])
                tgt_l.set_color(*target_clrs[t][0])
                tgt_l_trans = rendering.Transform(translation=tuple((self.target_positions[t]+0.5)*100*scale), rotation=np.pi/4)
                tgt_l.add_attr(tgt_l_trans)
                self.render_target.append(tgt_l_trans)
                self.viewer.add_geom(tgt_l)

            for t in range(self.n_target):
                tgt_r = rendering.FilledPolygon([(-target_w/2.0*scale,-target_l/2.0*scale), (-target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,target_l/2.0*scale), (target_w/2.0*scale,-target_l/2.0*scale)])
                tgt_r.set_color(*target_clrs[t][0])
                tgt_r_trans = rendering.Transform(translation=tuple((self.target_positions[t]+0.5)*100*scale), rotation=-np.pi/4)
                tgt_r.add_attr(tgt_r_trans)
                self.render_target.append(tgt_r_trans)
                self.viewer.add_geom(tgt_r)

            #-------------------draw target----contours
            self.render_target_contour = []
            for t in range(self.n_target):
                tgt_l = rendering.FilledPolygon([(-target_in_w/2.0*scale,-target_in_l/2.0*scale), (-target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,-target_in_l/2.0*scale)])
                tgt_l.set_color(*target_clrs[t][1])
                tgt_lc_trans = rendering.Transform(translation=tuple((self.target_positions[t]+0.5)*100*scale), rotation=np.pi/4)
                tgt_l.add_attr(tgt_lc_trans)
                self.render_target_contour.append(tgt_lc_trans)
                self.viewer.add_geom(tgt_l)

            for t in range(self.n_target):
                tgt_r = rendering.FilledPolygon([(-target_in_w/2.0*scale,-target_in_l/2.0*scale), (-target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,target_in_l/2.0*scale), (target_in_w/2.0*scale,-target_in_l/2.0*scale)])
                tgt_r.set_color(*target_clrs[t][1])
                tgt_rc_trans = rendering.Transform(translation=tuple((self.target_positions[t]+0.5)*100*scale), rotation=-np.pi/4)
                tgt_r.add_attr(tgt_rc_trans)
                self.render_target_contour.append(tgt_rc_trans)
                self.viewer.add_geom(tgt_r)

            #-------------------draw line-----------------
            for l in range(1, self.y_len):
                line = rendering.Line((0.0, l*100*scale), (screen_width, l*100*scale))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            for l in range(1, self.y_len):
                line = rendering.Line((l*100*scale, 0.0), (l*100*scale, screen_width))
                line.linewidth.stroke = 4
                line.set_color(0.50, 0.50, 0.50)
                self.viewer.add_geom(line)

            # line = rendering.Line((0.0, 0.0), (0.0, screen_width))
            # line.linewidth.stroke = 60
            # line.set_color(0.0, 0.0, 0.0)
            # self.viewer.add_geom(line)

            # line = rendering.Line((0.0, 0.0), (screen_width, 0.0))
            # line.linewidth.stroke = 60
            # line.set_color(0.0, 0.0, 0.0)
            # self.viewer.add_geom(line)

            # line = rendering.Line((screen_width, 0.0), (screen_width, screen_width))
            # line.linewidth.stroke = 60
            # line.set_color(0.0, 0.0, 0.0)
            # self.viewer.add_geom(line)

            # line = rendering.Line((0.0, screen_width), (screen_width, screen_width))
            # line.linewidth.stroke = 60
            # line.set_color(0.0, 0.0, 0.0)
            # self.viewer.add_geom(line)

        for i in range(self.n_target):
            self.render_target[i].set_translation((self.target_positions[i][0]+0.5)*100*scale, (self.target_positions[i][1]+0.5)*100*scale)
            self.render_target[i+self.n_target].set_translation((self.target_positions[i][0]+0.5)*100*scale, (self.target_positions[i][1]+0.5)*100*scale)
            self.render_target_contour[i].set_translation((self.target_positions[i][0]+0.5)*100*scale, (self.target_positions[i][1]+0.5)*100*scale)
            self.render_target_contour[i+self.n_target].set_translation((self.target_positions[i][0]+0.5)*100*scale, (self.target_positions[i][1]+0.5)*100*scale)

        for i in range(self.n_agent):
            self.render_agents[i].set_translation((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale)
            self.render_agents[i+self.n_agent].set_translation((self.agent_positions[i][0]+0.5)*100*scale, (self.agent_positions[i][1]+0.5)*100*scale)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')
