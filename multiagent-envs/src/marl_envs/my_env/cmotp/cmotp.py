# this version of CMOTP is based on Gregory's version at https://github.com/gjp1203/nui_in_madrl
import numpy as np
import copy
from sys import exit
import random
import cv2
from gym import spaces

def centered(A: np.ndarray, B: np.ndarray):
    '''
        put matrix A in the center of matrix B.
        A's shape has to be smaller or equal to B's in all dimensions.
    '''
    B = np.copy(B)
    bx, by = B.shape
    ax, ay = A.shape
    lower_x = bx // 2 - ax // 2
    upper_x = lower_x + ax
    lower_y = by // 2 - ay // 2
    upper_y = lower_y + ay
    B[lower_x:upper_x, lower_y:upper_y] = A
    return B

class CMOTP():
    """ Cooperative multi-agent transporation problem. """
    def __init__(self, version=1, terminate_step=10000, is_local_obs=True, local_obs_shape=(3,3), wall_punish=0):
        '''
        :param version: Integer specifying which configuration to use
        :param is_local_obs: if True, observation becomes local -- centered around the agent
        :param local_obs_shape: dimension of the local observation
        :param wall_punish: punishment if agent hit a wall
        '''
        is_valid_dim = lambda d: type(d) is int and d > 0 and d % 2
        if is_local_obs and (len(local_obs_shape) != 2 or not all(is_valid_dim(d) for d in local_obs_shape)):
            print('cannot create a local observation window with shape', local_obs_shape)
            print('for centered local observations, shape dims must be positive odd integers')
            exit('invalid CMOTP local obs window')
        
        self._is_local_obs, self._local_obs_shape = is_local_obs, local_obs_shape
        self._wall_punish = wall_punish

        if version == 1 or version == '1': # Standard
            from .envconfig_v1 import EnvConfigV1
            self.c = EnvConfigV1()
        if version == 2 or version == '2': # Bottleneck
            from .envconfig_v2 import EnvConfigV2
            self.c = EnvConfigV2()
        if version == 3 or version == '3': # Stochastic
            from .envconfig_v3 import EnvConfigV3
            self.c = EnvConfigV3()

        # Fieldnames for stats
        self.fieldnames = ['Episode',
                           'Steps',
                           'Coordinated_Steps', 
                           'Coop_Transport_Steps',
                           'Total_Reward',
                           'Goods_Delivered']

        self.episode_count = 0      # Episode counter
        self.terminate_step = terminate_step
        
        # Used to add noise to each cell
        self.ones = np.ones(self.c.DIM, dtype=np.float64)

        self.obs_shape = self._local_obs_shape if self._is_local_obs else self.c.DIM

        self.observation_space = spaces.Box(low=0.0, high=255.0, shape=self.obs_shape, dtype=np.float64)
        self.action_space = spaces.Discrete(self.c.ACTIONS)
        self.state_space = self.observation_space
        self.n_agent=2


    def obs_with_margin(self, obs):
        # add white margin to obs to match in the state size
        canvas = self.ones * 255
        return centered(obs, canvas)

    def render(self):
        '''
        Used to render the env.
        '''
        r = 16 # Number of times the pixel is to be repeated
        if self._is_local_obs:
            O = [centered(o, self.ones*255) for o in self.getObservations()]
            S = self.getNoisyState()
            seperater = np.ones((self.c.GH, 1)) * 255
            image = np.hstack([S] + [np.hstack((seperater,o)) for o in O])
        else:
            image = self.getNoisyState()

        try:
            img = np.repeat(np.repeat(image, r, axis=0), r, axis=1).astype(np.uint8)
            cv2.imshow('image', img)
            k = cv2.waitKey(50)
            if k == 27:         # If escape was pressed exit
                cv2.destroyAllWindows()
        except AttributeError:
            pass

    def stats(self):
        '''
        Returns stats dict
        '''
        stats = {'Episode': str(self.episode_count), 
                 'Steps': str(self.steps), 
                 'Coordinated_Steps':str(self.coordinatedTransportSteps),
                 'Coop_Transport_Steps': str(self.coopTransportSteps),
                 'Total_Reward': str(self.reward_total),
                 'Goods_Delivered':str(self.delivered)}
        return stats

    def reset(self):
        '''
        Reset everything. 
        '''
        # Set up the state array:
        # 0 = obstacles, 1 = goods, 2 = agents, 3 = self
        self.s_t = np.zeros(self.c.DIM, dtype=np.float64)

        # Obstacles, agents and goods are initialised:
        self.setObstacles()
        self.initGoods()
        self.initAgents()

        # Used to keep track of the reward total acheived throughout 
        # the episode:
        self.reward_total = 0.0

        # Episode counter is incremented:
        self.episode_count += 1
   
        # For statistical purposes:
        # Step counter for the episode is initialised
        self.steps = 0 

        # Number of steps the goods is carried by both agents
        self.coopTransportSteps = 0 

        # Moves taken in the same direction while carrying the goods
        self.coordinatedTransportSteps = 0 
      
        return self.getObservations()

    def terminal(self):
        '''
        Find out if terminal conditions have been reached.
        '''
        return self.delivered or self.steps >= self.terminate_step

    def step(self, actions):
        '''
        Change environment state based on actions.
        :param actions: list of integers
        '''
        # Agents move according to actions selected
        collisions = self.moveAgents(actions)

        # Observations are loaded for each agent
        observations = self.getObservations()

        # Goods pickup checks to see if the agents have
        # reached a position where they can grasp the goods.                    
        self.goodsPickup()

        # Counters are incremented:
        self.steps += 1 
        if self.holding_goods[0] == True and self.holding_goods[1] == True:
            self.coopTransportSteps += 1
            # Check if goods have reached dzone
            rewards = self.goodsDelivered()
        else:
            rewards = 0.0
        if collisions:
            rewards -= self._wall_punish

        self.reward_total += rewards # Team game, so one r is sufficient

        n_agents = self.c.NUMBER_OF_AGENTS
        # return observations, rewards, self.terminal()
        return actions, observations, [rewards]*n_agents, [self.terminal()]*n_agents, [1]*n_agents, self.stats()

    def initGoods(self):
        '''
        Goods position and carrier ids are initialised
        '''
      
        # x and y coordinates for each of the goods
        self.goods_x = self.c.GOODS_X
        self.goods_y = self.c.GOODS_Y
 
        # Goods l and r will store the ID of the agents on the 
        # left and right hand side of the goods once it has been picked up.
        # Initially set to -1. 
        self.goods_l = -1       
        self.goods_r = -1      
        
        # Each goods has a delivered status that is initially set to false. 
        self.delivered = False

        # The goods channel within the state matrix is 1, and each 
        # coordinates with an item of goods is set to the value used
        # to represent a goods item (self.GOODS). 
        self.s_t[self.goods_y][self.goods_x] += self.c.GOODS
	
    def initAgents(self):
        '''
        Method for initialising the required number of agents and 
        positionsing them on designated positions within the grid
        '''

        # Agent x and y positions can be set in the following lists.
        # Defaults are the bottom right and left corners, for two agents, 
        # since this is the way most CMOTPs start
        self.agents_x = copy.deepcopy(self.c.AGENTS_X)
        self.agents_y = copy.deepcopy(self.c.AGENTS_Y)

        # List used to indicate whether or not the agents are holding goods
        self.holding_goods= [False for i in range(self.c.NUMBER_OF_AGENTS)]
   
        # Agents are activated within the agent channel (2) of the gridworld
        # matrix.
        for i in range(self.c.NUMBER_OF_AGENTS):
                self.s_t[self.agents_y[i]][self.agents_x[i]] += self.c.AGENTS[i]

    def setObstacles(self):
        '''
        Method used to initiate the obstacles within the environment 
        '''
        for y, x in self.c.OBSTACLES_YX:
            self.s_t[y][x] = self.c.OBSTACLE

    def goodsPickup(self):
        '''
        Method for picking up the goods, if the agents
        find themselves in positions adjecent to the goods.
        '''
        # For each of a goods we check whether agents are 
        # in a position to pick the goods up:

        # Check to see if there is an agent on the left 
        # handside to pickup the goods:
        if (self.goods_x >= 1 and 
            self.goods_l == -1 and 
            self.s_t[self.goods_y][self.goods_x - 1] > 0):

            for j in range(self.c.NUMBER_OF_AGENTS):
                if (self.agents_x[j] == self.goods_x - 1 and 
                    self.agents_y[j] == self.goods_y and 
                    self.holding_goods[j] == False):
                    self.goods_l = j
                    self.holding_goods[j] = True

        # Check to see if there is an agent on the right 
        # handside to pickup the goods:
        if (self.goods_x < self.c.GW-1 and  
            self.goods_r == -1 and 
            self.s_t[self.goods_y][self.goods_x + 1] > 0):

            for j in range(self.c.NUMBER_OF_AGENTS):
                if (self.agents_x[j] == self.goods_x + 1 and  
                    self.agents_y[j] == self.goods_y and 
                    self.holding_goods[j] == False):
                    self.goods_r = j
                    self.holding_goods[j] = True

    def goodsDelivered(self):
        '''
        Method to check one of the goods 
        has been deliverd to the dropzone
        '''
        for (dropX, dropY, r) in self.c.DZONES:
            if self.goods_x == dropX and self.goods_y == dropY:
                self.delivered = True                
                self.s_t[self.goods_y][self.goods_x] -= self.c.GOODS
                self.goods_y = -1
                if self.goods_l > -1:
                    self.holding_goods[self.goods_l] = False
                if self.goods_r > -1:
                    self.holding_goods[self.goods_r] = False
                self.goods_l, self.goods_r = -1, -1
                return r()
        return 0.0

    def unsetAgents(self): 
        ''' 
        Method to release the agents from holding the goods
        '''
        if self.goods_x > -1 and self.goods_y > -1:
            self.s_t[self.goods_y][self.goods_x] = 0
            self.goods_l = -1
            self.goods_r = -1
        for i in range(self.c.NUMBER_OF_AGENTS):
                if self.agents_x[i] > -1 and self.agents_y[i] > -1:
                    self.s_t[self.agents_y[i]][self.agents_x[i]] = 0
                    self.holding_goods[i] = False

    def getNoisyState(self):
        ''' 
        Method returns noisy state.
        '''
        return self.s_t + (self.c.NOISE * self.ones *\
                           np.random.normal(self.c.MU,self.c.SIGMA, self.c.DIM))

    def getObservations(self):
        '''
        Returns centered observation for each agent
        '''
        observations = []
        noisy_state = np.copy(self.getNoisyState())
        if not self._is_local_obs:
            return [noisy_state] * self.c.NUMBER_OF_AGENTS
        padded_grid = self.pad_grid(noisy_state)
        obs_w, obs_h = self._local_obs_shape
        padding_x, padding_y = obs_w // 2, obs_h // 2
        for i in range(self.c.NUMBER_OF_AGENTS):
            # Store observation
            x, y = self.agents_x[i], self.agents_y[i]
            observations.append(padded_grid[y:y+obs_h,x:x+obs_w]) # TODO test this
        return observations

    def pad_grid(self, grid):
        obs_x, obs_y = self._local_obs_shape
        wall_world = np.ones((self.c.GH + obs_y - 1, self.c.GW + obs_x - 1)) * self.c.OBSTACLE
        return centered(grid, wall_world)

    def getDelta(self, action):
        '''
        Method that deterimines the direction 
        that the agent should take
        based upon the action selected. The
        actions are:
        'Up':0, 
        'Right':1, 
        'Down':2, 
        'Left':3, 
        'NOOP':4
        :param action: int
        '''
        if action == 0:
            return 0, -1
        elif action == 1:
            return 1, 0    
        elif action == 2:
            return 0, 1    
        elif action == 3:
            return -1, 0 
        elif action == 4:
            return 0, 0   

    def moveAgents(self, actions):
       '''
       Move agents according to actions.
       :param actions: List of integers providing actions for each agent
       '''
       for i in range(self.c.NUMBER_OF_AGENTS):
           if random.random() < self.c.WIND:
              actions[i] = random.randrange(self.c.__outs)

       collisions = False
       if self.goods_l > -1 and self.goods_r > -1:
           if actions[self.goods_l] == actions[self.goods_r]:
               self.coordinatedTransportSteps += 1
               dx, dy = self.getDelta(actions[self.goods_l])
               targetx_l = self.agents_x[self.goods_l] + dx
               targety_l = self.agents_y[self.goods_l] + dy
               targetx_r = self.agents_x[self.goods_r] + dx
               targety_r = self.agents_y[self.goods_r] + dy
               targetx_g = self.goods_x + dx
               targety_g = self.goods_y + dy 

               if dy > 0:
                   for j in range(abs(dy)):
                       if self.noCollision(targetx_l, targety_l-j) == False or\
                          self.noCollision(targetx_r, targety_r-j) == False or\
                          self.noCollision(targetx_g, targety_g-j) == False:
                           collisions = True

               elif dy < 0:
                   for j in range(abs(dy)):
                       if False == self.noCollision(targetx_l, targety_l+j) or\
                          False == self.noCollision(targetx_r, targety_r+j) or\
                          False == self.noCollision(targetx_g, targety_g+j):
                           collisions = True
               elif dx > 0 and self.noCollision(targetx_r, targety_r):
                   collisions = False
               elif dx < 0 and self.noCollision(targetx_l, targety_l):
                   collisions = False
               else:
                   collisions = True
               if collisions == False:
                   self.moveAgent(self.goods_l, targetx_l, targety_l)
                   self.moveAgent(self.goods_r, targetx_r, targety_r)
                   self.moveGoods(targetx_g, targety_g)


       for i in range(self.c.NUMBER_OF_AGENTS):
           if self.holding_goods[i] == False:
               dx, dy = self.getDelta(actions[i])
               targetx = self.agents_x[i] + dx
               targety = self.agents_y[i] + dy
               if self.noCollision(targetx, targety):
                   self.moveAgent(i, targetx, targety)

       return collisions
    
    def moveAgent(self, id, targetx, targety):
        '''
        Moves agent to target x and y
        :param targetx: Int, target x coordinate
        :param targety: Int, target y coordinate
        '''
        self.s_t[self.agents_y[id]][self.agents_x[id]] -= self.c.AGENTS[id]
        self.agents_x[id] = targetx
        self.agents_y[id] = targety
        self.s_t[self.agents_y[id]][self.agents_x[id]] += self.c.AGENTS[id]

    def moveGoods(self, targetx, targety):
        '''
        Moves goods to target x and y
        :param targetx: Int, target x coordinate
        :param targety: Int, target y coordinate
        '''
        self.s_t[self.goods_y][self.goods_x] -= self.c.GOODS
        self.goods_x = targetx
        self.goods_y = targety
        self.s_t[self.goods_y][self.goods_x] += self.c.GOODS

    def noCollision(self, x, y):
        '''
        Checks if x, y coordinate is currently empty 
        :param x: Int, x coordinate
        :param y: Int, y coordinate
        '''
        if x < 0 or x >= self.c.GW or\
           y < 0 or y >= self.c.GH or\
           self.s_t[y][x] != 0:
            return False
        else:
            return True
