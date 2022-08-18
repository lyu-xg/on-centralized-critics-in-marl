import numpy as np
from marl_envs.particle_envs.multiagent.core import World, Agent, Landmark
from marl_envs.particle_envs.multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self, obs_r=1.0, obs_resolution=8, flick_p=0.0, enable_boundary=False):
        world = World()
        # set any world properties first
        world.dim_c = 2
        num_agents = 3
        num_landmarks = 3
        world.collaborative = True
        world.enable_boundary = enable_boundary

        # add agents
        world.agents = [Agent() for i in range(num_agents)]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i
            agent.collide = True
            agent.silent = True
            agent.size = 0.15
            agent.obs_range = obs_r
            agent.obs_resolution = obs_resolution
            agent.obs_flick_p = flick_p
        # add landmarks
        world.landmarks = [Landmark() for i in range(num_landmarks)]
        for i, landmark in enumerate(world.landmarks):
            landmark.name = 'landmark %d' % i
            landmark.collide = False
            landmark.movable = False

        # make initial conditions
        self.reset_world(world)
        return world

    def reset_world(self, world):
        # random properties for agents
        for i, agent in enumerate(world.agents):
            agent.color = np.array([0.35, 0.35, 0.85])
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmark.color = np.array([0.25, 0.25, 0.25])
        # set random initial states
        for agent in world.agents:
            agent.state.p_pos = self._init_agent_pos(world)
            agent.state.p_vel = np.zeros(world.dim_p)
            agent.state.c = np.zeros(world.dim_c)
        for i, landmark in enumerate(world.landmarks):
            landmark.state.p_pos = self._init_landmark_pos(world)
            landmark.state.p_vel = np.zeros(world.dim_p)

    def _init_agent_pos(self, world):
        valid = False
        while not valid:
            pos = np.random.uniform(-1, +1, world.dim_p)
            collide = False
            for agent in world.agents:
                if agent.state.p_pos is not None:
                    delta_pos = pos - agent.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = agent.size*2 + 0.1 # agents have same size
                    if dist < dist_min:
                        collide = True
                        break
            if not collide:
                valid = True
        return pos

    def _init_landmark_pos(self, world):
        valid = False
        while not valid:
            pos = np.random.uniform(-1, +1, world.dim_p)
            collide = False
            for ld in world.landmarks:
                if ld.state.p_pos is not None:
                    delta_pos = pos - ld.state.p_pos
                    dist = np.sqrt(np.sum(np.square(delta_pos)))
                    dist_min = world.agents[0].size*2+0.1 # agents have same size
                    if dist < dist_min:
                        collide = True
                        break
            if not collide:
                valid = True
        return pos

    def benchmark_data(self, agent, world):
        rew = 0
        collisions = 0
        occupied_landmarks = 0
        min_dists = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            min_dists += min(dists)
            rew -= min(dists)
            if min(dists) < 0.1:
                occupied_landmarks += 1
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent):
                    rew -= 1
                    collisions += 1
        return (rew, collisions, min_dists, occupied_landmarks)

    def is_collision(self, agent1, agent2):
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return True if dist <= dist_min else False

    def reward(self, agent, world):
        # Agents are rewarded based on minimum agent distance to each landmark, penalized for collisions
        rew = 0
        for l in world.landmarks:
            dists = [np.sqrt(np.sum(np.square(a.state.p_pos - l.state.p_pos))) for a in world.agents]
            rew -= min(dists)
        if agent.collide:
            for a in world.agents:
                if self.is_collision(a, agent) and (a.name != agent.name):
                    rew -= 1.0
        return rew

    def observation(self, agent, world):
        # get positions of all entities in this agent's reference frame
        entity_pos = []
        for entity in world.landmarks:  # world.entities:
            if not agent.obs_flick_p:
                dist = np.sqrt(np.sum(np.square(entity.state.p_pos - agent.state.p_pos)))
                if dist <= agent.obs_range:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(np.zeros(world.dim_p))
            else:
                if np.random.random(1) > agent.obs_flick_p:
                    entity_pos.append(entity.state.p_pos - agent.state.p_pos)
                else:
                    entity_pos.append(np.zeros(world.dim_p))

        # entity colors
        entity_color = []
        for entity in world.landmarks:  # world.entities:
            entity_color.append(entity.color)

        # communication of all other agents
        other_pos = []
        for other in world.agents:
            if other is agent: continue
            if not agent.obs_flick_p:
                dist = np.sqrt(np.sum(np.square(other.state.p_pos - agent.state.p_pos)))
                if dist <= agent.obs_range:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                else:
                    other_pos.append(np.zeros(world.dim_p))
            else:
                if np.random.random(1) > agent.obs_flick_p:
                    other_pos.append(other.state.p_pos - agent.state.p_pos)
                else:
                    other_pos.append(np.zeros(world.dim_p))

        return np.around(np.concatenate([agent.state.p_vel] + [agent.state.p_pos] + entity_pos + other_pos), decimals=agent.obs_resolution)

    def state(self, world):
        
        # agent pos
        agent_pos = [agent.state.p_pos for agent in world.agents]

        # agent vel
        agent_vel = [agent.state.p_vel for agent in world.agents]

        # landmarks pos
        entity_pos = [entity.state.p_pos for entity in world.landmarks]

        return np.around(np.concatenate(agent_pos + agent_vel + entity_pos), decimals=world.agents[0].obs_resolution)

    def env_info(self, world):
        return {'state_shape': len(self.state(world))}
