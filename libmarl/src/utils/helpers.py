from gym import spaces


def get_obs_space_dim(space):
    if isinstance(space, spaces.Box):
        return space.shape[0]
    elif isinstance(space, spaces.Discrete) or isinstance(space, spaces.MultiBinary):
        return space.n
    else:
        assert "space type not specified" == 0
