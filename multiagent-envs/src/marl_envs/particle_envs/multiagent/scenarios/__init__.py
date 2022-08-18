import imp
import os.path as osp


def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    return imp.load_source('', pathname)

class ConfigNotFoundException(Exception): pass

def get_config(name):
    if name == 'merge':
        return {
            "n_agents": 2,
            "agents_x": [-0.9, -0.9],
            "agents_y": [0.2, -0.2],
            "landmarks_x": [0.9, 0.9],
            "landmarks_y": [-0.2, 0.2],
            "initial_std": 0.05
        }
    elif name == 'cross':
        return {
            "n_agents": 4,
            "agents_x": [-0.9, 0.9, 0.15, -0.15],
            "agents_y": [-0.15, 0.15, -0.9, 0.9],
            "landmarks_x": [0.9, -0.9, 0.15, -0.15],
            "landmarks_y": [-0.15, 0.15, 0.9, -0.9],
            "initial_std": 0
        }
    elif name == 'antipodal':
        return {
            "n_agents": 4,
            "agents_x": [-0.9, 0.9, -0.9, 0.9],
            "agents_y": [-0.9, 0.9, 0.9, -0.9],
            "landmarks_x": [0.9, -0.9, 0.9, -0.9],
            "landmarks_y": [0.9, -0.9, -0.9, 0.9],
            "initial_std": 0
        }
    else:
        raise ConfigNotFoundException('{} is not a valid advanced_spread environment config, valid config names are "merge", "cross" and "antipodal"'.format(name))
