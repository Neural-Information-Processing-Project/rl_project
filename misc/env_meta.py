import numpy as np

def build_PEARL_envs(seed,
                    env_name,
                    params=None
                    ):
    '''
      Build env from PEARL
    '''
    from rlkit.envs import ENVS
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.envs.safelife.file_finder import SafeLifeLevelIterator

    training_levels = 'rlkit/envs/safelife/levels/random/append-dynamic.yaml'

    slli=SafeLifeLevelIterator(training_levels, distinct_levels=8, total_levels=-1, seed=seed)

    if env_name == 'safelife-dir':
        env_params = {
                     'level_iterator': slli
                     }

    elif env_name == 'ant-dir':
        env_params = {
                    'n_tasks' : params.n_tasks,
                    'randomize_tasks': params.randomize_tasks,
                    #"low_gear": params.low_gear,
                    "forward_backward": params.forward_backward,
                     }

    elif env_name == 'ant-goal':
        env_params = {
                    'n_tasks' : params.n_tasks,
                    'randomize_tasks': params.randomize_tasks,
                    #"low_gear": params.low_gear,
                     }

    else:
        env_params = {
                  'n_tasks' : params.n_tasks,
                  'randomize_tasks': params.randomize_tasks
                 }

    env = ENVS[env_name](**env_params)
    env.seed(seed)

    env = NormalizedBoxEnv(env)

    env.action_space.np_random.seed(seed)

    return env