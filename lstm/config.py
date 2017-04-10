# coding:utf-8

__all__ = [
    "configuration",
    "make_config",
    "update_config",
    "test"
]


config_pool = {
    'default':
        {
            # Path for trace files
            'path_trace': 'res/trace/',
            # Path for logs
            'path_log': 'log/',
            # How often should we check the output
            'log_slot': 1,
            # Show warning info or not
            'show_warning': False,
            # Level of printing detail
            # 0 means mandatory printing only
            'print_level': 3,
            # How many nodes to learn on
            'num_node': 3,
            # Dimension of sample input: 2 for [x, y]; 3 for [sec, x, y]
            'dimension_sample': 3,
            # If only a full size batch will be accepted
            'strict_batch_size': True,
            # Length of edge when coordinates need to be mapped to grid
            'grain_grid': 100
        },
    'debug':
        {

            # Nano-size net config for debugging

            # Length of observed sequence
            'length_sequence_input': 4,
            # Length of predicted sequence
            'length_sequence_output': 4,
            # Number of units in embedding layer ([x, y] -> e)
            'dimension_embed_layer': 2,
            # Number of units in hidden (LSTM) layers
            # n: single layer of n units
            # (m, n): m * layers of n units
            'dimension_hidden_layer': (2, 2),

            # """
            # Social-LSTM config
            # """
            # 'size_pool': 100,
            # # x * x of pools around each node
            # 'range_neighborhood': 4,

            # Batch Size
            'size_batch': 10,
            # Number of epochs to train the net
            'num_epoch': 2,
            # Optimization learning rate
            'learning_rate_rmsprop': .03,
            'rho_rmsprop': .9,
            'epsilon_rmsprop': 1e-8,
            # All gradients above this will be clipped
            'grad_clip': 100
        },
    'run':
        {
            # Length of observed sequence
            'length_sequence_input': 10,
            # Length of predicted sequence
            'length_sequence_output': 10,
            # Number of units in embedding layer ([x, y] -> e)
            'dimension_embed_layer': 64,
            # Number of units in hidden (LSTM) layers
            # n: single layer of n units
            # (m, n): m * layers of n units
            'dimension_hidden_layer': (10, 32),

            # """
            # Social-LSTM config
            # """
            # 'size_pool': 100,
            # # x * x of pools around each node
            # 'range_neighborhood': 4,

            # Batch Size
            'size_batch': 10,
            # Number of epochs to train the net
            'num_epoch': 50,
            # Optimization learning rate
            'learning_rate_rmsprop': .001,
            'rho_rmsprop': .95,
            'epsilon_rmsprop': 1e-8,
            # All gradients above this will be clipped
            'grad_clip': 0
        }
}


def make_config(key='run'):
    try:
        default = config_pool['default']

        if key not in config_pool or key == 'default':
            available_keys = config_pool.keys()
            available_keys.remove('default')
            raise ValueError("make_config @ config: Invalid key '%s'. Must choose from %s."
                             % (key, available_keys))

        requested = config_pool[key]
        ret = {}
        ret.update(default)
        # Requested value will overwrite default value for the same config item
        ret.update(requested)
        return ret

    except:
        raise


configuration = make_config()


def update_config(key='run', config=None):
    try:
        global configuration
        if config is None:
            configuration.update(make_config(key=key))
        elif not isinstance(config, dict):
            raise ValueError("update_config @ config: Expect <dict> while getting %s instead." % type(config))
        else:
            configuration.update(config)
    except:
        raise


def test():
    try:
        run = make_config()
        config_pool['debug']['show_warning'] = True
        debug = make_config(key='debug')
        try:
            invalid = make_config(key='default')
        except Exception, e:
            pass
        try:
            invalid = make_config(key='invalid-key')
        except Exception, e:
            pass
        update_config(config={'_': True})
    except:
        raise
