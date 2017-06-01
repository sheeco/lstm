# coding:utf-8

__all__ = [
    "has_config",
    "get_config",
    "remove_config",
    "test"
]

config_pool = {
    # Group name
    'default':
        {
            # Path for trace files
            'path_trace': {
                'value': 'res/trace/NCSU/',
                'tags': ['path']
            },
            # Path for all the logs
            'path_log': {
                'value': 'log/',
                'tags': ['path']
            },
            # Sub log path for comparing logs
            'path_compare': {
                'value': 'compare/',
                'tags': ['path']
            },
            # Sub log path for pickle files
            'path_pickle': {
                'value': 'pickle/',
                'tags': ['path']
            },
            # Logging identifier
            # for naming of the log folder
            'identifier': {
                'value': None,
                'tags': []
            },
            # Tag for logging identifier
            'tag': {
                'value': None,
                'tags': []
            },
            # How many epochs of latest network history to keep
            # None means no limit
            # 0 means don't save network history
            'limit_network_history': {
                'value': 0,
                'tags': []
            },
            # Directory path that configs & parameters get imported from
            'path_import': {
                'value': None,
                'tags': ['path']
            },
            # File path that which parameters get unpickled & imported from
            'file_unpickle': {
                'value': None,
                'tags': ['path']
            },

            # How many nodes to learn on
            'num_node': {
                'value': 2,
                'tags': ['build']
            },
            # Selected nodes
            # This will override 'num_node'
            'nodes': {
                'value': ['31'],
                'tags': []
            },
            # Dimension of sample input: 2 for [x, y]; 3 for [sec, x, y]
            'dimension_sample': {
                'value': 3,
                'tags': ['build']
            },
            # Whether to map the coordinates to certain grid
            'sample_gridding': {
                'value': False,
                'tags': []
            },
            # Length of edge when coordinates need to be mapped to grid
            'scale_grid': {
                'value': 50,
                'tags': []
            },
            # Maximum deviation to be considered a hit (exclusive)
            'hit_range': {
                'value': 50,
                'tags': []
            },
            # Decoding scheme
            'decode_scheme': {
                # Among ['binorm', 'euclidean']
                'value': 'binorm',
                'tags': ['build']
            },
            # Sharing scheme for LSTMS of multiple nodes
            'share_scheme': {
                # Among ['parameter', 'input', 'olstm', 'none']
                # 'parameter' means all the nodes share the same set of parameters
                # 'input' means all the nodes share the embedded sample input with each other
                # 'olstm' means Occupancy Map sharing
                # 'none' means neither
                'value': 'input',
                'tags': ['build']
            },
            # Edge length of single neighborhood grid in social pooling
            'scale_pool': {
                'value': 100,
                'tags': ['build']
            },
            # Maximum number of grids to consider in each direction in social pooling
            # meaning pool dimension would be (2 * r, 2 * r)
            'range_pool': {
                'value': 5,
                'tags': ['build']
            },
            # If only a full size batch will be accepted
            'strict_batch_size': {
                'value': False,
                'tags': []
            },
            # Devision of train set & test set
            # <float> within (0, 1): x = train set size / total
            # <int> above 1: exact size of train set (in terms of sample)
            'trainset': {
                'value': .5,
                'tags': []
            },

            # Training scheme
            'train_scheme': {
                # Among ['rmsprop', 'adagrad', 'momentum', 'nesterov']
                'value': 'rmsprop',
                'tags': ['train']
            },
            # Decrease learning rate if training has failed
            # <float> within (0, 1): new = old * x
            # <float> within (-1, 0): new = old + x
            'adaptive_learning_rate': {
                'value': .5,
                'tags': ['train']
            },
            # Decrease gradient clipping if training has failed
            # , only works when gradient clipping is applied
            # negative <int>: new = old + x
            'adaptive_grad_clip': {
                'value': -100,
                'tags': ['train']
            },
            # Learning rate for training
            # Used for any training scheme
            'learning_rate': {
                'value': .005,
                'tags': ['train']
            },
            # All gradients above this will be clipped during training
            'grad_clip': {
                'value': 0,
                'tags': ['train']
            },

            # Parameters for RMSProp
            'rho': {
                'value': .9,
                'tags': ['train', 'rmsprop']
            },
            # Parameters for RMSProp / AdaGrad
            'epsilon': {
                'value': 1e-8,
                'tags': ['train', 'rmsprop', 'adagrad']
            },

            # Parameters for SGD with Momentum / Nesterov Momentum
            'momentum': {
                'value': .9,
                'tags': ['train', 'momentum', 'nesterov']
            },

        },

    'debug':
        # Nano-size net config for debugging
        {
            # Length of observed sequence
            'length_sequence_input': {
                'value': 4,
                'tags': ['build']
            },
            # Length of predicted sequence
            'length_sequence_output': {
                'value': 1,
                'tags': ['build']
            },
            # Number of units in embedding layer ([x, y] -> e)
            'dimension_embed_layer': {
                'value': 2,
                'tags': ['build']
            },
            # Number of units in hidden (LSTM) layers
            # n: single layer of n units
            # (m, n): m * layers of n units
            'dimension_hidden_layer': {
                'value': (2, 2),
                'tags': ['build']
            },

            # Batch Size
            'size_batch': {
                'value': 10,
                'tags': []
            },
            # Number of epochs to train the net
            'num_epoch': {
                'value': 2,
                'tags': []
            },
            # Tryout after every n * epochs of training
            'tryout_frequency': {
                'value': 1,
                'tags': []
            },
        },

    'run':
        {
            # Length of observed sequence
            'length_sequence_input': {
                'value': 10,
                'tags': ['build']
            },
            # Length of predicted sequence
            'length_sequence_output': {
                'value': 1,
                'tags': ['build']
            },
            # Number of units in embedding layer ([x, y] -> e)
            'dimension_embed_layer': {
                'value': 64,
                'tags': ['build']
            },
            # Number of units in hidden (LSTM) layers
            # n: single layer of n units
            # (m, n): m * layers of n units
            'dimension_hidden_layer': {
                'value': (10, 32),
                'tags': ['build']
            },

            # Batch Size
            'size_batch': {
                'value': 10,
                'tags': []
            },
            # Number of epochs to train the net
            'num_epoch': {
                'value': 300,
                'tags': []
            },
            # Tryout after every n * epochs of training
            'tryout_frequency': {
                'value': 10,
                'tags': []
            },
        }
}


def _get_config_from_pool_(group='default'):
    try:
        if group not in config_pool:
            available_keys = config_pool.keys()
            raise ValueError("Invalid group '%s'. Must choose from %s." % (group, available_keys))

        return config_pool[group]

    except:
        raise


global_configuration = {}


def _update_(config, source, tags=None):
    try:

        global global_configuration
        tags = [] if tags is None else tags
        tags = [tags] if not isinstance(tags, list) else tags
        for key, content in config.iteritems():

            # keep compatible with old version of config.log
            if not isinstance(content, dict) \
                    or 'value' not in content:
                content = {'value': content}

            # change value
            content['source'] = source

            # add tags given by arg
            content['tags'] = content['tags'] + [tag for tag in tags if tag not in content['tags']] \
                if 'tags' in content else tags

            # add to current tags uniquely
            # & copy other inkeys directly
            if key in global_configuration:
                for inkey, invalue in global_configuration[key].iteritems():
                    if inkey in ('value', 'source'):
                        continue
                    elif inkey in ('tags'):
                        content['tags'] = invalue + [tag for tag in content['tags'] if tag not in invalue] \
                            if 'tags' in content else invalue
                    else:
                        content[inkey] = invalue

            if len(content['tags']) == 0:
                content.pop('tags')
            global_configuration.update({key: content})

    except:
        raise


def has_config(key):
    try:
        global global_configuration
        return key in global_configuration
    except:
        raise


def get_config(key=None):
    try:
        global global_configuration
        if key is None:
            return global_configuration
        elif not has_config(key):
            raise KeyError("Key '%s' does not exist." % key)
        return global_configuration[key]['value']
    except:
        raise


def _filter_config_(tag, config=None):
    """
    Filter configs with certain tag, from `config` (global config by default).
    :param tag: <str>
    :param config: <dict> or None
    :return: <dict> Filtered config
    """
    try:
        global global_configuration
        config = global_configuration if config is None else config
        if tag is None:
            return config

        ret = {}
        for key, content in config.iteritems():
            if 'tags' in content \
                    and tag in content['tags']:
                ret[key] = content
        return ret

    except:
        raise


def _update_config_from_pool_(group='default'):
    try:
        if group is not None:
            _update_(_get_config_from_pool_(group=group), source=group)
    except:
        raise


def _update_config_(key, value, source, tags=None):
    """

    :param key: Configuration key. e.g. 'nodes', 'num_epoch'
    :param value: Configuration value.
    :param tags: List of tags. The 1st tag should be configuration source. e.g. 'command-line', 'import', 'runtime'
    :return:
    """
    try:
        _update_({key: value}, source, tags)
    except:
        raise


def _import_config_(config, tag=None):
    """

    :param config: <Dict>
    :param tag: <str> All the keys get imported if tag is None. Otherwise only the keys with this tag do.
    :return:
    """
    try:
        # import all keys if tag is None
        if tag is not None:
            config = _filter_config_(tag, config=config)

        for impkey, impvalue in config.iteritems():
            _update_config_(impkey, impvalue, 'imported')

    except:
        raise


def remove_config(key):
    try:
        global global_configuration
        if key in global_configuration:
            global_configuration[key]['value'] = None
    except:
        raise


def test():
    try:
        print 'Testing config ... ',

        debug = _get_config_from_pool_(group='debug')
        try:
            invalid = _get_config_from_pool_(group='invalid-key')
        except Exception, e:
            pass

        _update_config_from_pool_(group='run')

        _update_config_('num_epoch', 0, 'test', tags=['int'])
        # test arg `tags`
        _update_config_('num_epoch', 1, 'test', tags='positive')
        _update_config_('dimension_sample', 3, 'test', tags=['positive', 'build'])
        # test duplicate tags
        _update_config_('dimension_sample', 3, 'test', tags=['positive', 'imported'])
        # test arg `source`
        _update_config_('num_epoch', 2, 'command-line')

        # test getter
        _has = has_config('num_epoch')
        _value = get_config('num_epoch')
        _has = has_config('new-key')
        try:
            _value = get_config('new-key')
        except Exception, e:
            pass

        # test new key
        _update_config_('new-key', None, 'test')
        _value = get_config('new-key')
        print 'Fine'

    except:
        raise
