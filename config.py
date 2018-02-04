# coding:utf-8

import ast

__all__ = [
    "has_config",
    "get_config",
    "remove_config",
    "filter_config",
    "test"
]


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


def filter_config(tag, config=None):
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


def update_config_from_file(path, group='default'):
    """

    :param path: Path of config file
    :param group: The keyword of desired config group.
    :return: Echo message
    """
    try:
        pfile = open(path, 'r')
        content = pfile.read()
        config_groups = ast.literal_eval(content)
        if not isinstance(config_groups, dict):
            return "Load default configuration groups from '%s'." % path

        if group not in config_groups:
            raise KeyError("Cannot find configuration group '%s'. Must choose among %s." % (group, config_groups.keys()))
        _update_(config_groups[group], source='%s:%s' % (path, group))
        return "Update configurations according to group '%s' in '%s'." % (group, path)


    except IOError:
        raise IOError("Cannot find default configuration file '%s'." % path)
    except ValueError:
        raise ValueError("Invalid configuration format in '%s'" % path)
    except:
        raise


def update_config(key, value, source, tags=None):
    """

    :param key: Configuration key. e.g. 'nodes', 'num_epoch'
    :param value: Configuration value.
    :param source: A tag for logging, to indicate the source of latest config overwriting.
    :param tags: List of tags. The 1st tag should be configuration source. e.g. 'command-line', 'import', 'runtime'
    :return: Echo message
    """
    try:
        original = get_config(key)
        _update_({key: value}, source, tags)
        return "Update configuration '%s' from %s to %s (from %s)." % (key, original, value, source)
    except:
        raise


def import_config(config, tag=None):
    """

    :param config: <Dict>
    :param tag: <str> All the keys get imported if tag is None. Otherwise only the keys with this tag do.
    :return:
    """
    try:
        # import all keys if tag is None
        if tag is not None:
            config = filter_config(tag, config=config)

        for impkey, impvalue in config.iteritems():
            update_config(impkey, impvalue, 'imported')

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

        import lstm

        update_config_from_file(path='lstm.config', group='run')

        update_config('num_epoch', 0, 'test', tags=['int'])
        # test arg `tags`
        update_config('num_epoch', 1, 'test', tags='positive')
        update_config('dimension_sample', 3, 'test', tags=['positive', 'build'])
        # test duplicate tags
        update_config('dimension_sample', 3, 'test', tags=['positive', 'imported'])
        # test arg `source`
        update_config('num_epoch', 2, 'command-line')

        # test getter
        _has = has_config('num_epoch')
        _value = get_config('num_epoch')
        _has = has_config('new-key')
        try:
            _value = get_config('new-key')
        except KeyError, e:
            pass

        print 'Fine.'

    except:
        raise
