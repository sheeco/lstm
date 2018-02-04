# coding:utf-8

import utils
import numpy


def read_predictions_from_compare_file(path):
    try:
        if not utils.assertor.assert_exists(path, raising=False):
            return None

        lines = utils.filer.read_lines(path)
        predictions = {}

        # Read a triple per line & save to dict
        for i in range(0, len(lines)):
            sec, x_pred, y_pred, x_fact, y_fact = [numpy.float16(x) for x in lines[i].split()][0:5]
            predictions[sec] = numpy.array([x_pred, y_pred])

        return predictions
    except:
        raise


def save_triples_to_file(triples, path):
    try:
        sorted = utils.sorted_items(triples)
        lines = ['%s\t%s\t%s\n' % (sec, xy[0], xy[1]) for sec, xy in sorted]
        utils.filer.write_lines(path, lines)
    except:
        raise


def find_subpath_by_identifier(identifier, path):
    try:
        dirs = utils.filer.list_directory(path)
        for subpath in dirs:
            if subpath.find(identifier) >= 0:
                return utils.filer.format_subpath(path, subpath=subpath, isfile=False)
        raise IOError("Cannot find any folder with given identifier '%s'." % identifier)
    except:
        raise


def validate_logpath(path_or_identifier):
    try:
        path_logroot = utils.get_config('path_logroot')

        # e.g. 'log/[case]2018-01-18-20-53-10'
        if utils.filer.if_exists(path_or_identifier):
            return path_or_identifier
        # e.g. '[case]2018-01-18-20-53-10' -> 'log/[case]2018-01-18-20-53-10'
        elif utils.filer.if_exists(utils.filer.format_subpath(path_logroot, path_or_identifier, isfile=False)):
            return utils.filer.format_subpath(path_logroot, path_or_identifier, isfile=False)
        # e.g. '2018-01-18-20-53-10' -> 'log/[case]2018-01-18-20-53-10'
        else:
            return find_subpath_by_identifier(path_or_identifier, path_logroot)
    except:
        raise


# todo add dump_parameters()

def dump_predictions(frompath, epoch, dumpname, topath=None):
    """
    Find predictions of given epoch in training & testing logs, and dump them to given topath.
    :param frompath: Log folder / the identifier of an execution
    :param epoch: <int> epoch entry
    :param dumpname: Filename for dumped files e.g. node id
    :param topath: Destination path for dumping
    """
    try:
        frompath = validate_logpath(frompath)
        frompath = utils.filer.format_subpath(frompath, subpath=utils.get_config('path_compare'), isfile=False)
        topath = utils.get_sublogger().log_path if topath is None else topath

        logname_train = 'train-epoch%d.log' % epoch
        logname_test = 'test-epoch%d.log' % epoch

        dumpname_train = '%s.train.trace' % dumpname
        dumpname_test = '%s.test.trace' % dumpname
        dumpname_full = '%s.full.trace' % dumpname

        dumpname_train = utils.filer.format_subpath(topath, dumpname_train)
        dumpname_test = utils.filer.format_subpath(topath, dumpname_test)
        dumpname_full = utils.filer.format_subpath(topath, dumpname_full)

        path_train = utils.filer.format_subpath(frompath, subpath=logname_train, isfile=True)
        pred_train = read_predictions_from_compare_file(path_train)
        if pred_train is not None:
            save_triples_to_file(pred_train, dumpname_train)
            utils.xprint("Training predictions in '%s' are dumped to '%s'." % (path_train, dumpname_train), newline=True)
        else:
            utils.warn("Cannot find file '%s' (for training epoch %d)." % (path_train, epoch))

        path_test = utils.filer.format_subpath(frompath, subpath=logname_test, isfile=True)
        pred_test = read_predictions_from_compare_file(path_test)
        if pred_test is not None:
            save_triples_to_file(pred_test, dumpname_test)
            utils.xprint("Testing predictions in '%s' are dumped to '%s'." % (path_test, dumpname_test), newline=True)

            # Both are available
            if pred_train is not None:
                utils.filer.write(dumpname_full, utils.filer.read(dumpname_train))
                utils.filer.write(dumpname_full, utils.filer.read(dumpname_test))
                utils.xprint("Full predictions in '%s' & '%s' are dumped to '%s'." % (path_train, path_test, dumpname_full), newline=True)

        # Both are unavailable
        elif pred_train is None:
            raise IOError("Cannot find file '%s' & '%s'." % (path_train, path_test))

        else:
            utils.warn("Cannot find file '%s' (for testing epoch %d)." % (path_test, epoch))

    except:
        raise


def process_command_line_args(args):
    """
    e.g. dump.py [-f | --from <log-folder|identifier>] [-e | --epoch <iepoch>] [-n | --name <dump-filename>] [-t | --to <dump-folder>]
    :return:
    """
    try:
        # short-opts: "ha:i" means opt '-h' & '-i' don't take arg, '-a' does take arg
        # long-opts: ["help", "add="] means opt '--add' does take arg
        pairs, unknowns = utils.get_opt(args, "f:e:n:t:", longopts=["from=", "epoch=", "name=", "to="])

        arg_from, arg_epoch, arg_name, arg_to = None, None, None, None
        mandatory_args = [('-f', '--from'),
                          ('-e', '--epoch'),
                          ('-n', '--name')]
        optional_args = [('-t', '--to'),
                         ('-c', '--config')]

        opts = [each_pair[0] for each_pair in pairs]
        for some_arg in mandatory_args:
            # if some_opt[2] is None:
            if some_arg[0] not in opts and some_arg[1] not in opts:
                raise ValueError("Argument '%s|%s' is mandatory." % some_arg)

        for opt, val in pairs:
            if opt in mandatory_args[0]:
                try:
                    val = utils.literal_eval(val)
                except ValueError, e:
                    pass
                except SyntaxError, e:
                    pass

                val = str(val)
                if utils.assertor.assert_nonempty_str(val):
                    arg_from = val

            elif opt in mandatory_args[1]:
                try:
                    val = utils.literal_eval(val)
                except ValueError, e:
                    pass
                except SyntaxError, e:
                    pass

                if utils.assertor.assert_type(val, int):
                    arg_epoch = val

            elif opt in mandatory_args[2]:
                try:
                    val = utils.literal_eval(val)
                except ValueError, e:
                    pass
                except SyntaxError, e:
                    pass

                val = str(val)
                if utils.assertor.assert_nonempty_str(val):
                    arg_name = val

            elif opt in optional_args[0]:
                try:
                    val = utils.literal_eval(val)
                except ValueError, e:
                    pass
                except SyntaxError, e:
                    pass

                val = str(val)
                if utils.assertor.assert_nonempty_str(val):
                    arg_to = utils.filer.validate_path_format(val)

            elif opt in optional_args[1]:
                utils.assertor.assert_type(val, dict)
                for key, value in val.items():
                    utils.update_config(key, value, 'command-line', silence=False)

            else:
                raise ValueError("Unknown option '%s'." % opt)

        # if len(unknowns) > 0:
        if unknowns:
            raise ValueError("Unknown option(s) %s." % unknowns)

        return arg_from, arg_epoch, arg_name, arg_to

    except:
        raise


def test():

    utils.xprint("Test arg processing ...", newline=True)
    try:
        process_command_line_args(["-f", "'[case]2018-01-18-20-53-10'", "-e", "1", "-n", "'demo'", "-t", "'res/dump/demo'"])
        process_command_line_args(["-f", "'[case]2018-01-18-20-53-10'", "-e", "1", "-n", "'demo'"])
    except:
        raise
    try:
        process_command_line_args(["-e", "1", "-n", "'demo'", "-t", "'res/dump'"])
    except ValueError, e:
        utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)
    try:
        process_command_line_args(["-f", "'[case]2018-01-18-20-53-10'", "-e", "invalid-epoch", "-n", "'demo'"])
    except ValueError, e:
        utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)

    utils.xprint("Fine.", newline=True)

    utils.xprint("Test dumping ...", newline=True)
    try:
        dump_predictions('log/[case]2018-01-18-20-53-10', 0, 'demo0',
                         'res/dump/demo')
        dump_predictions('[case]2018-01-18-20-53-10', 1, 'demo1', 'res/dump/demo')
        dump_predictions('2018-01-18-20-53-10', 2, 'demo2')
    except:
        raise

    try:
        dump_predictions('[mock]invalid-path', 1, 'demo', 'res/dump/demo')
    except IOError, e:
        utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)
    try:
        dump_predictions('log/[case]2018-01-18-20-53-10', 3, 'demo', 'res/dump/demo')
    except IOError, e:
        utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)

    utils.xprint('Fine.', newline=True)


def demo():
    args = utils.get_sys_args()[1:]
    utils.get_sublogger().log_args(args)

    arg_from, arg_epoch, arg_name, arg_to = process_command_line_args(args)

    dump_predictions(arg_from, arg_epoch, arg_name, topath=arg_to)


if __name__ == '__main__':
    try:
        timestamp = utils.get_timestamp()
        utils.sub_logger = utils.Logger(identifier=timestamp)
        utils.get_sublogger().register_console()

        demo()
        # test()

    except Exception, e:
        utils.handle(e, exiting=True)
    else:
        utils.get_sublogger().complete()
        exit("Exit Successfully.")
