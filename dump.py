# coding:utf-8

import utils
from lstm.sampler import Sampler
import numpy


class DumpPrediction:
    def __init__(self, args):

        self.rootpath, self.frompath, self.epoch, self.dumpname, self.topath = \
            DumpPrediction.process_command_line_args(args)

        DumpPrediction.dump_predictions(self.rootpath, self.frompath, self.epoch, self.dumpname, self.topath)

    @staticmethod
    def read_predictions_from_compare_file(path):
        try:
            if not utils.assertor.assert_exists(path, raising=False):
                return None

            lines = utils.filer.read_lines(path)
            predictions = {}

            # Read a triple per line & save to dict
            for i in range(0, len(lines)):
                sec, x_pred, y_pred, x_fact, y_fact = [numpy.float32(x) for x in lines[i].split()][0:5]
                predictions[sec] = numpy.array([x_pred, y_pred])

            return predictions
        except:
            raise

    @staticmethod
    def save_triples_to_file(triples, path):
        try:
            sorted_triples = utils.sorted_items(triples)
            lines = ['%s\t%s\t%s\n' % (sec, xy[0], xy[1]) for sec, xy in sorted_triples]
            utils.filer.write_lines(path, lines)
        except:
            raise

    @staticmethod
    def find_subpath_by_identifier(path, identifier):
        try:
            matches = utils.filer.list_directory(path, pattern='(.*)(' + identifier + ')(.*)')

            if len(matches) == 0:
                raise ValueError("Cannot find any folder with given identifier '%s'." % identifier)
            elif len(matches) > 1:
                raise ValueError("Multiple folders with given identifier '%s' are found: %s." % (identifier, matches))
            else:
                return utils.filer.format_subpath(path, subpath=matches[0], isfile=False)
        except:
            raise

    @staticmethod
    def extract_timestamp(string):
        try:
            pattern = utils.regex_compile('(.*)(\d{4}?-\d{2}?-\d{2}?-\d{2}?-\d{2}?-\d{2}?)(.*)')
            identifier = utils.regex_match(pattern, string)
            if identifier:
                return identifier[1]
            else:
                return None
        except:
            raise

    @staticmethod
    def find_logpath(rootpath, frompath):
        try:
            identifier = DumpPrediction.extract_timestamp(frompath)
            if identifier is None:
                raise ValueError("Cannot find identifier string in given frompath '%s'." % frompath)
            else:
                return DumpPrediction.find_subpath_by_identifier(rootpath, identifier)

        except:
            raise

    @staticmethod
    def dump_predictions(rootpath, frompath, epoch, dumpname, topath=None):
        """
        Find predictions of given epoch in training & testing logs, and dump them to given topath.
        :param rootpath:
        :param frompath: Log folder / the identifier of an execution
        :param epoch: <int> epoch entry
        :param dumpname: Filename for dumped files e.g. node id
        :param topath: Destination path for dumping
        """
        try:
            frompath = DumpPrediction.find_logpath(rootpath, frompath)
            frompath = utils.filer.format_subpath(frompath, subpath=utils.get_config('path_compare'), isfile=False)
            # utils.filer.create_path(topath)

            logname_train = 'train-epoch%d.log' % epoch
            logname_test = 'test-epoch%d.log' % epoch

            dumpname_train = '%s.train.trace' % dumpname
            dumpname_test = '%s.test.trace' % dumpname
            dumpname_full = '%s.full.trace' % dumpname

            dumpname_train = utils.filer.format_subpath(topath, dumpname_train)
            dumpname_test = utils.filer.format_subpath(topath, dumpname_test)
            dumpname_full = utils.filer.format_subpath(topath, dumpname_full)

            path_train = utils.filer.format_subpath(frompath, subpath=logname_train, isfile=True)
            pred_train = DumpPrediction.read_predictions_from_compare_file(path_train)
            if pred_train is not None:
                DumpPrediction.save_triples_to_file(pred_train, dumpname_train)
                utils.xprint("Training predictions in '%s' are dumped to '%s'." % (path_train, dumpname_train),
                             newline=True)
            else:
                utils.warn("Cannot find file '%s' (for training epoch %d)." % (path_train, epoch))

            path_test = utils.filer.format_subpath(frompath, subpath=logname_test, isfile=True)
            pred_test = DumpPrediction.read_predictions_from_compare_file(path_test)
            if pred_test is not None:
                DumpPrediction.save_triples_to_file(pred_test, dumpname_test)
                utils.xprint("Testing predictions in '%s' are dumped to '%s'." % (path_test, dumpname_test),
                             newline=True)

                # Both are available
                if pred_train is not None:
                    utils.filer.write(dumpname_full, utils.filer.read(dumpname_train))
                    utils.filer.write(dumpname_full, utils.filer.read(dumpname_test))
                    utils.xprint(
                        "Full predictions in '%s' & '%s' are dumped to '%s'." % (
                            path_train, path_test, dumpname_full),
                        newline=True)

            # Both are unavailable
            elif pred_train is None:
                raise IOError("Cannot find file '%s' & '%s'." % (path_train, path_test))

            else:
                utils.warn("Cannot find file '%s' (for testing epoch %d)." % (path_test, epoch))

        except:
            raise

    @staticmethod
    def process_command_line_args(args):
        """
        e.g. dump.py [-p | --path <path-log-folders>] [-f | --from <folder-name-log|identifier>] [-e | --epoch <iepoch>]
        [-n | --name <filename-dump>] [-t | --to <folder-name-dump>]
        :return:
        """
        try:
            # short-opts: "ha:i" means opt '-h' & '-i' don't take arg, '-a' does take arg
            # long-opts: ["help", "add="] means opt '--add' does take arg
            pairs, unknowns = utils.get_opt(args, "p:f:e:n:t:c:",
                                            longopts=["path=", "from=", "epoch=", "name=", "to=", "config="])

            arg_root, arg_from, arg_epoch, arg_name, arg_to = None, None, None, None, None
            mandatory_args = [('-p', '--path'),
                              ('-f', '--from'),
                              ('-e', '--epoch'),
                              ('-n', '--name'),
                              ('-t', '--to')]
            optional_args = [('-c', '--config')]

            opts = [each_pair[0] for each_pair in pairs]
            for some_arg in mandatory_args:
                # if some_opt[2] is None:
                if some_arg[0] not in opts and some_arg[1] not in opts:
                    raise ValueError("Argument '%s|%s' is mandatory." % some_arg)

            for opt, val in pairs:
                if opt in ('-p', '--path'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    val = str(val)
                    if utils.assertor.assert_nonempty_str(val):
                        arg_root = val

                elif opt in ('-f', '--from'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    val = str(val)
                    if utils.assertor.assert_nonempty_str(val):
                        arg_from = val

                elif opt in ('-e', '--epoch'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    if utils.assertor.assert_type(val, int):
                        arg_epoch = val

                elif opt in ('-n', '--name'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    val = str(val)
                    if utils.assertor.assert_nonempty_str(val):
                        arg_name = val

                elif opt in ('-t', '--to'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    val = str(val)
                    if utils.assertor.assert_nonempty_str(val):
                        arg_to = utils.filer.validate_path_format(val)

                elif opt in ('-c', '--config'):
                    utils.assertor.assert_type(val, dict)
                    for key, value in val.items():
                        utils.update_config(key, value, 'command-line', silence=False)

                else:
                    raise ValueError("Unknown option '%s'." % opt)

            # if len(unknowns) > 0:
            if unknowns:
                raise ValueError("Unknown option(s) %s." % unknowns)

            return arg_root, arg_from, arg_epoch, arg_name, arg_to

        except:
            raise

    @staticmethod
    def test():
        utils.xprint("Test arg processing ...", newline=True)
        try:
            DumpPrediction.process_command_line_args(
                ["-f", "'[case]2018-01-18-20-53-10'", "-e", "1", "-n", "'demo'", "-t", "'res/dump/demo'"])
            DumpPrediction.process_command_line_args(
                ["-f", "'[case]2018-01-18-20-53-10'", "-e", "1", "-n", "'demo'"])
        except:
            raise
        try:
            DumpPrediction.process_command_line_args(["-e", "1", "-n", "'demo'", "-t", "'res/dump'"])
        except ValueError, e:
            utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)
        try:
            DumpPrediction.process_command_line_args(
                ["-f", "'[case]2018-01-18-20-53-10'", "-e", "invalid-epoch", "-n", "'demo'"])
        except ValueError, e:
            utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)

        utils.xprint("Fine.", newline=True)

        utils.xprint("Test dumping ...", newline=True)
        try:
            DumpPrediction.dump_predictions('log/', 'log/[case]2018-01-18-20-53-10', 0, 'demo0',
                                            'res/dump/demo')
            DumpPrediction.dump_predictions('log/', '[case]2018-01-18-20-53-10', 1, 'demo1', 'res/dump/demo')
            DumpPrediction.dump_predictions('log/', '2018-01-18-20-53-10', 2, 'demo2')
        except:
            raise

        try:
            DumpPrediction.dump_predictions('log/', '[mock]invalid-path', 1, 'demo', 'res/dump/demo')
        except IOError, e:
            utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)
        try:
            DumpPrediction.dump_predictions('log/', 'log/[case]2018-01-18-20-53-10', 3, 'demo', 'res/dump/demo')
        except IOError, e:
            utils.xprint("""Exception correctly caught: "%s"...""" % e.message, newline=True)

        utils.xprint('Fine.', newline=True)


# todo add dump_parameters()


class DumpPan:
    def __init__(self, args):

        self.frompath, self.node = DumpPan.process_command_line_args(args)
        DumpPan.dump_pan(self.frompath, self.node)

    @staticmethod
    def dump_pan(frompath, nodename):
        try:
            filenames = []
            if nodename is not None:
                this = '%s.trace' % nodename
                filenames += [this]
            else:
                filenames = utils.filer.list_directory(frompath, '.*\.trace')

            if len(filenames) == 0:
                raise ValueError("No trace file is found under '%s' with node name '%s'." % (frompath, nodename))

            for filename in filenames:
                nodename, _ = utils.filer.split_extension(filename)

                sampler = Sampler(path=frompath, nodes=[nodename], keep_positive=False)
                pan = sampler.pan_to_positive()

                panfile = '%s.pan' % nodename
                panfile = utils.filer.format_subpath(frompath, panfile, isfile=True)
                utils.filer.write(panfile, '%s\t%s' % (pan[0], pan[1]), mode='w')

                utils.xprint("Pan information (%s) for node %s is dumped to '%s'." % (pan, nodename, panfile),
                             newline=True)

        except:
            raise

    @staticmethod
    def process_command_line_args(args):
        """
        e.g. dump.py [-f | --from <log-folder|identifier>] [-n | --name <dump-filename>]
        :param args:
        """
        try:
            # short-opts: "ha:i" means opt '-h' & '-i' don't take arg, '-a' does take arg
            # long-opts: ["help", "add="] means opt '--add' does take arg
            pairs, unknowns = utils.get_opt(args, "f:n:c:", longopts=["from=", "node=", "config="])

            arg_root, arg_from, arg_epoch, arg_node, arg_to = None, None, None, None, None
            mandatory_args = [('-f', '--from')]
            optional_args = [('-n', '--node')]

            opts = [each_pair[0] for each_pair in pairs]
            for some_arg in mandatory_args:
                # if some_opt[2] is None:
                if some_arg[0] not in opts and some_arg[1] not in opts:
                    raise ValueError("Argument '%s|%s' is mandatory." % some_arg)

            for opt, val in pairs:
                if opt in ('-f', '--from'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    val = str(val)
                    if utils.assertor.assert_nonempty_str(val):
                        arg_from = val

                elif opt in ('-n', '--node'):
                    try:
                        val = utils.literal_eval(val)
                    except ValueError, e:
                        pass
                    except SyntaxError, e:
                        pass

                    val = str(val)
                    if utils.assertor.assert_nonempty_str(val):
                        arg_node = val

                elif opt in ('-c', '--config'):
                    utils.assertor.assert_type(val, dict)
                    for key, value in val.items():
                        utils.update_config(key, value, 'command-line', silence=False)

                else:
                    raise ValueError("Unknown option '%s'." % opt)

            # if len(unknowns) > 0:
            if unknowns:
                raise ValueError("Unknown option(s) %s." % unknowns)

            return arg_from, arg_node

        except:
            raise


def init_logger():
    try:
        timestamp = utils.get_timestamp()
        utils.sub_logger = utils.Logger(identifier=timestamp)
        utils.get_sublogger().register_console()
    except:
        raise


def test():
    init_logger()

    DumpPrediction.test()
    pass


if __name__ == '__main__':
    try:
        init_logger()

        args = utils.get_sys_args()[1:]
        utils.get_sublogger().log_args(args)
        if len(args) < 1:
            raise ValueError("No command is provided.")

        elif args[0] == 'prediction':
            DumpPrediction(args[1:])

        elif args[0] == 'pan':
            DumpPan(args[1:])

        else:
            raise ValueError("Unknown command: '%s'" % args[0])

    except Exception, e:
        utils.handle(e, exiting=True)
    else:
        utils.get_sublogger().complete()
        exit("Exit Successfully.")
