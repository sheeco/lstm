# coding:utf-8

import time
import traceback
import sys
import getopt
import ast
import os
import win32file
import win32con
import shutil
import numpy
import copy
import cPickle

import config

__all__ = [
    "Timer",
    "Logger",
    "match",
    "xprint",
    "warn",
    "handle",
    "assert_not_none",
    "assert_type",
    "assert_finite",
    "assert_unreachable",
    "ask",
    "interpret_confirm",
    "interpret_positive_int",
    "interpret_positive_float",
    "interpret_file_path",
    "peek_matrix",
    "format_var",
    "get_timestamp",
    "format_time_string",
    "if_exists",
    "assert_exists",
    "is_file",
    "is_directory",
    "is_hidden",
    "hide_path",
    "unhide_path",
    "list_directory",
    "split_path",
    "split_extension",
    "validate_path_format",
    "format_subpath",
    "create_path",
    "rename_path",
    "copy_file",
    "read",
    "read_lines",
    "dump_to_file",
    "load_from_file",
    "get_rootlogger",
    "get_sublogger",
    "has_config",
    "get_config",
    "remove_config",
    "update_config",
    "import_config",
    "process_command_line_args",
    "test"
]

# Global loggers

root_logger = None
sub_logger = None


def __init__():
    global root_logger
    global sub_logger

    root_logger = Logger()

    timestamp = get_timestamp()
    temp_identifier = '[%s]%s' % (config.get_config('tag'), timestamp) if config.has_config('tag') else timestamp
    sub_logger = Logger(identifier=temp_identifier)
    sub_logger.register_console()


class Timer:
    def __init__(self, formatted=True, start=True):
        self.sec_start = None
        self.sec_stop = None
        self.sec_elapse = None
        self.timer_pause = None
        self.duration_pause = 0
        self.formatted = formatted

        if start:
            self.start()

    def is_on(self):
        """
        Considered to be on if timer has not been started, or has been stopped.
        :return:
        """
        return self.sec_start is not None and self.sec_stop is None

    def is_off(self):
        return not self.is_on()

    def is_pausing(self):
        return self.timer_pause is not None \
               and self.timer_pause.is_on()

    def start(self):
        """
        Would restart & override.
        """
        try:
            self.sec_start = time.clock()
            self.sec_stop = None
            self.sec_elapse = None
            self.duration_pause = 0

            # Will stop pausing if it's still pausing
            if self.timer_pause is not None \
                    and self.is_on():
                self.timer_pause.stop()

        except:
            raise

    def stop(self):
        try:
            if self.is_off():
                raise RuntimeError("The timer must be started first.")

            # Will stop pausing if it's still pausing
            if self.timer_pause is not None \
                    and self.timer_pause.is_on():
                self.duration_pause += self.timer_pause.stop()

            if self.sec_stop is None:
                self.sec_stop = time.clock()
            return self.get_elapse()
        except:
            raise

    def pause(self):
        try:
            if self.timer_pause is None:
                self.timer_pause = Timer(formatted=False, start=False)
            elif self.timer_pause.is_on():
                raise RuntimeError("The timer is already pausing.")

            self.timer_pause.start()
        except:
            raise

    def resume(self):
        try:
            if self.timer_pause is None or self.timer_pause.is_off():
                raise RuntimeError("The timer is not pausing.")

            this_pause = self.timer_pause.stop()
            self.duration_pause += this_pause
            return this_pause
        except:
            raise

    def get_elapse(self, formatted=None):
        if self.sec_start is None \
                or self.sec_stop is None:
            return None

        self.sec_elapse = self.sec_stop - self.sec_start - self.duration_pause
        if formatted is None:
            formatted = self.formatted
        return format_time_string(self.sec_elapse) if formatted else self.sec_elapse

    @staticmethod
    def test():
        try:
            print "Testing timer ...",
            timer = Timer(start=False)
            elapse = timer.get_elapse()
            timer.start()
            elapse = timer.stop()
            try:
                # duplicate stop
                elapse = timer.stop()
            except Exception, e:
                pass
            timer.start()
            # duplicate start, meaning restart
            timer.start()
            timer.pause()
            try:
                # duplicate pause
                timer.pause()
            except Exception, e:
                pass
            pause = timer.resume()
            try:
                # duplicate resume
                timer.resume()
            except Exception, e:
                pass
            timer.pause()
            pause = timer.resume()
            elapse = timer.stop()
            timer.get_elapse()
            print "Fine"
        except:
            raise


class Logger:
    def __init__(self, path=None, identifier=None, tag=None, bound=False):
        """

        :param path:
        :param identifier:
        :param tag:
        :param bound: <bool> Whether to bound to config.
        """
        self.root_path = path if path is not None else config.get_config('path_log')
        if not if_exists(self.root_path):
            create_path(self.root_path)
        if not (self.root_path[-1] == '/'
                or self.root_path[-1] == '\\'):
            self.root_path += '/'
        self.root_path = self.root_path.replace('\\', '/')

        self.identifier = None
        self.tag = None
        self.log_path = None
        self._update_log_path_(identifier, tag)

        self.bound = bound

        # {'name':
        #   content=[(tag1, [...]),
        #            (tag2, [...])
        #           ]
        # }
        self.logs = {}

        self.filename_console = 'console'

    @staticmethod
    def _format_log_path_(root_path, identifier, tag):
        subfolder = '[%s]%s' % (tag, identifier) if tag is not None \
            else '%s' % identifier
        return format_subpath(root_path, subpath=subfolder, isfile=False)

    def _update_log_path_(self, identifier, tag):
        try:
            if self.log_path is None \
                    or identifier != self.identifier \
                    or tag != self.tag:
                new_log_path = Logger._format_log_path_(self.root_path, identifier if identifier is not None else '', tag)

                # Initialize log path
                if self.log_path is None:
                    create_path(new_log_path)
                    if identifier is not None:
                        hide_path(new_log_path)

                # Update log path
                else:
                    rename_path(self.log_path, new_log_path)
                self.identifier = identifier
                self.tag = tag
                self.log_path = new_log_path

        except:
            raise

    def _validate_log_path_(self):
        """
        Update log path if config 'identifier' or 'tag' has been changes.
        Should be called at the beginning of any public method in this class.
        :return:
        """
        try:
            # Ignore unbound loggers
            if not self.bound:
                return

            if (config.has_config('identifier')
                and config.get_config('identifier') != self.identifier) \
                    or (config.has_config('tag')
                        and config.get_config('tag') != self.tag):
                self._update_log_path_(config.get_config('identifier'), config.get_config('tag'))

        except:
            raise

    def log_config(self):
        try:
            self._validate_log_path_()

            self.register(name='config')
            string = "{\n"
            configs = config.get_config()
            for key, content in configs.iteritems():
                string += "\t'%s': {" % key
                string += "'value': '%s'" % (content['value'],) if isinstance(content['value'], str) \
                    else "'value': %s" % (content['value'],)
                string += ", 'source': '%s'" % content['source']
                if 'tags' in content:
                    string += ", 'tags': %s" % content['tags']

                for inkey, invalue in content.iteritems():
                    if inkey in ('value', 'source'):
                        continue
                    else:
                        string += ", '%s': " % inkey
                        string += "'%s'" % invalue if isinstance(invalue, str) else "%s" % invalue
                string += "},\n"
            string += "}"
            self.log(string, name='config')

        except:
            raise

    def log_file(self, path, rename=None):
        try:
            self._validate_log_path_()

            assert_exists(path)
            directory, filename = split_path(path)
            topath = self.log_path + filename if rename is None else self.log_path + rename
            assert_exists(topath, assertion=False)

            copy_file(path, topath)

        except:
            raise

    def has_log(self, name):
        return name in self.logs

    def register(self, name, columns=None):
        try:
            self._validate_log_path_()

            if not self.has_log(name):
                content = []
                if columns is None:
                    columns = ['']
                for tag in columns:
                    content += [(tag, [])]
                self.logs[name] = content

                filepath = '%s%s.log' % (self.log_path, name)
                pfile = open(filepath, 'a')
                hastag = False
                for tag in columns:
                    if tag != '':
                        hastag = True
                        pfile.write('%s\t' % tag)
                if hastag:
                    pfile.write('\n')
                pfile.flush()
                pfile.close()

            else:
                pass

        except:
            raise

    def register_console(self, filename=None):
        try:
            self._validate_log_path_()

            self.filename_console = filename if filename is not None else self.filename_console
            self.register(self.filename_console)
        except:
            raise

    def log(self, content, name=None):
        try:
            self._validate_log_path_()

            if name is None:
                name = self.filename_console
            if name not in self.logs:
                raise ValueError("Cannot find '%s' in log registry. "
                                 "Must `register` first." % name)
            else:
                registry = self.logs[name]
            path = '%s%s.log' % (self.log_path, name)
            pfile = open(path, 'a')

            if isinstance(content, dict):
                dict_content = copy.deepcopy(content)
                for column in registry:
                    tag = column[0]
                    rows = column[1]
                    if tag in dict_content:
                        row = '%s' % dict_content[tag]
                        row = row.replace('\n', ' ')
                        row = row.replace('\t', ' ')
                        rows += [row]
                        dict_content.pop(tag)
                    else:
                        rows += ['-']
                if len(dict_content) > 0:
                    raise ValueError("Cannot find tag %s in log registry. " % dict_content.keys())

                for column in registry:
                    pfile.write('%s\t' % column[1][-1])
                pfile.write('\n')

            elif isinstance(content, str):
                column0 = registry[0]
                tag0 = column0[0]
                rows0 = column0[1]

                if len(registry) > 1 \
                        or tag0 != '':
                    raise ValueError("A tag among %s is requested. " % [column[0] for column in registry])

                rows0 += [content]
                pfile.write(content)

            else:
                assert_unreachable()

            pfile.flush()
            pfile.close()

        except:
            raise

    def log_console(self, content):
        try:
            if not self.has_log(self.filename_console):
                self.register_console()

            self.log(content, name=self.filename_console)

        except:
            raise

    def complete(self):
        try:
            self._validate_log_path_()

            if self.identifier is None:
                return
            directory, filename = split_path(self.log_path)
            if filename[0] == '.':
                filename = filename[1:]
                complete_path = directory + filename
                rename_path(self.log_path, complete_path)
                self.log_path = complete_path
            unhide_path(self.log_path)

        except:
            raise

    @staticmethod
    def test():
        try:
            logger = Logger(identifier=get_timestamp())
            logger.log_config()

            tags = ['time', 'x', 'y']
            arr = numpy.zeros((2, 3, 2))
            _dict = {'time': [1, 2, 3], 'x': ['x1', 'x2', 'x3'], 'y': [arr, arr, arr]}

            logger.register('test', tags)
            logger.log(_dict, name='test')

            update_config('tag', 'test-tag', source='test')

            _dict.pop('y')
            logger.log(_dict, name='test')

            try:
                logger.log('test wrong content type ...', name='test')
            except Exception, e:
                pass

            try:
                _dict['z'] = 'test-wrong-key'
                logger.log(_dict, name='test')
            except Exception, e:
                pass

            try:
                logger.log(_dict, name='test-unregistered')
            except Exception, e:
                pass

            logger.register_console()
            logger.log('test console output ...')
            logger.complete()

        except:
            raise


def match(shape1, shape2):
    return (len(shape1) == len(shape2)
            and all(s1 is None
                    or s2 is None
                    or s1 == s2
                    for s1, s2 in zip(shape1, shape2)))


def xprint(what, newline=False, logger=None):
    # no level limit for logger
    if logger is None:
        logger = get_sublogger()
    if logger is not None:
        logger.log_console("%s\n" % what if newline else "%s" % what)

    print "%s\n" % what if newline else "%s" % what,


def warn(info):
    try:
        # flush before writing
        p_out = sys.stdout
        if p_out is not None:
            p_out.flush()
        p_in = sys.stdin
        if p_in is not None:
            p_in.flush()

        p_err = sys.stderr
        if p_err is None:
            # sys.stderr is None - warnings get lost
            return
        p_err.write("[Warning] %s\n" % info)
        p_err.flush()

    except (IOError, UnicodeError):
        raise


def handle(exception, logger=None):
    xprint('\n')
    xprint(traceback.format_exc(), newline=True, logger=logger)
    xprint('%s\n' % exception.message, newline=True, logger=logger)
    if logger is None:
        logger = get_sublogger()
    if logger is not None:
        logger.register("exception")
        logger.log('%s\n' % traceback.format_exc(), name="exception")
        logger.log('%s\n\n' % exception.message, name="exception")

    exit()


def assert_not_none(var, message, raising=True):
    """

    :param var:
    :param message: Message to form exception if assertion is not True.
    :param raising:
    :return:
    """
    fine = True if var is not None else False
    if raising \
            and not fine:
        raise AssertionError(message)
    return fine


def assert_type(var, assertion, raising=True):
    if isinstance(assertion, list) \
            or isinstance(assertion, tuple):
        fine = any(assert_type(var, iassertion, raising=False) for iassertion in assertion)
    else:
        fine = isinstance(var, assertion)
    if raising \
            and not fine:
        raise ValueError("Expect %s while getting %s instead." % (assertion, type(var)))
    return fine


def assert_finite(var, name):
    if not isinstance(var, list):
        var = [var]
    if any((not numpy.isfinite(ivar).all()) for ivar in var):
        raise AssertionError("`%s` contains 'nan' or 'inf'." % name)
    else:
        return True


def assert_unreachable():
    raise RuntimeError("Unexpected access of this block.")


def ask(message, code_quit='q', interpretor=None):
    if not callable(interpretor):
        raise ValueError("Argument `interpret` is not callable.")

    try:
        question = "%s\n$ " % message
        answer = raw_input(question)

        # log question & answer
        get_sublogger().log_console("%s%s\n" % (question, answer))

        if code_quit is not None \
                and answer == code_quit:
            return None

        try:
            if interpretor is None:
                return answer
            else:
                answer = interpretor(answer)
                return answer

        except AssertionError, e:
            return ask(e.message, code_quit=code_quit, interpretor=interpretor)
    except:
        pass
    return ask("Pardon?", code_quit=code_quit, interpretor=interpretor)


def _interpret_confirm_(answer):
    try:
        if answer in ('y', 'Y', 'yes'):
            return True
        elif answer in ('n', 'N', 'no'):
            return False
        else:
            raise AssertionError("Invalid Syntax. Only take 'y' / 'n' for an answer.")
    except:
        raise


interpret_confirm = _interpret_confirm_


def _interpret_positive_int_(answer):
    try:
        n = int(answer)
        if n >= 0:
            return n
        else:
            raise AssertionError("Only take positive integer for an answer.")
    except ValueError, e:
        raise AssertionError(e.message)


interpret_positive_int = _interpret_positive_int_


def _interpret_positive_float_(answer):
    try:
        f = float(answer)
        if f >= 0:
            return f
        else:
            raise AssertionError("Only take positive float for an answer.")
    except ValueError, e:
        raise AssertionError(e.message)


interpret_positive_float = _interpret_positive_float_


def _interpret_file_path_(answer):
    try:
        path = validate_path_format(answer)
        if is_file(path):
            return path
        else:
            raise AssertionError("Cannot find file '%s'." % path)
    except:
        raise


interpret_file_path = _interpret_file_path_


def peek_matrix(mat):
    return [numpy.mean(mat), numpy.min(mat), numpy.max(mat)]


def format_var(var, name=None, detail=False):
    string = ''
    if isinstance(var, list):
        assert isinstance(name, list)
        assert len(var) == len(name)
        for i in xrange(len(var)):
            string += '%s\n' % format_var(var[i], name=name[i], detail=detail)
    else:
        if name is not None:
            string += "%s: " % name
        if isinstance(var, numpy.ndarray):
            if detail:
                string += '\n%s' % var if name is not None else '%s' % var
            elif var.size == 1:
                string += '%.1f' % var[0]
            elif numpy.isfinite(var).all():
                string += '(mean: %.1f, min: %.1f, max: %.1f)' % tuple(peek_matrix(var))
            elif numpy.isnan(var).all():
                string += 'nan'
            elif numpy.isinf(var).all():
                string += 'inf'
            else:
                string += '\n%s' % var if name is not None else '%s' % var
        elif isinstance(var, float):
            string += '%.1f' % var
        else:
            string += '%s' % var
    return string


def get_timestamp():
    stamp = time.strftime("%Y-%m-%d-%H-%M-%S")
    return stamp


def format_time_string(seconds):
    if seconds < 60:
        return "%ds" % seconds
    elif seconds < 60 * 60:
        return "%dm %ds" % divmod(seconds, 60)
    else:
        h, sec = divmod(seconds, 60 * 60)
        min, sec = divmod(sec, 60)
        return "%dh %dm %ds" % (h, min, sec)


def if_exists(path):
    return os.path.exists(path)


def assert_exists(path, assertion=True, raising=True):
    if not if_exists(path) is assertion:
        if raising:
            if assertion is True:
                raise IOError("'%s' does not exists." % path)
            else:
                raise IOError("'%s' already exists." % path)
        # else:
            # if assertion is True:
            #     warn("assert_exists @ file: "
            #          "'%s' does not exists." % path)
            # else:
            #     warn("assert_exists @ file: "
            #          "'%s' already exists." % path)
        return False
    else:
        return True


def is_file(path):
    return os.path.isfile(path)


def is_directory(path):
    return os.path.isdir(path)


def is_hidden(path):
    try:
        file_flag = win32file.GetFileAttributesW(path)
        is_hiden = file_flag & win32con.FILE_ATTRIBUTE_HIDDEN
        return is_hiden
    except:
        raise


def hide_path(path):
    try:
        file_flag = win32file.GetFileAttributesW(path)
        win32file.SetFileAttributesW(path, file_flag | win32con.FILE_ATTRIBUTE_HIDDEN)
    except:
        raise


def unhide_path(path):
    try:
        if not is_hidden(path):
            return
        file_flag = win32file.GetFileAttributesW(path)
        win32file.SetFileAttributesW(path, file_flag & ~win32con.FILE_ATTRIBUTE_HIDDEN)
    except:
        raise


def list_directory(path):
    assert_exists(path)
    try:
        return os.listdir(path)
    except:
        raise


def split_path(path):
    try:
        if path[-1] == '/' \
                or path[-1] == '\\':
            tail = '/'
            path = path[:-1]
        else:
            tail = ''
        directory, filename = os.path.split(path)
        return '%s/' % directory, '%s%s' % (filename, tail)
    except:
        raise


def split_extension(filename):
    try:
        body, extension = os.path.splitext(filename)
        return body, extension
    except:
        raise


def validate_path_format(path):
    try:
        body, extension = split_extension(path)
        if extension == '' \
                and path[-1] not in ('/', '\\'):
            path += '/'
        path = path.replace('\\', '/')
        return path
    except:
        raise


def format_subpath(path, subpath='', isfile=True):
    try:
        ret = '%s/' % path if path[-1] != '/' else path
        ret += subpath
        ret = '%s/' % ret if ret[-1] != '/' \
                             and not isfile else ret
        return ret
    except:
        raise


def create_path(path):
    try:
        if not assert_exists(path, assertion=False, raising=False):
            return
        os.makedirs(path)
        return True
    except:
        raise


def rename_path(old, new):
    try:
        assert_exists(old)
        assert_exists(new, assertion=False)
        os.rename(old, new)
        return True
    except:
        raise


def copy_file(frompath, topath):
    try:
        shutil.copy(frompath, topath)
    except:
        raise


def read(path):
    assert_exists(path)
    try:
        pfile = open(path, 'r')
        all_ = pfile.read()
        pfile.close()
        return all_
    except:
        raise


def read_lines(path):
    assert_exists(path)
    try:
        pfile = open(path, 'r')
        lines = pfile.readlines()
        pfile.close()
        return lines
    except:
        raise


def dump_to_file(what, path):
    try:
        PROTOCOL_ASCII = 0
        PROTOCOL_BINARY = 2
        PROTOCOL_HIGHEST = -1

        pfile = open(path, 'wb')
        cPickle.dump(what, pfile, protocol=PROTOCOL_HIGHEST)
        pfile.close()
    except:
        raise


def load_from_file(path):
    try:
        assert_exists(path)
        pfile = open(path, 'rb')

        # To fix wrongly pickled files,
        # which were written thru mode 'w' instead of 'wb', or closed improperly.
        original = pfile.read()
        pfile.close()
        converted = original.replace('\r\n', '\n')
        pfile = open(path, 'wb')
        pfile.write(converted)
        pfile.close()
        pfile = open(path, 'rb')

        what = cPickle.load(pfile)
        pfile.close()

        return what

    except:
        raise


def get_rootlogger():
    global root_logger
    return root_logger


def get_sublogger():
    global sub_logger
    return sub_logger


def _validate_config_():
    try:
        # 'nodes' will override 'num_node'
        if config.has_config('nodes') \
                and config.get_config('nodes') is not None \
                and config.has_config('num_node') \
                and config.get_config('num_node') is not None:
            warn("Configuration 'nodes' (%s) will override 'num_node' (%s)."
                 % (config.get_config('nodes'), config.get_config('num_node')))
            remove_config('num_node')

        # Validate path formats
        for key, content in config._filter_config_('path').iteritems():
            original = content['value']
            validated = validate_path_format(original)
            if validated != original:
                config._update_config_(key, validated, source=content['source'])

    except:
        raise
    pass


"""
Wrap methods in config.py to apply validation
"""

has_config = config.has_config
get_config = config.get_config
remove_config = config.remove_config


def update_config(key, value, source, tags=None):
    try:
        config._update_config_(key, value, source, tags)
        _validate_config_()
    except:
        raise


def import_config(config_imported, tag=None):
    try:
        config._import_config_(config_imported, tag)
        _validate_config_()
    except:
        raise


def process_command_line_args(args=None):
    """
    e.g. test.py [-c | --config <dict-config>] [-t | --tag <tag-for-logging>] [-i | --import <import-path>]
    :return:
    """
    try:
        args = sys.argv[1:] if args is None else args

        # log command line args to file args.log
        if args != '' \
                and get_sublogger() is not None:
            get_sublogger().register('args')
            get_sublogger().log("%s\n" % args, name='args')

        # shortopts: "ha:i" means opt '-h' & '-i' don't take arg, '-a' does take arg
        # longopts: ["--help", "--add="] means opt '--add' does take arg
        opts, unknowns = getopt.getopt(args, "c:t:i:", longopts=["config=", "tag=", "import="])

        # handle importing first
        for opt, argv in opts:
            if opt in ("-i", "--import"):
                # Import config.log & params.pkl if exists
                if is_directory(argv):
                    path_import = validate_path_format(argv)
                    key = 'path_import'
                    update_config(key, path_import, 'command-line')

                    path_config = format_subpath(path_import, subpath='config.log')
                    config_imported = read(path_config)
                    try:
                        config_imported = ast.literal_eval(config_imported)
                    except:
                        raise
                    import_config(config_imported, tag='build')
                    xprint("Import configurations from '%s'." % path_config, newline=True)

                    opts.remove((opt, argv))

                # Import params.pkl
                elif is_file(argv):
                    update_config('path_unpickle', argv, 'command-line', tags=['path'])

                else:
                    raise ValueError("Invalid path '%s' to import." % argv)
            else:
                pass

        for opt, argv in opts:
            if argv != '':
                try:
                    argv = ast.literal_eval(argv)
                except:
                    pass

            # Manual configs will override imported configs
            if opt in ("-c", "--config"):
                if isinstance(argv, dict):
                    for key, value in argv.items():
                        if config.has_config(key):
                            xprint("Update configuration '%s' from %s to %s (from command line)." % (
                            key, config.get_config(key), value), newline=True)
                        else:
                            xprint("Add configuration '%s' to be %s." % (key, value), newline=True)
                        update_config(key, value, 'command-line')
                else:
                    raise ValueError("The configuration must be a dictionary.")

            elif opt in ("-t", "--tag"):
                key = 'tag'
                if config.has_config(key):
                    xprint("Update tag from '%s' to '%s' (from command line)." % (config.get_config(key), argv), newline=True)
                else:
                    xprint("Set tag to be '%s' (from command line)." % argv, newline=True)
                update_config(key, argv, 'command-line')

            else:
                raise ValueError("Unknown option '%s'." % opt)

        if len(unknowns) > 0:
            raise ValueError("Unknown option(s) %s." % unknowns)

    except:
        raise


def test():
    try:

        def test_warn():
            warn("test warning")

        def test_assert():
            assert_type("test", str)

        def test_timestamp():
            timestamp = get_timestamp()

        def test_xprint():
            xprint(format_time_string(59.9), newline=True)
            xprint(format_time_string(222.2), newline=True)
            xprint(format_time_string(7777.7), newline=True)

        def test_args():
            process_command_line_args()
            process_command_line_args(args=[])
            args = ["-c", "{'num_node': 9, 'tag': 'x'}", "-t", "xxx"]
            process_command_line_args(args=args)

        def test_exception():

            # import thread
            # import win32api
            #
            # # Load the DLL manually to ensure its handler gets
            # # set before our handler.
            # # basepath = imp.find_module('numpy')[1]
            # # ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
            # # ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))
            #
            # # Now set our handler for CTRL_C_EVENT. Other control event
            # # types will chain to the next handler.
            # def handler(dw_ctrl_type, hook_sigint=thread.interrupt_main):
            #     if dw_ctrl_type == 0:  # CTRL_C_EVENT
            #         hook_sigint()
            #         return 1  # don't chain to the next handler
            #     return 0  # chain to the next handler
            #
            # win32api.SetConsoleCtrlHandler(handler, 1)

            def _test():
                i = 0
                while True:
                    try:
                        print "%d ..." % i
                        i += 1
                        time.sleep(1)
                    except KeyboardInterrupt, e:
                        print "KeyboardInterrupt caught: %s" % e.message
                        return i
                    finally:
                        print "finally %d" % i

            print "return %d" % _test()

        def test_hiding():
            path = config.get_config('path_log')
            _hidden = is_hidden(path)
            hide_path(path)
            unhide_path(path)

        def test_formatting():
            path = "\log/test"
            path = validate_path_format(path)
            path = format_subpath(path, '', isfile=False)
            path = format_subpath(path, 'file.txt')
            path = format_subpath(path, 'subfolder', isfile=False)

            path = "\\log/test"
            path = validate_path_format(path)

        def test_ask():
            yes = ask('Enter yes or no:', interpretor=interpret_confirm)
            n = ask('Enter positive integer:', interpretor=interpret_positive_int)
            f = ask('Enter positive float:', interpretor=interpret_positive_float)
            path = ask('Enter path:', interpretor=interpret_file_path)
            try:
                ans = ask('Test wrong interpretor.', interpretor='test')
            except Exception, e:
                pass

        def test_pickling():
            temp_logger = Logger(identifier='pickle')
            temp_logger.register('pickle')
            temp_logger.log('content used for pickling test', name='pickle')
            filename = format_subpath(temp_logger.log_path, 'logger.pkl')
            dump_to_file(temp_logger, filename)
            logger_loaded = load_from_file(filename)

        # test_hiding()
        # test_formatting()
        # test_ask()
        # test_pickling()

        # test_warn()
        # test_args()
        # test_exception()
        # test_ask()
        Timer.test()

    except:
        raise
