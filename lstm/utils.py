# coding:utf-8

import time
import traceback
import warnings
import numpy

from config import configuration as config


__all__ = [
    "match",
    "xprint",
    "warn",
    "handle",
    "assert_type",
    "assert_finite",
    "assert_unreachable",
    "confirm",
    "ask_int",
    "mean_min_max",
    "format_var",
    "get_timestamp",
    "format_time_string",
    "Timer",
    "test"
]


def match(shape1, shape2):
    return (len(shape1) == len(shape2) and
            all(s1 is None or s2 is None or s1 == s2
                for s1, s2 in zip(shape1, shape2)))


def xprint(what, level=0, newline=False, logger=None):
    # no level limit for logger
    if logger is not None:
        logger.log("%s\n" % what if newline else "%s" % what)

    if level == 0 or level <= config['print_level']:
        print "%s\n" % what if newline else "%s" % what,


def warn(info):
    # if config['show_warning']:
        # xprint("[Warning] %s" % info)
    if not config['show_warning']:
        warnings.filterwarnings("ignore")
    warnings.warn(info)


def handle(exception, logger=None):
    xprint('\n\n')
    xprint(exception.message, newline=True)
    xprint(traceback.format_exc(), newline=True)
    if logger is not None:
        logger.register("exception")
        logger.log(exception.message + '\n', name="exception")
        logger.log(traceback.format_exc() + '\n', name="exception")


def assert_type(var, assertion, raising=True):
    if isinstance(assertion, list) \
            or isinstance(assertion, tuple):
        fine = any(assert_type(var, iassertion, raising=False) for iassertion in assertion)
    else:
        fine = isinstance(var, assertion)
    if raising and not fine:
        raise ValueError("assert_type @ utils: Expect %s while getting %s instead." % (assertion, type(var)))
    return fine


def assert_finite(var, name):
    if not isinstance(var, list):
        var = [var]
    if any((not numpy.isfinite(ivar).all()) for ivar in var):
        raise AssertionError("assert_finite @ utils: <%s> contains 'nan' or 'inf'." % name)
    else:
        return True


def assert_unreachable():
    raise RuntimeError("assert_unreachable @ utils: \n\tUnexpected access of this block.")


def confirm(info):
    try:
        ans = raw_input("%s (y/n): " % info)
        if ans == 'y' or ans == 'Y':
            return True
        elif ans == 'n' or ans == 'N':
            return False
    except:
        pass
    return confirm("Pardon?")


def ask_int(info, code_quit='q'):
    try:
        answer = raw_input("%s (positive integer): " % info)
        if answer == code_quit:
            return None
        n = int(answer)
        if n >= 0:
            return n
    except:
        pass
    return ask_int("Pardon?", code_quit=code_quit)


def mean_min_max(mat):
    return [numpy.mean(mat), numpy.min(mat), numpy.max(mat)]


def format_var(var, name=None, detail=False):
    string = ''
    if isinstance(var, list):
        assert isinstance(name, list)
        assert len(var) == len(name)
        for i in xrange(len(var)):
            string += format_var(var[i], name=name[i], detail=detail) + '\n'
    else:
        if name is not None:
            string += "%s: " % name
        if isinstance(var, numpy.ndarray):
            if detail:
                string += '\n%s' % var if name is not None else '%s' % var
            elif var.size == 1:
                string += '%.1f' % var[0]
            elif numpy.isfinite(var).all():
                string += '(mean: %.1f, min: %.1f, max: %.1f)' % tuple(mean_min_max(var))
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


class Timer:
    def __init__(self, formatted=True):
        self.sec_start = None
        self.sec_stop = None
        self.sec_elapse = None
        self.formatted = formatted
        self.start()

    def start(self):
        """
        Would restart & override.
        """
        try:
            self.sec_start = time.clock()
            self.sec_stop = None
            self.sec_elapse = None
        except:
            raise

    def stop(self):
        try:
            if self.sec_start is None:
                return None

            if self.sec_stop is None:
                self.sec_stop = time.clock()
                self.sec_elapse = self.sec_stop - self.sec_start
            return self.get_elapse()
        except:
            raise

    def get_elapse(self, formatted=None):
        if formatted is None:
            formatted = self.formatted
        return format_time_string(self.sec_elapse) if formatted else self.sec_elapse


def test():

    def test_warn():
        config['show_warning'] = True
        warn("test warning")

    def test_assert():
        assert_type("test", str)

    def test_timestamp():
        timestamp = get_timestamp()

    def test_xprint():
        xprint(format_time_string(59.9), level=1, newline=True)
        xprint(format_time_string(222.2), level=1, newline=True)
        xprint(format_time_string(7777.7), level=1, newline=True)

    def test_timer():
        timer = Timer()
        xprint(timer.stop(), level=1, newline=True)
        timer.start()
        xprint(timer.stop(), level=1, newline=True)
        xprint(timer.stop(), level=1, newline=True)
        timer = Timer(formatted=False)
        xprint(timer.stop(), level=1, newline=True)

    test_warn()
