# coding:utf-8

import time
import traceback
import warnings
import numpy

import config


def match(shape1, shape2):
    return (len(shape1) == len(shape2) and
            all(s1 is None or s2 is None or s1 == s2
                for s1, s2 in zip(shape1, shape2)))


def check_range(mat):
    return [numpy.min(mat), numpy.max(mat), numpy.mean(mat)]


def warn(info):
    # if config.SHOW_WARNING:
        # print "[Warning] %s" % info
    if not config.SHOW_WARNING:
        warnings.filterwarnings("ignore")
    warnings.warn(info)


def handle(exception):
    print ''
    print traceback.format_exc()


def assert_type(var, assertion):
    if not isinstance(var, assertion):
        raise ValueError("assert_type @ utils: Expect %s while getting %s instead." % (assertion, type(var)))
    else:
        return True


def assert_unreachable():
    raise RuntimeError("assert_unreachable @ utils: \n\tUnexpected access of this block.")


def confirm(info):
        ans = raw_input("%s (y/n): " % info)
        if ans == 'y' or ans == 'Y':
            return True
        elif ans == 'n' or ans == 'N':
            return False
        else:
            return confirm("Pardon?")


def get_timestamp():
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
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
    # config.SHOW_WARNING = True
    # warn("test warning")
    # assert_type("test", str)
    # timestamp = get_timestamp()
    # yes = confirm("Confirm")

    print format_time_string(59.9)
    print format_time_string(222.2)
    print format_time_string(7777.7)
    timer = Timer()
    print timer.stop()
    timer.start()
    print timer.stop()
    print timer.stop()
    timer = Timer(formatted=False)
    print timer.stop()
