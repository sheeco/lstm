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


def timestamp():
    stamp = time.strftime("%Y-%m-%d %H-%M-%S")
    return stamp


def test():
    # config.SHOW_WARNING = True
    warn("test warning")
    assert_type("test", str)
    # assert_type("test", tuple)
    stamp = timestamp()
    yes = confirm("Confirm")

