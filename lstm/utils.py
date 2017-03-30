# coding:utf-8

import time
import traceback
import numpy

import config


def match(shape1, shape2):
    return (len(shape1) == len(shape2) and
            all(s1 is None or s2 is None or s1 == s2
                for s1, s2 in zip(shape1, shape2)))


def check_range(mat):
    return [numpy.min(mat), numpy.max(mat), numpy.mean(mat)]


def warn(info):
    if config.SHOW_WARNING:
        print "[Warning] " + info


def handle(exception):
        # print str(type(exception)) + exception.message
        print traceback.format_exc()


def assert_type(var, assertion):
    if not isinstance(var, assertion):
        raise ValueError("assert_type @ utils: Expect " + str(assertion) + " while getting " + str(type(var)) + " instead.")


def confirm(info):
        ans = raw_input(info + " (y/n): ")
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
    assert_type("test", str)
    # assert_type("test", tuple)
    stamp = timestamp()
    yes = confirm("Confirm")

