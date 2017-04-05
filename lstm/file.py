# coding:utf-8

import os
from utils import *


__all__ = [
    "if_exists",
    "assert_exists",
    "is_file",
    "create_path",
    "rename_path",
    "list_directory",
    "split_filename",
    "read_lines",
    "test"
]


def if_exists(path):
    return os.path.exists(path)


def assert_exists(path, assertion=True, raising=True):
    if not if_exists(path) is assertion:
        if raising:
            if assertion is True:
                raise IOError("assert_exists @ file: '%s' does not exists." % path)
            else:
                raise IOError("assert_exists @ file: '%s' already exists." % path)
        else:
            if assertion is True:
                warn("assert_exists @ file: '%s' does not exists." % path)
            else:
                warn("assert_exists @ file: '%s' already exists." % path)


def is_file(path):
    return os.path.isfile(path)


def create_path(path):
    assert_exists(path, assertion=False, raising=False)
    try:
        os.makedirs(path)
        return True
    except Exception, e:
        raise


def rename_path(old, new):
    assert_exists(old)
    assert_exists(new, assertion=False)
    try:
        os.name(old, new)
        return True
    except Exception, e:
        raise


def list_directory(path):
    assert_exists(path)
    try:
        return os.listdir(path)
    except Exception, e:
        raise


def split_filename(filename):
    try:
        body, extension = os.path.splitext(filename)
        return body, extension
    except Exception, e:
        raise


def read_lines(path):
    assert_exists(path)
    try:
        return open(path, 'rb').readlines()
    except Exception, e:
        raise


def test():
    pass
