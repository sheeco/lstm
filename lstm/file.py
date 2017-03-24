# coding:utf-8

import os
from utils import *


def if_exists(path):
    return os.path.exists(path)


def assert_exists(path, assertion=True, raising=True):
    if not if_exists(path) is assertion:
        if raising:
            if assertion is True:
                raise IOError("assert_exists @ file: " + "'" + path + "' does not exists.")
            else:
                raise IOError("assert_exists @ file: " + "'" + path + "' already exists.")
        else:
            if assertion is True:
                warn("assert_exists @ file: " + "'" + path + "' does not exists.")
            else:
                warn("assert_exists @ file: " + "'" + path + "' already exists.")


def is_file(path):
    return os.path.isfile(path)


def create_path(path):
    assert_exists(path, assertion=False, raising=False)
    try:
        os.makedirs(path)
        return True
    except Exception, e:
        raise


def rename_path(oldPath, newPath):
    assert_exists(oldPath)
    assert_exists(newPath, assertion=False)
    try:
        os.name(oldPath, newPath)
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
