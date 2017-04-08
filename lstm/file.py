# coding:utf-8

import os
import win32file
import win32con
import shutil
import copy
import numpy

import config
from utils import *


__all__ = [
    "if_exists",
    "assert_exists",
    "is_file",
    "create_path",
    "rename_path",
    "list_directory",
    "split_path",
    "split_extension",
    "read_lines",
    "test",
    "Logger"
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
    except:
        raise


def rename_path(old, new):
    assert_exists(old)
    assert_exists(new, assertion=False)
    try:
        os.rename(old, new)
        return True
    except:
        raise


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
        if path[-1] == '/' or path[-1] == '\\':
            tail = '/'
            path = path[:-1]
        else:
            tail = ''
        directory, filename = os.path.split(path)
        return directory + '/', filename + tail
    except:
        raise


def split_extension(filename):
    try:
        body, extension = os.path.splitext(filename)
        return body, extension
    except:
        raise


def copy_file(frompath, topath):
    try:
        shutil.copy(frompath, topath)
    except:
        raise 


def read_lines(path):
    assert_exists(path)
    try:
        return open(path, 'rb').readlines()
    except:
        raise


def test():
    path = config.PATH_LOG
    _hidden = is_hidden(path)
    hide_path(path)
    unhide_path(path)


class Logger:

    def __init__(self, path=config.PATH_LOG, identifier=None):
        self.root_path = path
        if not if_exists(path):
            create_path(self.root_path)
        if not(self.root_path[-1] == '/' or self.root_path[-1] == '\\'):
            self.root_path += '/'
        self.root_path = self.root_path.replace('\\', '/')
        self.real_path = self.root_path
        self.identifier = identifier

        # {'name':
        #   content=[(tag1, [...]),
        #            (tag2, [...])
        #           ]
        # }
        self.logs = {}

        if self.identifier is not None:
            self.real_path = self.real_path + '.' + self.identifier + '/'
            create_path(self.real_path)
            hide_path(self.real_path)

            self.filename_console = 'console'
            self.register(self.filename_console)
            self.copy_config()

    def copy_config(self):
        try:
            copy_file('./config.py', self.real_path)

        except:
            raise

    def register(self, name, tags=None):
        try:
            if name not in self.logs:
                content = []
                if tags is None:
                    tags = ['']
                for tag in tags:
                    content += [(tag, [])]
                self.logs[name] = content

                filepath = self.real_path + name + '.log'
                pfile = open(filepath, 'a')
                hastag = False
                for tag in tags:
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

    def log(self, content, name=None):
        try:
            if name is None:
                name = self.filename_console
            if name not in self.logs:
                raise ValueError("""log @ Logger: Cannot find '%s' in log registry.
                                    Must `register` first.""" % name)
            else:
                registry = self.logs[name]
            path = self.real_path + name + '.log'
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
                    raise ValueError("log @ Logger: Cannot find tag %s in log registry. " % dict_content.keys())

                for column in registry:
                    pfile.write('%s\t' % column[1][-1])
                pfile.write('\n')

            elif isinstance(content, str):
                column0 = registry[0]
                tag0 = column0[0]
                rows0 = column0[1]

                if len(registry) > 1 or tag0 != '':
                    raise ValueError("log @ Logger: A tag among %s is requested. " % [column[0] for column in registry])

                rows0 += [content]
                pfile.write(content)

            else:
                assert_unreachable()

            pfile.flush()
            pfile.close()

        except:
            raise

    def complete(self):
        try:
            if self.identifier is None:
                return
            directory, filename = split_path(self.real_path)
            if filename[0] == '.':
                filename = filename[1:]
            complete_path = directory + filename
            rename_path(self.real_path, complete_path)
            self.real_path = complete_path
            unhide_path(self.real_path)

        except:
            raise

    @staticmethod
    def test():
        try:
            logger = Logger()
            tags = ['time', 'x', 'y']
            arr = numpy.zeros((2, 3, 2))
            _dict = {'time': [1, 2, 3], 'x': ['x1', 'x2', 'x3'], 'y': [arr, arr, arr]}

            logger.register('test', tags)
            logger.log(_dict, name='test')

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

            logger.log('test console output ...')
            logger.complete()

        except:
            raise

