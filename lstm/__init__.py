# coding:utf-8

import numpy


def _set_numpy_():
    """
    Set print options for numpy.
    """
    numpy.set_printoptions(precision=2, suppress=False)


_set_numpy_()


# import all the files first to ensure that
# all the needed packages get imported before the handler is reset
import config
import utils
import sampler
import model

import thread
import win32api


def _set_handler_():
    """
    Enable to catch KeyboardInterrupt.
    """
    # Load the DLL manually to ensure its handler gets
    # set before our handler.
    # basepath = imp.find_module('numpy')[1]
    # ctypes.CDLL(os.path.join(basepath, 'core', 'libmmd.dll'))
    # ctypes.CDLL(os.path.join(basepath, 'core', 'libifcoremd.dll'))

    # Now set our handler for CTRL_C_EVENT. Other control event
    # types will chain to the next handler.
    def handler(dw_ctrl_type, hook_sigint=thread.interrupt_main):
        if dw_ctrl_type == 0:  # CTRL_C_EVENT
            hook_sigint()
            return 1  # don't chain to the next handler
        return 0  # chain to the next handler

    win32api.SetConsoleCtrlHandler(handler, 1)


_set_handler_()

PATH_CONFIG = "lstm/lstm.config"


def _init_config_():
    """
    Initialize for global variables in config module.
    """
    try:
        echo = config.update_config_from_file(PATH_CONFIG, group='default')
        utils.xprint(echo, newline=True)
        if __debug__:
            echo = config.update_config_from_file(PATH_CONFIG, group='debug')
            utils.xprint(echo, newline=True)
        else:
            echo = config.update_config_from_file(PATH_CONFIG, group='run')
            utils.xprint(echo, newline=True)

    except:
        raise


_init_config_()
utils.sub_logger.log_config()

