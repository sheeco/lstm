# coding:utf-8

# import all the files first to ensure that
# all the needed packages get imported before the handler is reset
import config
import utils
import filer
import sampler
import model

import thread
import win32api


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
