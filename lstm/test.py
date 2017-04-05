# coding:utf-8

import utils
import file
from sampler import *
from model import *


if __name__ == '__main__':

    try:
        if __debug__:
            utils.xprint('[Debug Mode]', level=1, newline=True)

        # utils.test()
        # file.test()
        # Sampler.test()
        SharedLSTM.test()
        utils.xprint('', newline=True)
        utils.xprint('Completed', newline=True)

    except KeyboardInterrupt, e:
        exit()
    except Exception, e:
        utils.handle(e)
