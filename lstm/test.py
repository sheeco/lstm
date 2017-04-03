# coding:utf-8

import utils
import file
from sampler import *
from model import *


if __name__ == '__main__':

    try:
        if __debug__:
            print '[Debug Mode]'

        # utils.test()
        # file.test()
        # Sampler.test()
        SharedLSTM.test()
        print 'Exit'

    except KeyboardInterrupt, e:
        exit()
    except Exception, e:
        utils.handle(e)
