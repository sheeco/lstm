# coding:utf-8

import utils
import file
import Sampler
import model


if __name__ == '__main__':

    try:
        # utils.test()
        # file.test()
        Sampler.test()
        model.test()
        print 'Exit'

    except KeyboardInterrupt, e:
        exit()
    except Exception, e:
        utils.handle(e)
