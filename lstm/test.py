# coding:utf-8

import utils
import file
import sample
import model


if __name__ == '__main__':

    try:
        # utils.test()
        # file.test()
        # sample.test()
        model.test()
        print 'Exit'

    except KeyboardInterrupt, e:
        exit()
    except Exception, e:
        print str(type(e)) + e.message
