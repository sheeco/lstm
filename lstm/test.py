# coding:utf-8

import utils
import filer
import sampler
import model


if __name__ == '__main__':

    try:
        if __debug__:
            utils.xprint('[Debug Mode]', level=1, newline=True)

        # utils.test()
        # file.test()
        # filer.Logger.test()
        # sampler.Sampler.test()
        model.SharedLSTM.test()

    except KeyboardInterrupt, e:
        exit()
    except Exception, e:
        utils.handle(e)
