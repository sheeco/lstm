# coding:utf-8

import config

if __debug__:
    config.update_config(key='debug')

import utils
import filer
import sampler
import model

if __name__ == '__main__':

    try:
        # if __debug__:
        #     utils.xprint('[Debug Mode]', level=1, newline=True)

        # config.test()
        # utils.test()
        # file.test()
        # filer.Logger.test()
        # sampler.Sampler.test()
        model.SharedLSTM.test()

    except KeyboardInterrupt, e:
        exit()
    except Exception, e:
        utils.handle(e)
