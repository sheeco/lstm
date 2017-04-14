# coding:utf-8

import lstm.config
import lstm.utils
import lstm.filer
import lstm.sampler
import lstm.model

if __name__ == '__main__':

    try:
        lstm.utils.parse_command_line_args()

        # lstm.config.test()
        # lstm.utils.test()
        # lstm.filer.test()
        # lstm.filer.Logger.test()
        # lstm.sampler.Sampler.test()

        lstm.model.SharedLSTM.test()

    except KeyboardInterrupt, e:
        print "\nStopped manually."
        lstm.utils.handle(e)
    except Exception, e:
        lstm.utils.handle(e)
