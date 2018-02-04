# coding:utf-8

import utils
import lstm

if __name__ == '__main__':

    try:
        # lstm.utils.process_command_line_args()
        # import config
        # config.test()
        # utils.test()
        # utils.Timer.test()
        # lstm.sampler.Sampler.test()

        lstm.model.SocialLSTM.demo()

    except Exception, e:
        utils.handle(e, exiting=True)
    else:
        exit("Exit Successfully.")

