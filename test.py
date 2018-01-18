# coding:utf-8

import lstm

if __name__ == '__main__':

    try:
        # lstm.utils.process_command_line_args()
        #
        # lstm.config.test()
        # lstm.utils.test()
        # lstm.utils.Timer.test()
        # lstm.sampler.Sampler.test()

        lstm.model.SocialLSTM.test()

        exit("Exit Successfully.")

    except Exception, e:
        lstm.utils.handle(e, exiting=True)
