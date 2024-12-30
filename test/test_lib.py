##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTester_lib import CVRPTester as Tester

problem_scale = 0

test_paras = {
    0: ['vrplib_1.txt', 1, 1, 0]
}
##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')

env_params = {
    'problem_size': problem_scale,
    'pomo_size': problem_scale,
    'data_path': b + f"/data/{test_paras[problem_scale][0]}",
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'problem_scale': problem_scale
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './result/saved_CVRP100_model',  # directory path of pre-trained model and log files saved.
        'epoch': 656,  # epoch version of pre-trained model to laod.
    },
    'test_episodes': test_paras[problem_scale][1],
    'test_batch_size': test_paras[problem_scale][2],
    'begin_index': test_paras[problem_scale][3],
    'augmentation_enable': True,
    'aug_factor': 8,
    'aug_batch_size': test_paras[problem_scale][2],
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']
logger_params = {
    'log_file': {
        'desc': 'test_cvrp',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()