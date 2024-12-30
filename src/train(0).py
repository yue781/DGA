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
sys.path.insert(0, "../..")  # for problem_def
sys.path.insert(0, "../../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from CVRPTrainer import CVRPTrainer as Trainer

##########################################################################################
# parameters
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [101],
        'gamma': 0.1
    }
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,

    'size' : 'same',
    'epochs': 100,
    'train_episodes':  10000*10,

    # 'size' : 'various',
    # 'epochs': 500,
    # 'train_episodes': 1000,

    'prev_model_path': None,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 10,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_cvrp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # load training phase one model
        'path': './result/saved_CVRP_model',  # directory path of pre-trained model and log files saved.
        'epoch': 100,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_cvrp_n100_with_instNorm',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    if trainer_params['size'] == 'various':
        env_params = {
            'problem_size': 100,
            'pomo_size': 100,
            'finetune': True,
            'train_batch_size': 128
        }
        trainer = Trainer(env_params=env_params,
                          model_params=model_params,
                          optimizer_params=optimizer_params,
                          trainer_params=trainer_params)
        copy_all_src(trainer.result_folder)
        for i in range(trainer_params['epochs'] - trainer_params['model_load']['epoch']):
            # 修改env_params
            problem_scale = 101 + i
            env_params = {
                'problem_size': problem_scale,
                'pomo_size': problem_scale,
                'finetune': True,
                'train_batch_size': int(80 * ((100 / problem_scale) ** 2))
            }
            print('problem_scale={}'.format(problem_scale))
            trainer.run(env_params=env_params, idx=i + 1, epochs=trainer_params['model_load']['epoch'])

    elif trainer_params['size'] == 'same':
        problem_scale = 100
        env_params = {
            'problem_size': problem_scale,
            'pomo_size': problem_scale,
            'finetune': False,
            'train_batch_size': 128
        }
        trainer = Trainer(env_params=env_params,
                          model_params=model_params,
                          optimizer_params=optimizer_params,
                          trainer_params=trainer_params)
        copy_all_src(trainer.result_folder)
        for i in range(trainer_params['epochs']):
            trainer.run(env_params=env_params, idx=i + 1, epochs=0)


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
