import torch

import os
from logging import getLogger

from CVRPEnv_lib import CVRPEnv as Env
from CVRPModel import CVRPModel as Model

from utils.utils import *
# import argparse


class CVRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):
        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        self.env.load_raw_data()

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        # if self.tester_params['test_data_load']['enable']:
        #     self.env.use_saved_problems(self.tester_params['test_data_load']['filename'], self.device)

        test_num_episode = self.tester_params['test_episodes']
        episode = self.tester_params['begin_index']

        problems_200 = []
        problems_200_500 = []
        problems_500_1000 = []

        # problems_A = []
        # problems_B = []
        # problems_E = []
        # problems_F = []
        # problems_M = []
        # problems_P = []
        # problems_X = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student, problems_size, vrpname = self._test_one_batch(episode, batch_size)

            current_gap = (score_student - score_teacher) / score_teacher

            if problems_size <= 200:
                problems_200.append(current_gap)
            elif 200 < problems_size <= 500:
                problems_200_500.append(current_gap)
            elif 500 < problems_size <= 1000:
                problems_500_1000.append(current_gap)

            # if vrpname[:2] == 'A-':
            #     problems_A.append(current_gap)
            # elif vrpname[:2] == 'B-':
            #     problems_B.append(current_gap)
            # elif vrpname[:2] == 'E-':
            #     problems_E.append(current_gap)
            # elif vrpname[:2] == 'F-':
            #     problems_F.append(current_gap)
            # elif vrpname[:2] == 'M-':
            #     problems_M.append(current_gap)
            # elif vrpname[:2] == 'P-':
            #     problems_P.append(current_gap)
            # elif vrpname[:2] == 'X-':
            #     problems_X.append(current_gap)

            self.logger.info(
                " problems_200 mean gap:{:4f}%, num:{}".format(np.mean(problems_200) * 100, len(problems_200)))
            self.logger.info(
                " problems_200_500 mean gap:{:4f}%, num:{}".format(np.mean(problems_200_500) * 100, len(problems_200_500)))
            self.logger.info(
                " problems_500_1000 mean gap:{:4f}%, num:{}".format(np.mean(problems_500_1000) * 100, len(problems_500_1000)))
            # self.logger.info(
            #     " problems_A    mean gap:{:4f}%, num:{}".format(np.mean(problems_A) * 100, len(problems_A)))
            # self.logger.info(
            #     " problems_B    mean gap:{:4f}%, num:{}".format(np.mean(problems_B) * 100, len(problems_B)))
            # self.logger.info(
            #     " problems_E    mean gap:{:4f}%, num:{}".format(np.mean(problems_E) * 100, len(problems_E)))
            # self.logger.info(
            #     " problems_F    mean gap:{:4f}%, num:{}".format(np.mean(problems_F) * 100, len(problems_F)))
            # self.logger.info(
            #     " problems_M    mean gap:{:4f}%, num:{}".format(np.mean(problems_M) * 100, len(problems_M)))
            # self.logger.info(
            #     " problems_P    mean gap:{:4f}%, num:{}".format(np.mean(problems_P) * 100, len(problems_P)))
            # self.logger.info(
            #     " problems_X    mean gap:{:4f}%, num:{}".format(np.mean(problems_X) * 100, len(problems_X)))

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info(
                "episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Best_Score:{:.4f}, Aug_Score: {:.4f}".format(
                    episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                all_result_gaps = problems_200+problems_200_500+problems_500_1000
                gap_ = np.mean(all_result_gaps) * 100
                self.logger.info(" Gap: {:.4f}%".format(gap_))

    def _test_one_batch(self, episode, batch_size):

        # Augmentation
        ###############################################
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
        else:
            aug_factor = 1

        # Ready
        ###############################################
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems(episode, batch_size, aug_factor)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

        # POMO Rollout
        ###############################################
        state, reward, reward_best, done = self.env.pre_step()
        while not done:
            cur_dist = self.env.current_node_distance()
            selected, _ = self.model(state, cur_dist)
            # shape: (batch, pomo)
            state, reward, reward_best, done = self.env.step(selected)
            self.env.plot_route(batch_idx=0, pomo_idx=0)

        # Return
        ###############################################
        best_score = -reward_best.mean()
        aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
        # shape: (augmentation, batch, pomo)

        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        # shape: (augmentation, batch)
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        # shape: (batch,)
        aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value

        return best_score.item(), aug_score.item(), self.env.problem_size, self.env.vrplib_name
