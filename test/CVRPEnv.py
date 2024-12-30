from dataclasses import dataclass
import torch
from CVRProblemDef import augment_xy_data_by_8_fold
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os

@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class CVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']
        self.data_path = env_params['data_path']
        self.raw_data_depot = []
        self.raw_data_node = []
        self.raw_data_demand = []
        self.raw_data_cost = []
        self.episode = None
        self.step_state = None

        self.FLAG__use_saved_problems = False
        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.cost = None
        self.distances = None

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def use_saved_problems(self, filename, device):
        self.FLAG__use_saved_problems = True

        loaded_dict = torch.load(filename, map_location=device)
        self.saved_depot_xy = loaded_dict['depot_xy']
        self.saved_node_xy = loaded_dict['node_xy']
        self.saved_node_demand = loaded_dict['node_demand']
        self.saved_index = 0

    def load_problems(self, episode, batch_size, aug_factor=1):
        self.batch_size = batch_size
        self.episode = episode

        if not self.FLAG__use_saved_problems:
            depot_xy = self.raw_data_depot[episode:episode + batch_size]
            node_xy = self.raw_data_node[episode:episode + batch_size]
            node_demand = self.raw_data_demand[episode:episode + batch_size]
            self.cost = self.raw_data_cost[episode:episode + batch_size]
        else:
            depot_xy = self.saved_depot_xy[self.saved_index:self.saved_index + batch_size]
            node_xy = self.saved_node_xy[self.saved_index:self.saved_index + batch_size]
            node_demand = self.saved_node_demand[self.saved_index:self.saved_index + batch_size]
            self.saved_index += batch_size

        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                node_xy = augment_xy_data_by_8_fold(node_xy)
                node_demand = node_demand.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand

        self.distances = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def load_raw_data(self, episode, begin_index=0):
        print('load raw dataset begin!')

        for line in tqdm(open(self.data_path, "r").readlines()[0 + begin_index:episode + begin_index], ascii=True):
            line = line.strip().split(',')

            # 读取depot数据
            depot_index = line.index('depot')
            depot_xy = [float(line[depot_index + 1]), float(line[depot_index + 2])]
            self.raw_data_depot.append(depot_xy)

            # 读取customer数据
            customer_index = line.index('customer')
            capacity_index = line.index('capacity')
            node_xy = [[float(line[i]), float(line[i + 1])] for i in range(customer_index + 1, capacity_index, 2)]
            self.raw_data_node.append(node_xy)

            # 读取demand数据
            demand_index = line.index('demand')
            cost_index = line.index('cost')
            demand = [int(line[i]) for i in range(demand_index + 1, cost_index)]
            self.raw_data_demand.append(demand)

            # 读取cost数据
            cost = float(line[cost_index + 1])
            self.raw_data_cost.append(cost)

        problem_scale = self.problem_size
        if problem_scale == 20:
            demand_scaler = 30
        elif problem_scale == 50:
            demand_scaler = 40
        elif problem_scale == 100:
            demand_scaler = 50
        elif problem_scale == 200:
            demand_scaler = 80
        elif problem_scale == 300:
            demand_scaler = 90
        elif problem_scale == 500:
            demand_scaler = 100
        elif problem_scale == 1000:
            demand_scaler = 250
        else:
            raise NotImplementedError
        self.raw_data_depot = torch.tensor(self.raw_data_depot, requires_grad=False).view(episode, 1, 2)
        self.raw_data_node = torch.tensor(self.raw_data_node, requires_grad=False).view(episode, problem_scale, 2)
        self.raw_data_demand = torch.tensor(self.raw_data_demand, requires_grad=False).view(episode, problem_scale) / float(demand_scaler)
        self.raw_data_cost = torch.tensor(self.raw_data_cost, requires_grad=False)
        print(f'load raw dataset done!', )

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + 1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        reward_best = None
        return self.step_state, reward, reward_best, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][
            ~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()

        #偏移量，用于避免舍入误差
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            reward, reward_best = self._get_travel_distance()  # note the minus sign!
        else:
            reward, reward_best = None, None

        return self.step_state, reward, reward_best, done

    def _get_travel_distance(self):

        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)

        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)
        travel_distances_best = self.raw_data_cost
        return -travel_distances, travel_distances_best

    def current_node_distance(self):
        if self.current_node is None:
            return None

        # 获得当前节点的索引
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1,
                                                                  self.problem_size + 1)
        expanded_dist = self.distances[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1,
                                                             self.problem_size + 1)
        # 根据当前节点的索引提取元素
        cur_node_dist = torch.gather(expanded_dist, dim=2, index=current_node).squeeze(2)

        return cur_node_dist

    def plot_route(self, batch_idx=0, pomo_idx=0):
        """
        Plot the route of visited nodes for a specific batch and pomo instance.

        Args:
            batch_idx (int): Index of the batch to plot.
            pomo_idx (int): Index of the pomo instance to plot.
        """
        # 获取仓库和节点的坐标
        depot_xy = self.depot_node_xy[batch_idx, 0].cpu().numpy()
        node_xy = self.depot_node_xy[batch_idx, 1:].cpu().numpy()

        # 获取访问顺序
        visited_order = self.selected_node_list[batch_idx, pomo_idx].cpu().numpy()

        # 清除之前的图像，准备绘制新的
        plt.clf()

        # 绘制仓库和节点
        plt.scatter(depot_xy[0], depot_xy[1], color='red', marker='s', s=2)
        plt.scatter(node_xy[:, 0], node_xy[:, 1], color='blue', s=2)

        # 设置颜色循环
        colors = plt.cm.get_cmap('tab20', 20)  # 自动分配不同颜色
        color_index = 0  # 初始颜色索引

        # 用来存储每段路径的坐标
        path_segment = [depot_xy]
        for i, node in enumerate(visited_order):
            if node == 0 and i != 0:  # 每次返回仓库时绘制段路径
                path_segment = np.array(path_segment)
                plt.plot(path_segment[:, 0], path_segment[:, 1],
                         color=colors(color_index), linestyle='-', marker='o', markersize=2)
                color_index += 1
                path_segment = [depot_xy]  # 开始新路径段
            if node > 0:
                path_segment.append(node_xy[node - 1])  # 将节点加入路径段

        # 最后一段路径回到仓库
        path_segment.append(depot_xy)
        path_segment = np.array(path_segment)
        plt.plot(path_segment[:, 0], path_segment[:, 1],
                 color=colors(color_index), linestyle='-', marker='o', markersize=2)

        # 添加标题和图例
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Route for Batch {batch_idx}, Pomo {pomo_idx}')
        plt.grid()

        plt.savefig(os.path.abspath(".").replace('\\', '/')+ f"/images/batch_{batch_idx}_pomo_{pomo_idx}.png")