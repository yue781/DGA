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
    dist: torch.Tensor = None
    # shape: (batch, problem+1, problem+1)

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
    def __init__(self, pomo_size, **env_params):
        # Const @INIT
        ####################################
        self.input_mask = None
        self.vrplib = False
        self.env_params = env_params
        self.problem_size = None
        self.pomo_size = pomo_size
        self.step_state = None
        self.unscaled_depot_node_xy = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.dist = None
        # shape: (batch, problem+1, problem+1)

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

    def load_vrplib_problems(self, instance, aug_factor=1):
        self.vrplib = True
        self.batch_size = 1

        node_coords = torch.FloatTensor(instance['node_coord']).unsqueeze(0)
        demands = torch.FloatTensor(instance['demand']).unsqueeze(0)
        demands = demands / instance['capacity']
        self.unscaled_depot_node_xy = node_coords
        # shape: (batch, problem+1, 2)

        min_x = torch.min(node_coords[:, :, 0], 1)[0]
        min_y = torch.min(node_coords[:, :, 1], 1)[0]
        max_x = torch.max(node_coords[:, :, 0], 1)[0]
        max_y = torch.max(node_coords[:, :, 1], 1)[0]
        scaled_depot_node_x = (node_coords[:, :, 0] - min_x) / (max_x - min_x)
        scaled_depot_node_y = (node_coords[:, :, 1] - min_y) / (max_y - min_y)
        self.depot_node_xy = torch.cat((scaled_depot_node_x[:, :, None], scaled_depot_node_y[:, :, None]), dim=2)
        depot_xy = self.depot_node_xy[:, instance['depot'], :]
        # shape: (batch, problem+1)
        if aug_factor > 1:
            if aug_factor == 8:
                self.batch_size = self.batch_size * 8
                depot_xy = augment_xy_data_by_8_fold(depot_xy)
                self.depot_node_xy = augment_xy_data_by_8_fold(self.depot_node_xy)
                self.unscaled_depot_node_xy = augment_xy_data_by_8_fold(self.unscaled_depot_node_xy)
                demands = demands.repeat(8, 1)
            else:
                raise NotImplementedError

        self.depot_node_demand = demands
        # shape: (batch, problem+1)
        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = self.depot_node_xy[:, 1:, :]
        self.reset_state.node_demand = demands[:, 1:]
        self.problem_size = self.reset_state.node_xy.shape[1]

        self.dist = (self.depot_node_xy[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
        # shape: (batch, problem+1, problem+1)
        self.reset_state.dist = self.dist

        # self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        # self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        #
        # self.step_state.BATCH_IDX = self.BATCH_IDX
        # self.step_state.POMO_IDX = self.POMO_IDX

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
        if self.input_mask is not None:
            self.visited_ninf_flag = self.input_mask[:, None, :].expand(self.batch_size, self.pomo_size, self.problem_size+1).clone()
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
        return self.step_state, reward, done

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

        self.visited_ninf_flag.scatter_(2, self.selected_node_list, float('inf'))
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 1e-6
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
            reward = self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

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

        return -travel_distances

    def _get_unscaled_travel_distance(self, solutions=None, rounding=True):
        if solutions is None:
            solutions = self.selected_node_list
        gathering_index = solutions[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.unscaled_depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)
        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)
        rolled_seq = ordered_seq.roll(dims=2, shifts=-1)
        segment_lengths = ((ordered_seq - rolled_seq) ** 2).sum(3).sqrt()
        # shape: (batch, pomo, selected_list_length)
        if rounding:
            segment_lengths = torch.round(segment_lengths)
        travel_distances = segment_lengths.sum(2)
        # shape: (batch, pomo)

        return -travel_distances
    
    def current_node_distance(self):
        if self.current_node is None:
            return None

        # 获得当前节点的索引
        current_node = self.current_node[:, :, None, None].expand(self.batch_size, self.pomo_size, 1,
                                                                  self.problem_size + 1)
        expanded_dist = self.dist[:, None, :, :].expand(self.batch_size, self.pomo_size, self.problem_size + 1,
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
        plt.scatter(depot_xy[0], depot_xy[1], color='grey', marker='s', s=2)
        plt.scatter(node_xy[:, 0], node_xy[:, 1], color='grey', s=2)

        # 设置颜色循环
        colors = plt.cm.get_cmap('tab20', 20)
        colors = colors(np.linspace(0,1,20))
        colors = np.tile(colors, (5,1))
        color_index = 0  # 初始颜色索引

        # 用来存储每段路径的坐标
        path_segment = [depot_xy]
        for i, node in enumerate(visited_order):
            if node == 0 and i != 0:  # 每次返回仓库时绘制段路径
                path_segment = np.array(path_segment)
                plt.plot(path_segment[:, 0], path_segment[:, 1],
                         color=colors[color_index], linestyle='-', marker='o', markersize=2)
                color_index += 1
                path_segment = [depot_xy]  # 开始新路径段
            if node > 0:
                path_segment.append(node_xy[node - 1])  # 将节点加入路径段

        # 最后一段路径回到仓库
        path_segment.append(depot_xy)
        path_segment = np.array(path_segment)
        plt.plot(path_segment[:, 0], path_segment[:, 1],
                 color=colors[color_index], linestyle='-', marker='o', markersize=2)

        # 添加标题和图例
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Route for Batch {batch_idx}, Pomo {pomo_idx}')
        plt.grid()

        plt.savefig(os.path.abspath(".").replace('\\', '/')+ f"/images/batch_{batch_idx}_pomo_{pomo_idx}.png")