import numpy as np
import pickle
from copy import deepcopy

class EnvCore(object):
    """
    # 环境中的智能体
    """

    def __init__(self, eval_data):
        with open(eval_data, "rb") as file:
            loaded_data = pickle.load(file)
        _, _, _, _, eps, _, _, _, _, _, _, _, _, attend_path_id, path2neighboredge, data, edge_index, action_mask = loaded_data
        self.attend_path_id = attend_path_id
        self.path2neighboredge = path2neighboredge

        self.edge_index = edge_index
        self.action_mask = action_mask

        a, self.e = self.action_mask.shape
        self.edge_action = -1 * np.ones((self.e, 4))

        self.state = data
        self.agent_num = action_mask.shape[0]
        self.one_dim = 6
        self.action_dim = self.action_mask.sum(1).max()
        self.obs_dim = self.one_dim * self.action_dim
        self.keep_time = 5
        self.get_avail_actions()

        # [新增] 统计相关变量初始化
        self.global_step_count = 0  # 全局时间步
        self.path_last_success_time = {}  # 记录每个path_id上一次成功的时间 {path_id: step_time}
        self.completion_intervals = []  # 存储所有的间隔数据 [2, 5, 1, ...]

    def seed(self, seed):
        self.rng = np.random.RandomState(seed=seed)

    def reset(self):
        """
        # self.agent_num设定为2个智能体时，返回值为一个list，每个list里面为一个shape = (self.obs_dim, )的观测数据
        """
        # [新增] 重置统计变量 (如果希望跨episode统计，可以注释掉这几行)
        self.global_step_count = 0
        self.path_last_success_time = {}
        self.completion_intervals = []

        sub_agent_obs = []
        for i in range(self.agent_num):
            sub_obs_ = self.state[:self.e,:][self.action_mask[i] == 1].reshape(-1)
            pad_width = self.obs_dim - self.action_mask[i].sum() * self.one_dim
            sub_obs = np.pad(sub_obs_, (0, pad_width), mode='constant')
            sub_agent_obs.append(sub_obs)
        return sub_agent_obs, self.avail_actions

    def step(self, actions):
        """
        # self.agent_num设定为2个智能体时，actions的输入为一个2纬的list...
        """
        # [新增] 时间步推进
        self.global_step_count += 1

        path_n = 0
        sub_agent_obs = []
        sub_agent_reward = np.zeros(self.agent_num)
        sub_agent_done = []
        # sub_agent_info = []
        for i in range(self.agent_num):
            index = np.where(self.action_mask[i, :] == 1)[0]
            action_ = np.where(actions[i] == 1)[0]
            action = index[action_]
            if self.rng.random() <= self.state[action, 0]:
                replace_index = np.argmin(self.state[action, 2:]) + 2
                self.state[action, replace_index] = self.keep_time
                self.edge_action[action, replace_index - 2] = i
                self.state[action, 1] = np.sum(
                    self.state[action, 2:] != 0)

        temp_state = self.state.copy()

        for path_id in self.attend_path_id:
            test_state = temp_state.copy()
            edge_ids = self.path2neighboredge[path_id]
            edge_fea = temp_state[edge_ids, 2:]
            edge_action_fea = self.edge_action[edge_ids]
            
            # 调用更新函数
            sub_agent_reward_, neighboredge_fea, clear_indices = self.update_path_fea(
                path_id, edge_fea, edge_action_fea)
            
            if clear_indices.shape[1] > 0:
                rows = np.arange(clear_indices.shape[0])[:, None]  # 创建行索引
                test = self.state[edge_ids, 2:].copy()
                test[rows, clear_indices] = 0  # 使用高级索引进行修改
                self.state[edge_ids, 2:] = test
            sub_agent_reward += sub_agent_reward_
            temp_state[edge_ids, 2:] = neighboredge_fea
            path_n += self.state[path_id, 1]

        time_slot_fea = self.state[:, 2:]
        time_slot_fea[time_slot_fea > 0] -= 1
        self.state[:, 1] = np.sum(self.state[:, 2:] != 0, axis=1)

        for i in range(self.agent_num):
            sub_obs_ = self.state[:self.e,:][self.action_mask[i] == 1].reshape(-1)
            pad_width = self.obs_dim - self.action_mask[i].sum() * self.one_dim
            sub_obs = np.pad(sub_obs_, (0, pad_width), mode='constant')
            sub_agent_obs.append(sub_obs)
            sub_agent_done.append(False)
        sub_agent_info = self.completion_intervals.copy()

        return [sub_agent_obs, sub_agent_reward, sub_agent_done, sub_agent_info, self.avail_actions, path_n]

    def get_avail_actions(self):
        self.avail_actions = []
        for i in range(self.agent_num):
            sub_avail_actions = np.zeros(self.action_dim)
            sub_avail_actions[:self.action_mask[i].sum()] = 1
            self.avail_actions.append(sub_avail_actions)
        return self.avail_actions

    def update_path_fea(self, path_id, neighboredge_fea, neighboredge_action_fea):
        reward = np.zeros(self.agent_num)
        sorted_fea = np.sort(neighboredge_fea, axis=1)[:, ::-1]
        sorted_indices = np.argsort(neighboredge_fea, axis=1)[:, ::-1]
        sorted_action_fea = neighboredge_action_fea[np.arange(
            neighboredge_action_fea.shape[0])[:, None], sorted_indices]
        self.state[path_id, 2:] = np.min(sorted_fea, 0)
        clear_indices = sorted_indices[:, self.state[path_id, 2:] != 0]
        self.state[path_id, 1] = np.sum(self.state[path_id, 2:] != 0)
        unique_elements, counts = np.unique(
            sorted_action_fea[:, self.state[path_id, 2:] != 0], return_counts=True)
        
        # 如果 unique_elements 不为空，说明发生了资源的消耗/清除，意味着 Path 成功建立/传输
        if len(unique_elements) != 0:
            # [新增] 统计间隔代码
            current_time = self.global_step_count
            if path_id in self.path_last_success_time:
                # 如果以前成功过，计算间隔
                interval = current_time - self.path_last_success_time[path_id]
                self.completion_intervals.append(interval)
            
            # 更新该 path 最后一次成功的时间
            self.path_last_success_time[path_id] = current_time

            index = sorted_indices[:, self.state[path_id, 2:] != 0]
            neighboredge_fea[np.arange(index.shape[0])[:, None], index] = 0
            reward[unique_elements.astype(int)] = 1

        return reward, neighboredge_fea, clear_indices

    # [新增] 获取用于绘制CDF的数据
    def get_cdf_data(self):
        """
        返回排序后的间隔数据和对应的CDF概率值
        """
        if not self.completion_intervals:
            return [], []
        
        data_sorted = np.sort(self.completion_intervals)
        # y轴数据：从 1/N 到 1
        p = 1. * np.arange(1, len(data_sorted) + 1) / len(data_sorted)
        return data_sorted, p