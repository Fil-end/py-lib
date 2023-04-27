import collections
import random
import numpy as np
import math

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)
    


class SumTree:
    def __init__(self, capacity):
        # sum tree 能存储的最多优先级个数
        self.capacity = capacity
        # 顺序表存储二叉树
        self.tree = [0] * (2 * capacity - 1)
        # 每个优先级所对应的经验片段
        self.data = [None] * capacity
        self.size = 0
        self.curr_point = 0

    # 添加一个节点数据，默认优先级为当前的最大优先级+1
    def add(self, data):
        self.data[self.curr_point] = data

        self.update(self.curr_point, max(self.tree[self.capacity - 1:self.capacity + self.size]) + 1)

        self.curr_point += 1
        if self.curr_point >= self.capacity:
            self.curr_point = 0

        if self.size < self.capacity:
            self.size += 1

    # 更新一个节点的优先级权重
    def update(self, point, weight):
        idx = point + self.capacity - 1
        change = weight - self.tree[idx]

        self.tree[idx] = weight

        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] += change
            parent = (parent - 1) // 2

    def get_total(self):
        return self.tree[0]

    # 获取最小的优先级，在计算重要性比率中使用
    def get_min(self):
        return min(self.tree[self.capacity - 1:self.capacity + self.size - 1])

    # 根据一个权重进行抽样
    def sample(self, v):
        idx = 0
        while idx < self.capacity - 1:
            l_idx = idx * 2 + 1
            r_idx = l_idx + 1
            if self.tree[l_idx] >= v:
                idx = l_idx
            else:
                idx = r_idx
                v = v - self.tree[l_idx]

        point = idx - (self.capacity - 1)
        # 返回抽样得到的 位置，transition信息，该样本的概率
        return point, self.data[point], self.tree[idx] / self.get_total()
    

class ReplayBuffer_PER(object):
    def __init__(self, batch_size, max_size, beta):
        self.batch_size = batch_size  # mini batch大小
        self.max_size = 2**math.floor(math.log2(max_size)) # 保证 sum tree 为完全二叉树
        self.beta = beta

        self._sum_tree = SumTree(max_size)

    def store_transition(self, s, a, r, s_, done):
        self._sum_tree.add((s, a, r, s_, done))

    def get_mini_batches(self):
        n_sample = self.batch_size if self._sum_tree.size >= self.batch_size else self._sum_tree.size
        total = self._sum_tree.get_total()
        
        # 生成 n_sample 个区间
        step = total // n_sample
        points_transitions_probs = []
        # 在每个区间中均匀随机取一个数，并去 sum tree 中采样
        for i in range(n_sample):
            v = np.random.uniform(i * step, (i + 1) * step - 1)
            t = self._sum_tree.sample(v)
            points_transitions_probs.append(t)

        points, transitions, probs = zip(*points_transitions_probs)

        # 计算重要性比率
        max_impmortance_ratio = (n_sample * self._sum_tree.get_min())**-self.beta
        importance_ratio = [(n_sample * probs[i])**-self.beta / max_impmortance_ratio
                            for i in range(len(probs))]

        return points, tuple(np.array(e) for e in zip(*transitions)), importance_ratio

    def update(self, points, td_error):
        for i in range(len(points)):
            self._sum_tree.update(points[i], td_error[i])
