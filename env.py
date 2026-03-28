import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any


# ==========================================
# 1. Config 参数配置类
# ==========================================
@dataclass
class Config:
    # 环境设定
    env_size: np.ndarray = field(default_factory=lambda: np.array([50.0, 50.0, 20.0]))  # L, W, H
    max_steps: int = 200
    physics_dt: float = 0.01  # 物理仿真内部步长
    control_dt: float = 0.1  # RL 控制步长

    # 无人机与货物物理参数
    m_q: float = 1.5
    m_p: float = 0.5
    l_0: float = 1.0
    v_max: float = 5.0
    k_s: float = 150.0
    k_d: float = 10.0
    g: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -9.81]))

    # 障碍物与感知参数 (已修复：集中定义)
    num_obstacles_lvl2: int = 10  # 难度2的障碍物数量
    num_obstacles_lvl3: int = 15  # 难度3的障碍物数量
    obs_radius_min: float = 2.0  # 障碍物最小半径/半边长
    obs_radius_max: float = 2.5  # 障碍物最大半径/半边长
    num_rays: int = 24
    ray_max_dist: float = 10.0
    safe_radius: float = 3.0
    min_obs_gap: float = 5.0  # 障碍物之间的最小通行净间距

    # 奖励系数
    # === 优化后的奖励系数 ===

    # 1. 基础惩罚
    c_step: float = 0.01  # 保持不变，适度的存活压力

    # 2. 核心正向引导 (大幅增强)
    c_dist: float = 0.15  # 原 0.05 -> 0.15 (增强 3 倍，鼓励向终点移动)
    c_dir: float = 0.05  # 原 0.02 -> 0.05 (增强 2.5 倍，鼓励机头对准终点)

    # 3. 约束惩罚 (大幅削弱)
    c_smooth: float = 0.01  # 保持不变，防止高频抖动即可
    c_alt: float = 0.001  # 原 0.02 -> 0.001 (削弱 20 倍！解决开局自杀问题)
    c_payload: float = 0.002  # 原 0.01 -> 0.002 (削弱 5 倍，允许初期有一定的摇摆试错)

    # 4. 稀疏奖励
    c_goal: float = 2.0  # 成功奖励
    c_crash: float = -2.0  # 碰撞惩罚
    h_cruise: float = 10.0


# ==========================================
# 2. Env 环境类 (严格遵循 Gymnasium)
# ==========================================
class SlungLoadUAVEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, config: Config = Config()):
        super().__init__()
        self.cfg = config
        self.difficulty_level = 3  # 1: 无障碍, 2: 简单障碍, 3: 复杂障碍

        # 动作空间：无人机的三维期望速度分量 [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # 状态空间维度计算:
        # v_q(3) + z_q(1) + dp_pq(3) + dv_pq(3) + dp_goal(3) + d_goal(1) + rays(N)
        obs_dim = 3 + 1 + 3 + 3 + 3 + 1 + self.cfg.num_rays
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)

        self.obstacles = []  # 存储字典: {'type': 'cyl'/'cube', 'pos': [x,y], 'size': r 或 half_l}
        self.reset()

    def set_difficulty(self, level: int):
        """课程学习接口：在训练代码中动态调整环境难度"""
        assert 1 <= level <= 3
        self.difficulty_level = level

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)
        self.step_count = 0

        # 1. 生成起终点 (保证距离不小于20)
        dist = 0
        while dist < 50.0:
            self.p_start = np.array([np.random.uniform(5, self.cfg.env_size[0] - 5),
                                     np.random.uniform(5, self.cfg.env_size[1] - 5), 0.0])
            self.p_goal = np.array([np.random.uniform(5, self.cfg.env_size[0] - 5),
                                    np.random.uniform(5, self.cfg.env_size[1] - 5), 0.0])
            dist = np.linalg.norm(self.p_start[:2] - self.p_goal[:2])

        # 2. 课程学习：根据难度生成障碍物 (修复了重叠与间距逻辑)
        self.obstacles = []
        if self.difficulty_level >= 2:
            num_obs = self.cfg.num_obstacles_lvl2 if self.difficulty_level == 2 else self.cfg.num_obstacles_lvl3
            attempts = 0  # 防止死循环

            while len(self.obstacles) < num_obs and attempts < 1000:
                attempts += 1
                pos = np.random.uniform(0, self.cfg.env_size[:2])
                size = np.random.uniform(self.cfg.obs_radius_min, self.cfg.obs_radius_max)

                # a. 检查起终点安全区 (圆心距离需要大于 安全半径 + 障碍物自身半径)
                if np.linalg.norm(pos - self.p_start[:2]) < self.cfg.safe_radius + size or \
                        np.linalg.norm(pos - self.p_goal[:2]) < self.cfg.safe_radius + size:
                    continue

                # b. 检查与其他障碍物的最小间距
                overlap = False
                for obs in self.obstacles:
                    center_dist = np.linalg.norm(pos - obs['pos'])
                    # 必须满足: 两圆心距离 >= (r1 + r2) + 最小净间距
                    if center_dist < (size + obs['size'] + self.cfg.min_obs_gap):
                        overlap = True
                        break

                if not overlap:
                    obs_type = np.random.choice(['cyl', 'cube'])
                    self.obstacles.append({'type': obs_type, 'pos': pos, 'size': size})

        # 3. 初始化物理状态 (地面起飞)
        self.p_q = np.copy(self.p_start)
        self.v_q = np.zeros(3)
        self.p_p = np.copy(self.p_start)
        self.v_p = np.zeros(3)

        self.prev_action = np.zeros(3)
        self.prev_dist_to_goal = np.linalg.norm(self.p_goal - self.p_p)

        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.clip(action, -1.0, 1.0)
        self.step_count += 1

        # 1. 物理动力学子步积分 (Euler Integration)
        steps = int(self.cfg.control_dt / self.cfg.physics_dt)
        v_cmd = action * self.cfg.v_max
        perturbation_a = np.zeros(3)

        for _ in range(steps):
            # 绳索动力学
            dp = self.p_q - self.p_p
            dist = np.linalg.norm(dp)
            F_p = np.zeros(3)

            if dist > self.cfg.l_0:
                n = dp / (dist + 1e-8)
                rel_v = (self.v_q - self.v_p).dot(n)
                tension = self.cfg.k_s * (dist - self.cfg.l_0) + self.cfg.k_d * rel_v
                tension = max(0.0, tension)  # 绳子只能受拉力
                F_p = tension * n

            # 货物运动学 (含地面碰撞)
            a_p = self.cfg.g + F_p / self.cfg.m_p
            self.v_p += a_p * self.cfg.physics_dt
            self.p_p += self.v_p * self.cfg.physics_dt
            if self.p_p[2] < 0:
                self.p_p[2] = 0.0
                self.v_p[2] = max(0.0, self.v_p[2])
                self.v_p[:2] *= 0.9  # 地面摩擦

            # 无人机速度更新 (受控指令 + 货物扰动)
            F_q = -F_p
            a_q_perturb = F_q / self.cfg.m_q
            perturbation_a = a_q_perturb  # 记录供奖励函数使用
            self.v_q = v_cmd + a_q_perturb * self.cfg.physics_dt
            self.p_q += self.v_q * self.cfg.physics_dt
            if self.p_q[2] < 0:
                self.p_q[2] = 0.0
                self.v_q[2] = max(0.0, self.v_q[2])  # 禁止继续向下钻

        # 2. 获取状态与感知
        obs = self._get_obs()

        # 3. 奖励计算与截断保护
        reward, info_rewards, terminated = self._compute_reward(action, perturbation_a)

        # 单步奖励强制截断，保证训练数值稳定
        reward = np.clip(reward, -2.0, 2.0)

        truncated = bool(self.step_count >= self.cfg.max_steps)
        self.prev_action = action

        return obs, reward, terminated, truncated, info_rewards

    def _compute_reward(self, action: np.ndarray, a_perturb: np.ndarray) -> Tuple[float, Dict, bool]:
        dist_to_goal = np.linalg.norm(self.p_goal - self.p_p)
        dist_2d = np.linalg.norm(self.p_goal[:2] - self.p_q[:2])
        terminated = False

        # 1. 密集奖励组件
        r_step = -self.cfg.c_step
        r_dist = self.cfg.c_dist * (self.prev_dist_to_goal - dist_to_goal)

        v_norm = np.linalg.norm(self.v_q)
        r_dir = 0.0
        if v_norm > 0.1 and dist_to_goal > 0.1:
            dir_vec = (self.p_goal - self.p_q) / (np.linalg.norm(self.p_goal - self.p_q) + 1e-8)
            r_dir = self.cfg.c_dir * np.dot(self.v_q / v_norm, dir_vec)

        r_smooth = -self.cfg.c_smooth * np.sum(np.square(action - self.prev_action))

        # 自适应高度期望
        h_target = self.cfg.h_cruise if dist_2d > self.cfg.safe_radius else \
            max(0.5, self.cfg.h_cruise * (dist_2d / self.cfg.safe_radius))
        r_alt = -self.cfg.c_alt * ((self.p_q[2] - h_target) ** 2)

        r_payload = -self.cfg.c_payload * np.linalg.norm(self.v_p - self.v_q, ord=1)
        if self.p_p[2] > self.p_q[2]:
            r_payload -= 0.05  # 防甩上天硬惩罚

        reward = r_step + r_dist + r_dir + r_smooth + r_alt + r_payload

        # 2. 稀疏终止条件
        if self._check_collision():
            reward = self.cfg.c_crash
            terminated = True
        elif dist_2d < 0.5 and self.p_p[2] < 0.2 and np.linalg.norm(self.v_p) < 0.5:
            # 成功降落并停稳
            reward = self.cfg.c_goal + 0.05 * (self.cfg.max_steps - self.step_count)
            terminated = True

        # 越界惩罚
        if not (0 <= self.p_q[0] <= self.cfg.env_size[0] and
                0 <= self.p_q[1] <= self.cfg.env_size[1] and
                -0.1 <= self.p_q[2] <= self.cfg.env_size[2]):
            reward = self.cfg.c_crash
            terminated = True

        self.prev_dist_to_goal = dist_to_goal

        info = {
            "r_step": r_step, "r_dist": r_dist, "r_dir": r_dir,
            "r_smooth": r_smooth, "r_alt": r_alt, "r_payload": r_payload,
            "r_total_raw": reward
        }
        return float(reward), info, terminated

    def _get_obs(self) -> np.ndarray:
        # 归一化处理
        norm_v_q = self.v_q / self.cfg.v_max
        norm_z_q = np.array([(self.p_q[2] / self.cfg.env_size[2]) * 2 - 1.0])
        norm_dp_pq = (self.p_p - self.p_q) / self.cfg.l_0
        norm_dv_pq = (self.v_p - self.v_q) / (self.cfg.v_max * 2)

        dp_goal = self.p_goal - self.p_p
        dist_goal = np.linalg.norm(dp_goal)
        norm_dp_goal = dp_goal / max(dist_goal, 1e-5)
        norm_dist_goal = np.array([dist_goal / np.linalg.norm(self.cfg.env_size)])

        rays = self._cast_rays() / self.cfg.ray_max_dist  # [0, 1]

        obs = np.concatenate([
            norm_v_q, norm_z_q, norm_dp_pq, norm_dv_pq,
            norm_dp_goal, norm_dist_goal, rays
        ])
        return np.clip(obs, -1.0, 1.0).astype(np.float32)

    def _cast_rays(self) -> np.ndarray:
        """2.5D 射线检测：将障碍物视为无限高，利用水平射线相交加三角函数推算3D距离"""
        rays = np.full(self.cfg.num_rays, self.cfg.ray_max_dist)
        angles = np.linspace(0, 2 * math.pi, self.cfg.num_rays, endpoint=False)

        for i, angle in enumerate(angles):
            # 简化：仅在水平面发射射线（也可扩展3D仰角）
            ray_dir = np.array([math.cos(angle), math.sin(angle)])
            min_dist = self.cfg.ray_max_dist

            for obs in self.obstacles:
                op = obs['pos'] - self.p_q[:2]
                proj_len = np.dot(op, ray_dir)
                if proj_len > 0:
                    perp_dist = np.linalg.norm(op - proj_len * ray_dir)
                    if obs['type'] == 'cyl' and perp_dist < obs['size']:
                        hit_dist = proj_len - math.sqrt(obs['size'] ** 2 - perp_dist ** 2)
                        if hit_dist < min_dist: min_dist = hit_dist
                    elif obs['type'] == 'cube':
                        # 简化 AABB 近似碰撞
                        if np.max(np.abs(op - proj_len * ray_dir)) < obs['size']:
                            hit_dist = proj_len - obs['size']
                            if hit_dist < min_dist: min_dist = hit_dist

            rays[i] = max(0.0, min_dist)
        return rays

    def _check_collision(self) -> bool:
        rays = self._cast_rays()
        if np.min(rays) < 0.5:  # 无人机碰撞半径近似 0.5m
            return True
        # 货物碰柱子检测
        for obs in self.obstacles:
            dist = np.linalg.norm(self.p_p[:2] - obs['pos'])
            if obs['type'] == 'cyl' and dist < obs['size'] + 0.2:
                return True
            if obs['type'] == 'cube' and np.max(np.abs(self.p_p[:2] - obs['pos'])) < obs['size'] + 0.2:
                return True
        return False


# ==========================================
# 3. Plot 绘图监控类
# ==========================================
class Plot:
    @staticmethod
    def plot_learning_curve(episode_rewards: List[float], window: int = 50):
        """绘制训练奖励曲线及滑动平均"""
        plt.figure(figsize=(10, 5))
        plt.plot(episode_rewards, label='Episode Reward', alpha=0.3, color='blue')

        if len(episode_rewards) >= window:
            moving_avg = np.convolve(episode_rewards, np.ones(window) / window, mode='valid')
            plt.plot(range(window - 1, len(episode_rewards)), moving_avg,
                     label=f'{window}-Episode Moving Avg', color='red', linewidth=2)

        plt.title('Training Learning Curve')
        plt.xlabel('Episode')
        plt.ylabel('Total Reward (Clipped scale)')
        plt.legend()
        plt.grid(True)
        plt.show()

