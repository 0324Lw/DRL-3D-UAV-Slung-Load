import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import time
from collections import deque
from env import SlungLoadUAVEnv, Config, Plot


# ==========================================
# 1. PPO 超参数配置
# ==========================================
class PPOConfig:
    total_timesteps = 3_000_000  # 总训练步数
    num_steps = 2048  # 每次更新前收集的步数 (Rollout length)
    batch_size = 256  # PPO 更新的 Batch Size
    n_epochs = 10  # 每次收集数据后网络更新的轮数

    learning_rate = 3e-4  # 初始学习率
    gamma = 0.99  # 折扣因子
    gae_lambda = 0.95  # GAE 参数
    clip_coef = 0.2  # PPO 截断范围
    ent_coef = 0.01  # 熵奖励系数 (鼓励探索)
    vf_coef = 0.5  # 价值损失系数
    max_grad_norm = 0.5  # 梯度裁剪阈值

    save_freq = 500_000  # 模型保存频率
    log_freq = 1  # 日志打印频率 (按回合)


# ==========================================
# 2. 神经网络结构定义 (Actor-Critic)
# ==========================================
def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """稳定性调优 1：正交初始化 (Orthogonal Initialization)"""
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Critic 网络：评估当前状态的价值 V(s)
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        # Actor 网络：输出动作的高斯分布均值
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        # 可学习的动作对数标准差 (独立于状态)
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)

        if action is None:
            action = probs.sample()

        # 稳定性调优 2：动作空间截断限制
        clipped_action = torch.clamp(action, -1.0, 1.0)

        return action, clipped_action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


# ==========================================
# 3. 核心训练主循环
# ==========================================
def train():
    cfg = PPOConfig()
    env_cfg = Config()
    env = SlungLoadUAVEnv(env_cfg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 开始训练! 使用设备: {device}")

    # 实例化网络与优化器
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=cfg.learning_rate, eps=1e-5)

    # 存储 Rollout 数据
    obs_buf = torch.zeros((cfg.num_steps, obs_dim)).to(device)
    act_buf = torch.zeros((cfg.num_steps, act_dim)).to(device)
    logprobs_buf = torch.zeros((cfg.num_steps)).to(device)
    rewards_buf = torch.zeros((cfg.num_steps)).to(device)
    dones_buf = torch.zeros((cfg.num_steps)).to(device)
    values_buf = torch.zeros((cfg.num_steps)).to(device)

    # 日志记录队列
    ep_rewards = deque(maxlen=50)
    ep_lengths = deque(maxlen=50)
    ep_payload_penalties = deque(maxlen=50)
    success_count = deque(maxlen=50)
    all_episode_rewards = []  # 用于最终绘图

    global_step = 0
    num_updates = cfg.total_timesteps // cfg.num_steps
    start_time = time.time()

    obs, _ = env.reset()
    next_obs = torch.Tensor(obs).to(device)
    next_done = torch.zeros(1).to(device)

    current_level = 1
    env.set_difficulty(current_level)
    print(f"📖 阶段 1: 基础起降与平稳飞行 (难度 Level 1)")

    os.makedirs("models", exist_ok=True)

    for update in range(1, num_updates + 1):
        # 稳定性调优 3：学习率线性衰减 (Linear LR Scheduler)
        frac = 1.0 - (update - 1.0) / num_updates
        lrnow = frac * cfg.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        # --- 课程学习进度控制器 ---
        if global_step >= 500_000 and current_level == 1:
            current_level = 2
            env.set_difficulty(current_level)
            print(f"\n🌟 进阶！开启阶段 2: 稀疏障碍物避障 (难度 Level 2)")
        elif global_step >= 1_500_000 and current_level == 2:
            current_level = 3
            env.set_difficulty(current_level)
            print(f"\n🔥 终极压榨！开启阶段 3: 密集复杂环境挑战 (难度 Level 3)")

        # --- 1. 数据收集 (Rollout) ---
        ep_reward_sum = 0
        ep_step_count = 0
        ep_payload_sum = 0

        for step in range(0, cfg.num_steps):
            global_step += 1
            obs_buf[step] = next_obs
            dones_buf[step] = next_done

            with torch.no_grad():
                action, clipped_action, logprob, _, value = agent.get_action_and_value(next_obs.unsqueeze(0))
                values_buf[step] = value.flatten()
            act_buf[step] = action.flatten()
            logprobs_buf[step] = logprob.flatten()

            # 与环境交互
            env_action = clipped_action.cpu().numpy()[0]
            next_obs_np, reward, terminated, truncated, info = env.step(env_action)
            if update % 5 == 0 and len(ep_rewards) == 0 and ep_step_count < 5:
                if ep_step_count == 0:
                    print("\n" + "=" * 50)
                    print(f"🔍 [DEBUG PROBE] 正在监控 Update {update} 的开局阶段物理状态")
                    print("=" * 50)

                print(f"▶ Step {ep_step_count + 1}:")
                print(f"  [指令] 神经网络输出动作 (期望速度 v_cmd): {env_action}")
                print(f"  [本体] 无人机位置 (X,Y,Z): {env.p_q.round(3)}")
                print(f"  [本体] 无人机实际速度: {env.v_q.round(3)}")
                print(f"  [负载] 货物位置 (X,Y,Z): {env.p_p.round(3)}")
                print(f"  [负载] 绳长: {np.linalg.norm(env.p_q - env.p_p):.3f} m")
                print(f"  [奖励] 单步总奖励: {reward:.4f}")
                print(f"  [明细] {info}")
                if terminated:
                    print(f"  🚨 [警告] 回合在此步异常终止 (Terminated)！越界或碰撞！")
            rewards_buf[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor([terminated or truncated]).to(device)

            # 记录数据
            ep_reward_sum += reward
            ep_step_count += 1
            ep_payload_sum += info.get('r_payload', 0)

            if terminated or truncated:
                ep_rewards.append(ep_reward_sum)
                ep_lengths.append(ep_step_count)
                ep_payload_penalties.append(ep_payload_sum)
                all_episode_rewards.append(ep_reward_sum)

                # 判断是否成功 (通过终点的大额奖励判断)
                is_success = 1.0 if reward > 1.0 else 0.0
                success_count.append(is_success)

                next_obs_np, _ = env.reset()
                next_obs = torch.Tensor(next_obs_np).to(device)
                ep_reward_sum = 0
                ep_step_count = 0
                ep_payload_sum = 0

        # --- 2. 优势函数估计 (GAE) ---
        with torch.no_grad():
            next_value = agent.get_value(next_obs.unsqueeze(0)).flatten()
            advantages = torch.zeros_like(rewards_buf).to(device)
            lastgaelam = 0
            for t in reversed(range(cfg.num_steps)):
                if t == cfg.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones_buf[t + 1]
                    nextvalues = values_buf[t + 1]
                delta = rewards_buf[t] + cfg.gamma * nextvalues * nextnonterminal - values_buf[t]
                advantages[t] = lastgaelam = delta + cfg.gamma * cfg.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values_buf

        # --- 3. PPO 网络更新 ---
        b_obs = obs_buf.reshape((-1, obs_dim))
        b_actions = act_buf.reshape((-1, act_dim))
        b_logprobs = logprobs_buf.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_buf.reshape(-1)

        # 稳定性调优 4：优势函数归一化 (Advantage Normalization)
        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        b_inds = np.arange(cfg.num_steps)
        clipfracs = []

        for epoch in range(cfg.n_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, cfg.num_steps, cfg.batch_size):
                end = start + cfg.batch_size
                mb_inds = b_inds[start:end]

                _, _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                # Actor 损失计算
                mb_advantages = b_advantages[mb_inds]
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - cfg.clip_coef, 1 + cfg.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Critic 损失计算
                v_loss = 0.5 * ((newvalue.view(-1) - b_returns[mb_inds]) ** 2).mean()

                # 熵奖励 (防止过早收敛)
                entropy_loss = entropy.mean()
                loss = pg_loss - cfg.ent_coef * entropy_loss + v_loss * cfg.vf_coef

                # 梯度更新
                optimizer.zero_grad()
                loss.backward()
                # 稳定性调优 5：梯度裁剪 (Gradient Clipping)
                nn.utils.clip_grad_norm_(agent.parameters(), cfg.max_grad_norm)
                optimizer.step()

        # --- 4. 日志记录与模型持久化 ---
        if update % cfg.log_freq == 0 and len(ep_rewards) > 0:
            avg_rew = np.mean(ep_rewards)
            avg_len = np.mean(ep_lengths)
            avg_payload = np.mean(ep_payload_penalties)
            success_rate = np.mean(success_count) * 100
            sps = int(global_step / (time.time() - start_time))

            print(f"| Step: {global_step:8d} | "
                  f"Avg Rew: {avg_rew:7.2f} | "
                  f"Avg Len: {avg_len:5.1f} | "
                  f"Success: {success_rate:5.1f}% | "
                  f"Payload Pen: {avg_payload:6.2f} | "
                  f"Loss: {loss.item():6.3f} | "
                  f"SPS: {sps:4d} |")

        if global_step % cfg.save_freq == 0:
            model_path = f"models/ppo_uav_step_{global_step}.pth"
            torch.save(agent.state_dict(), model_path)
            print(f"💾 模型已保存至: {model_path}")

    # --- 5. 训练结束，绘制数据曲线 ---
    print("\n🎉 训练完全结束！正在生成数据图表...")
    torch.save(agent.state_dict(), "models/ppo_uav_final.pth")
    Plot.plot_learning_curve(all_episode_rewards, window=100)


if __name__ == "__main__":
    train()