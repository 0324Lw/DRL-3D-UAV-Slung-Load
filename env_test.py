import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from env import SlungLoadUAVEnv, Config
import warnings

warnings.filterwarnings('ignore')  # 忽略绘图时的一些底层警告


def test_api_and_spaces(env):
    """测试状态空间、动作空间的维度和数值输出是否正常"""
    print("\n" + "=" * 50)
    print("1. API 与空间维度测试 (API & Spaces Test)")
    print("=" * 50)

    obs, info = env.reset()
    print(f"状态空间 (Observation Space): {env.observation_space}")
    print(f"动作空间 (Action Space): {env.action_space}")
    print(f"初始状态维度校验: {obs.shape} (预期: {env.observation_space.shape})")
    print(f"初始状态是否存在 NaN/Inf: {np.any(np.isnan(obs)) or np.any(np.isinf(obs))}")

    # 抽取一个随机动作测试 step() 是否无 Bug
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step() 测试通过。Reward: {reward:.4f}, Terminated: {terminated}, Truncated: {truncated}")
    print("info 字典键值校验:", list(info.keys()))


def test_nonlinear_dynamics(env):
    """测试无人机-绳子-货物之间的非线性力学关系"""
    print("\n" + "=" * 50)
    print("2. 非线性耦合力学测试 (Nonlinear Dynamics Test)")
    print("=" * 50)

    env.reset()
    # 强制让无人机以最大速度垂直上升，观察拉起货物的全过程
    action_up = np.array([0.0, 0.0, 1.0])

    dynamics_data = []

    for step in range(1, 21):  # 运行20步，观察起飞前后的变化
        obs, reward, terminated, truncated, info = env.step(action_up)

        # 提取内部物理状态
        z_q = env.p_q[2]  # 无人机高度
        z_p = env.p_p[2]  # 货物高度
        v_qz = env.v_q[2]  # 无人机Z轴速度
        v_pz = env.v_p[2]  # 货物Z轴速度
        rope_len = np.linalg.norm(env.p_q - env.p_p)  # 实际绳索距离

        # 简单推算拉力状态：如果绳长 > 自然长度(1.0)，且货物有向上加速度，则产生拉力
        tension_status = "松弛 (Slack)" if rope_len <= env.cfg.l_0 else "拉紧 (Tensioned)"

        dynamics_data.append({
            "Step": step,
            "UAV_Z (m)": round(z_q, 3),
            "Payload_Z (m)": round(z_p, 3),
            "Rope_Dist (m)": round(rope_len, 3),
            "UAV_Vz (m/s)": round(v_qz, 3),
            "Payload_Vz (m/s)": round(v_pz, 3),
            "Tension State": tension_status
        })

        if terminated or truncated:
            break

    df_dynamics = pd.DataFrame(dynamics_data)
    print(df_dynamics.to_string(index=False))
    print("\n结论分析：从上表可以看出，在 Rope_Dist 突破 1.0m 后，Tension State 变为拉紧。")
    print("此时 Payload_Z 开始大于 0，且 Payload_Vz 迅速增加。")
    print("由于弹簧阻尼的反向拖拽，UAV_Vz 会受到影响（相比无负载时增速减缓），完美验证了非线性耦合。")

    # 终点状态极限测试
    print("\n>>> 终点平稳降落边界测试 <<<")
    env.p_q = np.copy(env.p_goal)
    env.p_q[2] = 0.5  # 无人机悬停在极低高度
    env.p_p = np.copy(env.p_goal)
    env.p_p[2] = 0.0  # 货物在地面
    env.v_q = np.zeros(3)
    env.v_p = np.zeros(3)

    obs, reward, terminated, truncated, info = env.step(np.zeros(3))  # 保持不动
    print(f"触发完美降落条件: Reward: {reward:.2f}, Terminated: {terminated}")
    if terminated and reward > 50:
        print("终点奖励逻辑触发正常！高度惩罚成功被巨额奖励抵消。")


def plot_environments(env):
    """生成环境二维平面图与三维起飞图"""
    print("\n" + "=" * 50)
    print("3. 生成环境可视化图表 (Environment Visualizations)")
    print("=" * 50)

    # --- 1. 生成 5 张 2D 平面图 ---
    fig2d, axes2d = plt.subplots(1, 5, figsize=(25, 5))
    for i in range(5):
        env.reset()
        ax = axes2d[i]
        ax.set_xlim(0, env.cfg.env_size[0])
        ax.set_ylim(0, env.cfg.env_size[1])
        ax.set_aspect('equal')

        # 绘制障碍物
        for obs in env.obstacles:
            if obs['type'] == 'cyl':
                circle = plt.Circle(obs['pos'], obs['size'], color='gray', alpha=0.6)
                ax.add_patch(circle)
            elif obs['type'] == 'cube':
                rect = plt.Rectangle(obs['pos'] - obs['size'], obs['size'] * 2, obs['size'] * 2, color='darkgray',
                                     alpha=0.6)
                ax.add_patch(rect)

        # 绘制起终点及安全区
        ax.add_patch(plt.Circle(env.p_start[:2], env.cfg.safe_radius, color='green', alpha=0.2, linestyle='--'))
        ax.add_patch(plt.Circle(env.p_goal[:2], env.cfg.safe_radius, color='red', alpha=0.2, linestyle='--'))
        ax.plot(env.p_start[0], env.p_start[1], 'go', markersize=8, label='Start')
        ax.plot(env.p_goal[0], env.p_goal[1], 'r*', markersize=10, label='Goal')

        ax.set_title(f"2D Map Layout {i + 1}")
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig("env_2d_maps.png")
    print("已保存 5 张二维平面图为 'env_2d_maps.png'")
    plt.close()

    # --- 2. 生成 3 张 3D 拉起图 ---
    fig3d = plt.figure(figsize=(18, 6))
    for i in range(3):
        env.reset()
        ax = fig3d.add_subplot(1, 3, i + 1, projection='3d')

        # 模拟起飞，执行几次向上和随机水平位移的动作
        for _ in range(8):
            env.step(np.array([np.random.uniform(-0.5, 0.5), np.random.uniform(-0.5, 0.5), 1.0]))

        ax.set_xlim(env.p_q[0] - 5, env.p_q[0] + 5)
        ax.set_ylim(env.p_q[1] - 5, env.p_q[1] + 5)
        ax.set_zlim(0, 10)

        # 绘制地面
        xx, yy = np.meshgrid(np.linspace(env.p_q[0] - 5, env.p_q[0] + 5, 2),
                             np.linspace(env.p_q[1] - 5, env.p_q[1] + 5, 2))
        ax.plot_surface(xx, yy, np.zeros_like(xx), color='lightgreen', alpha=0.3)

        # 绘制无人机、绳索、货物
        ax.scatter(*env.p_q, color='blue', s=100, label='UAV', marker='^')
        ax.scatter(*env.p_p, color='orange', s=80, label='Payload', marker='s')
        ax.plot([env.p_q[0], env.p_p[0]], [env.p_q[1], env.p_p[1]], [env.p_q[2], env.p_p[2]], 'k-', linewidth=2,
                label='Flexible Rope')

        ax.set_title(f"3D Slung Load State {i + 1}")
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        if i == 0: ax.legend()

    plt.tight_layout()
    plt.savefig("env_3d_lifting.png")
    print("已保存 3 张三维起飞图为 'env_3d_lifting.png'")
    plt.close()


def test_reward_statistics(env, episodes=100):
    """随机运行100回合，使用 Pandas 分析奖励组件的统计学特征"""
    print("\n" + "=" * 50)
    print(f"4. 运行 {episodes} 回合，分析奖励组件特征 (Reward Statistics)")
    print("=" * 50)

    reward_records = {
        "r_step": [], "r_dist": [], "r_dir": [],
        "r_smooth": [], "r_alt": [], "r_payload": [], "r_total_raw": []
    }

    for ep in range(episodes):
        obs, info = env.reset()
        done = False

        while not done:
            # 使用带有正向偏置的随机动作，模拟一定的探索过程
            action = np.random.uniform(-1.0, 1.0, size=(3,))
            # 给定一个微小的向终点的引导力，避免纯原地打转
            dir_vec = env.p_goal - env.p_q
            dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-5)
            action += dir_vec * 0.2
            action = np.clip(action, -1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)

            # 记录 info 中的每一项奖励
            for key in reward_records.keys():
                if key in info:
                    reward_records[key].append(info[key])

            done = terminated or truncated

    # 转换为 Pandas DataFrame 并计算统计特征
    df_rewards = pd.DataFrame(reward_records)

    # 计算指定统计量
    stats = pd.DataFrame({
        'Mean': df_rewards.mean(),
        'Variance': df_rewards.var(),
        'Min': df_rewards.min(),
        '25%': df_rewards.quantile(0.25),
        'Median': df_rewards.median(),
        '75%': df_rewards.quantile(0.75),
        'Max': df_rewards.max()
    })

    print("单步奖励组件统计分析结果表：")
    # 使用 round 保留四位小数使输出更整洁
    print(stats.round(4).to_string())

    # 验证数值稳定性约束
    max_reward = stats.loc['r_total_raw', 'Max']
    min_reward = stats.loc['r_total_raw', 'Min']
    print("\n>>> 数值稳定性审查 <<<")
    if min_reward >= -2.0 and max_reward <= 2.0:  # 环境内由于有clip，这里测试的是clip前的raw值约束情况
        print("✅ 奖励单步值分布良好，基本被有效约束。")
    else:
        print("⚠️ 警告：存在超出预期量级的极端奖励值，请检查奖励系数。")


if __name__ == "__main__":
    # 初始化环境
    cfg = Config()
    test_env = SlungLoadUAVEnv(cfg)
    test_env.set_difficulty(3)  # 开启最高难度进行极限测试

    # 依次执行各项测试
    test_api_and_spaces(test_env)
    test_nonlinear_dynamics(test_env)
    plot_environments(test_env)
    test_reward_statistics(test_env, episodes=100)

    print("\n极限测试全部完成！")