import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from env import SlungLoadUAVEnv, Config
from train import Agent  # 导入我们在 train.py 中定义的网络结构
import warnings

warnings.filterwarnings('ignore')


def render_environment_gif(env, agent, device, episode_idx, save_dir="gifs"):
    """运行一回合，并保存为三视图 GIF 动画"""
    obs, _ = env.reset()

    # 记录轨迹数据
    history_pq = []  # 无人机位置
    history_pp = []  # 货物位置

    done = False
    step_count = 0

    # --- 1. 数据收集阶段 ---
    while not done and step_count < env.cfg.max_steps:
        # 记录当前帧状态
        history_pq.append(np.copy(env.p_q))
        history_pp.append(np.copy(env.p_p))

        obs_tensor = torch.Tensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            # 推理阶段：直接使用输出均值 (确定性策略)，剥离随机探索噪声
            action = agent.actor_mean(obs_tensor)
            action = torch.clamp(action, -1.0, 1.0).cpu().numpy()[0]

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        step_count += 1

    # 把最后一帧也加进去
    history_pq.append(np.copy(env.p_q))
    history_pp.append(np.copy(env.p_p))

    pq_arr = np.array(history_pq)
    pp_arr = np.array(history_pp)

    # --- 2. 图形化渲染阶段 ---
    fig = plt.figure(figsize=(18, 6))
    plt.subplots_adjust(left=0.05, right=0.95, wspace=0.2)
    fig.suptitle(f"Trained Model Evaluation - Episode {episode_idx + 1}", fontsize=16)

    # 子图1：3D 俯仰视角 (全局)
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("3D View (Global)")
    ax1.set_xlim(0, env.cfg.env_size[0])
    ax1.set_ylim(0, env.cfg.env_size[1])
    ax1.set_zlim(0, env.cfg.env_size[2])
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.view_init(elev=30, azim=45)

    # 绘制静态障碍物 (仅绘制一次以提升渲染速度)
    for obs_obj in env.obstacles:
        pos = obs_obj['pos']
        size = obs_obj['size']
        if obs_obj['type'] == 'cyl':
            z = np.linspace(0, env.cfg.env_size[2], 10)
            theta = np.linspace(0, 2 * np.pi, 20)
            theta_grid, z_grid = np.meshgrid(theta, z)
            x_grid = size * np.cos(theta_grid) + pos[0]
            y_grid = size * np.sin(theta_grid) + pos[1]
            ax1.plot_surface(x_grid, y_grid, z_grid, alpha=0.15, color='gray')
        elif obs_obj['type'] == 'cube':
            # 简化的 3D 柱体边界线代替实体面，防止过于遮挡
            half_l = size
            x_b = [pos[0] - half_l, pos[0] + half_l, pos[0] + half_l, pos[0] - half_l, pos[0] - half_l]
            y_b = [pos[1] - half_l, pos[1] - half_l, pos[1] + half_l, pos[1] + half_l, pos[1] - half_l]
            ax1.plot(x_b, y_b, zs=0, zdir='z', color='darkgray', alpha=0.5)
            ax1.plot(x_b, y_b, zs=env.cfg.env_size[2], zdir='z', color='darkgray', alpha=0.5)
            for i in range(4):
                ax1.plot([x_b[i], x_b[i]], [y_b[i], y_b[i]], [0, env.cfg.env_size[2]], color='darkgray', alpha=0.5)

    # 绘制起终点
    ax1.scatter(*env.p_start, color='green', s=100, marker='o', label='Start')
    ax1.scatter(*env.p_goal, color='red', s=150, marker='*', label='Goal')

    # 动态对象：无人机、货物、绳索
    uav_3d, = ax1.plot([], [], [], 'b^', markersize=10, label='UAV')
    payload_3d, = ax1.plot([], [], [], 'rs', markersize=8, label='Payload')
    rope_3d, = ax1.plot([], [], [], 'k-', linewidth=2, label='Rope')
    ax1.legend(loc='upper left')

    # 子图2：XOZ 截面动态跟随视角 (展示摇晃)
    ax2 = fig.add_subplot(132)
    ax2.set_title("XOZ Plane (Dynamic Window: Payload Sway)")
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Z (m)')
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 动态对象：无人机、货物、绳索 (XOZ)
    uav_xoz, = ax2.plot([], [], 'b^', markersize=12)
    payload_xoz, = ax2.plot([], [], 'rs', markersize=10)
    rope_xoz, = ax2.plot([], [], 'k-', linewidth=2)
    # 货物轨迹拖尾
    tail_xoz, = ax2.plot([], [], 'r--', alpha=0.5)

    # 子图3：XOY 俯视全局图 (展示避障与路径)
    ax3 = fig.add_subplot(133)
    ax3.set_title("XOY Plane (Global Trajectory)")
    ax3.set_xlim(0, env.cfg.env_size[0])
    ax3.set_ylim(0, env.cfg.env_size[1])
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Y (m)')
    ax3.set_aspect('equal')
    ax3.grid(True, linestyle='--', alpha=0.3)

    # 绘制 2D 障碍物
    for obs_obj in env.obstacles:
        if obs_obj['type'] == 'cyl':
            circle = plt.Circle(obs_obj['pos'], obs_obj['size'], color='gray', alpha=0.3)
            ax3.add_patch(circle)
        elif obs_obj['type'] == 'cube':
            rect = plt.Rectangle(obs_obj['pos'] - obs_obj['size'], obs_obj['size'] * 2, obs_obj['size'] * 2,
                                 color='darkgray', alpha=0.3)
            ax3.add_patch(rect)

    ax3.plot(env.p_start[0], env.p_start[1], 'go', markersize=10, label='Start')
    ax3.plot(env.p_goal[0], env.p_goal[1], 'r*', markersize=15, label='Goal')

    # 动态对象与轨迹拖尾
    traj_uav_xoy, = ax3.plot([], [], 'b-', alpha=0.6, label='UAV Path')
    traj_payload_xoy, = ax3.plot([], [], 'r--', alpha=0.6, label='Payload Path')
    uav_xoy, = ax3.plot([], [], 'b^', markersize=10)
    payload_xoy, = ax3.plot([], [], 'rs', markersize=8)
    ax3.legend(loc='upper right')

    # --- 3. 动画更新函数 ---
    def update(frame):
        # 取当前帧数据
        q = pq_arr[frame]
        p = pp_arr[frame]

        # 更新 3D 图
        uav_3d.set_data([q[0]], [q[1]])
        uav_3d.set_3d_properties([q[2]])
        payload_3d.set_data([p[0]], [p[1]])
        payload_3d.set_3d_properties([p[2]])
        rope_3d.set_data([q[0], p[0]], [q[1], p[1]])
        rope_3d.set_3d_properties([q[2], p[2]])

        # 更新 XOZ 动态跟随图 (以无人机 X 为中心，横跨 6 米窗口)
        ax2.set_xlim(q[0] - 3.0, q[0] + 3.0)
        ax2.set_ylim(max(0, p[2] - 1.0), q[2] + 2.0)
        uav_xoz.set_data([q[0]], [q[2]])
        payload_xoz.set_data([p[0]], [p[2]])
        rope_xoz.set_data([q[0], p[0]], [q[2], p[2]])
        # 绘制最近 10 步的货物运动拖尾，观察震荡
        start_idx = max(0, frame - 10)
        tail_xoz.set_data(pp_arr[start_idx:frame + 1, 0], pp_arr[start_idx:frame + 1, 2])

        # 更新 XOY 全局图
        traj_uav_xoy.set_data(pq_arr[:frame + 1, 0], pq_arr[:frame + 1, 1])
        traj_payload_xoy.set_data(pp_arr[:frame + 1, 0], pp_arr[:frame + 1, 1])
        uav_xoy.set_data([q[0]], [q[1]])
        payload_xoy.set_data([p[0]], [p[1]])

        return uav_3d, payload_3d, rope_3d, uav_xoz, payload_xoz, rope_xoz, tail_xoz, traj_uav_xoy, traj_payload_xoy, uav_xoy, payload_xoy

    print(f"  > 正在生成动画 (共 {len(pq_arr)} 帧)...", end="", flush=True)

    # interval=100 意味着每帧 100 毫秒，也就是 10 FPS，播放速度较慢，方便观察摇摆
    ani = animation.FuncAnimation(fig, update, frames=len(pq_arr), interval=100, blit=False)

    os.makedirs(save_dir, exist_ok=True)
    gif_path = os.path.join(save_dir, f"test_env_{episode_idx + 1}.gif")

    # 保存为 GIF
    ani.save(gif_path, writer='pillow')
    plt.close(fig)
    print(f" ✅ 已保存至: {gif_path}")


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("🎬 开始 PPO 模型渲染与评估测试")
    print("=" * 50)

    device = torch.device("cpu")  # 渲染过程无需 GPU，CPU 足矣
    env_cfg = Config()
    env = SlungLoadUAVEnv(env_cfg)

    # 强制开启最高难度 (Level 3: 全场景密集障碍物) 进行期末大考
    env.set_difficulty(3)

    # 初始化网络并加载权重
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    agent = Agent(obs_dim, act_dim).to(device)

    # 注意：这里加载你跑出来的最新模型，建议使用 140万步的那个或者 final
    # 如果找不到 final，请替换为 'models/ppo_uav_step_1000000.pth' 等实际存在的文件
    model_path = "models/ppo_uav_final.pth"

    if os.path.exists(model_path):
        agent.load_state_dict(torch.load(model_path, map_location=device))
        agent.eval()  # 设置为评估模式
        print(f"✅ 成功加载模型权重: {model_path}")
    else:
        print(f"❌ 找不到模型权重 {model_path}，请检查路径！(你可能需要将 ppo_uav_step_xxx.pth 重命名)")
        exit()

    num_test_envs = 10
    print(f"\n🚀 开始在 {num_test_envs} 个随机生成的高难度环境中进行飞行测试...\n")

    for i in range(num_test_envs):
        print(f"▶ 正在测试环境 {i + 1}/{num_test_envs}")
        render_environment_gif(env, agent, device, i)

    print("\n🎉 全部评估完成！快去 gifs 文件夹里看看成果吧！")