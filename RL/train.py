import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env import Zelda_Env, game_file, save_file
try:
    from viz_callback import StepRenderCallback, LiveRenderCallback
except Exception:
    from viz_callback import StepRenderCallback
    LiveRenderCallback = None
from stable_baselines3.common.callbacks import CallbackList
from pathlib import Path

# 指定训练显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

TOTAL_STEPS = 1000000

save_file = "/home/crafter_zelda/agent_tiny/RL/game_state/Room_51_saved_no_enemy.state"
game_file = "/home/crafter_zelda/agent_tiny/RL/game_state/Link's awakening.gb"

# 可在此切换子任务:
# None / 'get_key' / 'reach_area' / 'press_button' / 'kill_enemy' / 'explore_tiles'
TASK_NAME = None

# 定义子任务的具体参数
TASK_PARAMS = {
    # 将 (82, 47) 定义为一个 5x5 的区域 [x1, y1, x2, y2]
    "area_xyxy": [80, 45, 84, 49], 
    # 需要在区域内停留 15 帧才算“踩下”
    "dwell_steps": 15,
    # 覆盖默认的 "press_button" 奖励配置
    "reward_weights": {
        # 重新启用距离奖励，引导 Agent 走向按钮
        # (默认的 press_button 配置中 distance_coef 为 0)
        "distance_coef": 0.0001 
    }
}
# 踩下按钮后，立即结束本回合
TERMINATE_ON_SUBTASK = True

TASK_PARAMS = {}
TERMINATE_ON_SUBTASK = False
env = Zelda_Env(game_file=game_file, save_file=save_file,task_name=TASK_NAME,
    task_params=TASK_PARAMS,
    terminate_on_subtask=TERMINATE_ON_SUBTASK)
env = Monitor(env)

"""
# 检查环境是否封装完好
from gymnasium.utils.env_checker import check_env
try:
    check_env(env)
    print("Environment passes all checks!")
except Exception as e:
    print(f"Environment has issues: {e}")
"""

model = PPO(
    "MultiInputPolicy",
    env,
    learning_rate=3e-4,
    n_steps=4096,
    batch_size=512,
    n_epochs=3,
    gamma=0.95,
    gae_lambda=0.65,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    verbose=1,
    tensorboard_log="./ppo_zelda_tensorboard/"
)

output_dir = str((Path(__file__).parent / "videos").resolve())
viz_cb = StepRenderCallback(output_dir=output_dir, save_every_n_episodes=1, fps=15, flush_every_n_steps=1024, verbose=1)
live_cb = LiveRenderCallback() if LiveRenderCallback is not None else None
callbacks = CallbackList([cb for cb in [viz_cb, live_cb] if cb is not None])
model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True, callback=callbacks)
model.save("RL/RL_model/ppo_51_noenemy_final")
env.close()