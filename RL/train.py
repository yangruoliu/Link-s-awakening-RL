import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from env import Zelda_Env, game_file, save_file

# 指定训练显卡
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7"

TOTAL_STEPS = 1000000

save_file = "Zelda-Link-s-awakening-agent copy/RL/game_state/Room_51.state"
game_file = "Zelda-Link-s-awakening-agent copy/RL/game_state/Link's awakening.gb"

env = Zelda_Env(game_file=game_file, save_file=save_file)
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

model.learn(total_timesteps=TOTAL_STEPS, progress_bar=True)
model.save("RL/RL_model/ppo_51_final")
env.close()