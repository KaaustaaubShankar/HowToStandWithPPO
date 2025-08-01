import gymnasium as gym
import numpy as np
import os
import time
import mujoco
from torch import nn
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# -------------------------------------------------------------------
# Reward-shaping constants
# -------------------------------------------------------------------
KNEE_OK = 0.7
KNEE_PENALTY_W = 4.0
COM_TOL = 0.12
COM_PENALTY_W = 6.0
STABILITY_W = 0.01
POSTURE_W = 1.0
UPRIGHT_SCALE = 15.0
ALIVE_BONUS = 10.0
WARMUP_STEPS = 400
PUSH_INTERVAL = 50
MAX_EPISODE_STEPS = 1000

# -------------------------------------------------------------------
# Custom wrapper
# -------------------------------------------------------------------
class UprightHumanoidWrapper(gym.Wrapper):
    def __init__(self, env, perturb=True):
        super().__init__(env)
        self.perturb = perturb
        self.step_count = 0

        model = self.env.unwrapped.model
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso")
        self.lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_foot")
        self.rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_foot")
        self.lknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.rknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        self.lknee_qadr = model.jnt_qposadr[self.lknee_id]
        self.rknee_qadr = model.jnt_qposadr[self.rknee_id]

    def reset(self, **kwargs):
        self.step_count = 0
        obs, info = self.env.reset(**kwargs)
        return np.array(obs, dtype=np.float32), info

    def step(self, action):
        self.step_count += 1
        obs, _, terminated, truncated, info = self.env.step(action)

        data = self.env.unwrapped.data
        torso_height = obs[0]
        torso_pitch = obs[1]
        torso_roll = obs[2]
        ang_vel = obs[27:30]

        k_l = abs(data.qpos[self.lknee_qadr])
        k_r = abs(data.qpos[self.rknee_qadr])

        com_xy = data.subtree_com[0][:2]
        feet_xy = 0.5 * (data.xpos[self.lfoot_id][:2] + data.xpos[self.rfoot_id][:2])
        com_xy_dist = np.linalg.norm(com_xy - feet_xy)

        upright_bonus = np.clip((torso_height - 1.2) * UPRIGHT_SCALE, 0, UPRIGHT_SCALE)
        posture_penalty = POSTURE_W * (abs(torso_pitch) + abs(torso_roll))
        stability_penalty = STABILITY_W * np.sum(np.square(ang_vel))
        com_penalty = COM_PENALTY_W * max(0.0, com_xy_dist - COM_TOL)

        overbend = max(0.0, k_l - 0.9) + max(0.0, k_r - 0.9)
        underbend = max(0.0, 0.1 - k_l) + max(0.0, 0.1 - k_r)
        bend_reward = 2.0 * ((0.3 <= k_l <= 0.6) + (0.3 <= k_r <= 0.6))
        knee_penalty = 3.0 * overbend + 1.0 * underbend
        knee_reward = bend_reward - knee_penalty

        alive_bonus = ALIVE_BONUS if torso_height > 1.3 else -ALIVE_BONUS

        reward = (
            alive_bonus
            + upright_bonus
            + knee_reward
            - (posture_penalty + stability_penalty + com_penalty)
        )

        if self.perturb and self.step_count > WARMUP_STEPS and self.step_count % PUSH_INTERVAL == 0:
            impulse = np.random.uniform(-150, 150)
            data.xfrc_applied[self.torso_id, 0] = impulse
            print(f"[Step {self.step_count}] Applied push: {impulse:.1f} N")

        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

# -------------------------------------------------------------------
# Environment Creation
# -------------------------------------------------------------------
def make_env():
    base = gym.make("Humanoid-v5")
    base = TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    return UprightHumanoidWrapper(base, perturb=False)

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def train():
    log_dir = "./sac_humanoid_logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[256, 256, 128]   # shared MLP for actor and critic
    )


    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=3e-4,
        gamma=0.99,
        buffer_size=500_000,
        batch_size=256,
        tau=0.005,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
    )

    print("Training SAC agent…")
    model.learn(total_timesteps=1_000_000)
    model.save("sac_humanoid_upright")
    env.save("sac_humanoid_upright_vecnorm.pkl")
    print("Saved model and VecNormalize stats.")

# -------------------------------------------------------------------
# Testing
# -------------------------------------------------------------------
def test():
    model = SAC.load("sac_humanoid_upright")
    vec_env = DummyVecEnv([make_env])
    vec_env = VecNormalize.load("sac_humanoid_upright_vecnorm.pkl", vec_env)
    vec_env.training = False
    vec_env.norm_reward = False

    obs = vec_env.reset()
    total_reward, steps = 0.0, 0
    done = False

    while not done:
        steps += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = vec_env.step(action)
        total_reward += reward[0]
        if hasattr(vec_env.envs[0], "render"):
            vec_env.envs[0].render()
        time.sleep(0.01)

    print(f"Episode reward: {total_reward:.1f}  |  steps: {steps}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    train()
    test()
