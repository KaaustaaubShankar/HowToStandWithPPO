import gymnasium as gym
import numpy as np
import os
import mujoco
import time
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from torch import nn

# -------------------------------------------------------------------
# Reward-shaping constants (tune these if you like)
# -------------------------------------------------------------------
KNEE_OK          = 0.7     # rad ≈ 40° — free bend zone
KNEE_PENALTY_W   = 4.0     # weight for bending past the zone
COM_TOL          = 0.12    # m   — COM allowed 12 cm past mid-feet
COM_PENALTY_W    = 6.0
STABILITY_W      = 0.01
POSTURE_W        = 1.0
UPRIGHT_SCALE    = 15.0
ALIVE_BONUS      = 10.0
WARMUP_STEPS     = 400     # delay pushes until agent can stand
PUSH_INTERVAL    = 50
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
        # body & joint IDs we’ll need every step
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "torso")
        self.lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "left_foot")
        self.rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "right_foot")
        self.lknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.rknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        # qpos addresses for knee angles
        self.lknee_qadr = model.jnt_qposadr[self.lknee_id]
        self.rknee_qadr = model.jnt_qposadr[self.rknee_id]

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        obs, _, terminated, truncated, info = self.env.step(action)

        data  = self.env.unwrapped.data
        model = self.env.unwrapped.model

        # ------ Base observations ------
        torso_height = obs[0]
        torso_pitch  = obs[1]
        torso_roll   = obs[2]
        ang_vel      = obs[27:30]

        # ------ Joint angles (knees) ------
        k_l = abs(data.qpos[self.lknee_qadr])  # left knee angle
        k_r = abs(data.qpos[self.rknee_qadr])  # right knee angle

        # ------ COM lateral deviation from feet center ------
        com_xy = data.subtree_com[0][:2]
        feet_xy = 0.5 * (data.xpos[self.lfoot_id][:2] + data.xpos[self.rfoot_id][:2])
        com_xy_dist = np.linalg.norm(com_xy - feet_xy)

        # =======================================================
        #               REWARD COMPONENTS
        # =======================================================

        # --- Standing upright ---
        upright_bonus = np.clip((torso_height - 1.2) * 15.0, 0, 15.0)

        # --- Torso stability ---
        posture_penalty   = 1.0 * (abs(torso_pitch) + abs(torso_roll))
        stability_penalty = 0.01 * np.sum(np.square(ang_vel))

        # --- COM position penalty (if COM drifts past feet center) ---
        com_penalty = 6.0 * max(0.0, com_xy_dist - 0.12)

        # --- Knee shaping ---
        # Allow soft bend (≈ 15–35°), discourage too-straight and too-bent
        overbend = max(0.0, k_l - 0.9) + max(0.0, k_r - 0.9)       # deep squat
        underbend = max(0.0, 0.1 - k_l) + max(0.0, 0.1 - k_r)      # stiff stilting

        bend_reward = 2.0 * ((0.3 <= k_l <= 0.6) + (0.3 <= k_r <= 0.6))  # reward good bend
        knee_penalty = 3.0 * overbend + 1.0 * underbend
        knee_reward = bend_reward - knee_penalty

        # --- Alive bonus ---
        alive_bonus = 10.0 if torso_height > 1.3 else -10.0

        # --- Final reward ---
        reward = (
            alive_bonus
            + upright_bonus
            + knee_reward
            - (posture_penalty + stability_penalty + com_penalty)
        )

        # =======================================================
        #               PERTURBATION LOGIC
        # =======================================================
        if self.perturb and self.step_count > 200 and self.step_count % 50 == 0:
            impulse = np.random.uniform(-150, 150)
            data.xfrc_applied[self.torso_id, 0] = impulse
            print(f"[Step {self.step_count}] Applied push: {impulse:.1f} N")

        return obs, reward, terminated, truncated, info

# -------------------------------------------------------------------
# Training
# -------------------------------------------------------------------
def make_env():
    base = gym.make("Humanoid-v5")
    base = TimeLimit(base, max_episode_steps=MAX_EPISODE_STEPS)
    return UprightHumanoidWrapper(base, perturb=True)

def train():
    log_dir = "./ppo_humanoid_logs/"
    os.makedirs(log_dir, exist_ok=True)

    env = DummyVecEnv([make_env])
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[256, 256, 128],
                       vf=[256, 256, 128])]
    )
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=4096,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.98,
        learning_rate=3e-5,
        clip_range=0.3,
        n_epochs=10,
    )

    print("training …")
    model.learn(total_timesteps=1_000_000)
    model.save("ppo_humanoid_upright")
    print("saved model → ppo_humanoid_upright")

# -------------------------------------------------------------------
# Test
# -------------------------------------------------------------------
def test():
    model = PPO.load("ppo_humanoid_upright")
    env   = UprightHumanoidWrapper(
        TimeLimit(gym.make("Humanoid-v5", render_mode="human"),
                  max_episode_steps=MAX_EPISODE_STEPS),
        perturb=True,
    )

    total_reward, steps = 0.0, 0
    obs, _ = env.reset()
    done = False
    while not done:
        steps += 1
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        time.sleep(0.5)           # slow render
    env.close()
    print(f"Episode reward: {total_reward:.1f}  |  steps: {steps}")

# -------------------------------------------------------------------
if __name__ == "__main__":
    train()       
    test() 
