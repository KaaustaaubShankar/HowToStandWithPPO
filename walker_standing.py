import gymnasium as gym
import numpy as np
import os
import mujoco
import math
import time
from gymnasium.wrappers import TimeLimit, RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from torch import nn

# -------------------------------------------------------------------
# Reward-shaping constants (tuned)
# -------------------------------------------------------------------
KNEE_OK          = 0.7     # rad ≈ 40° - free bend zone
KNEE_PENALTY_W   = 4.0     # weight for bending past the zone
COM_TOL          = 0.12    # m - COM allowed 12 cm past mid-feet
COM_PENALTY_W    = 6.0
STABILITY_W      = 0.01
POSTURE_W        = 1.0
UPRIGHT_SCALE    = 15.0    # bonus for maintaining height
ALIVE_BONUS      = 5.0     # per-step survival bonus
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
        # Body & joint IDs
        self.torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "torso")
        self.lfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "left_foot")
        self.rfoot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,  "right_foot")
        self.lknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "left_knee")
        self.rknee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_knee")
        # qpos addresses for knee angles
        self.lknee_qadr = model.jnt_qposadr[self.lknee_id]
        self.rknee_qadr = model.jnt_qposadr[self.rknee_id]
        
    @staticmethod
    def quaternion_to_euler(w, x, y, z):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(t0, t1)
        
        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch = math.asin(t2)
        
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(t3, t4)
        
        return roll, pitch, yaw

    def reset(self, **kwargs):
        self.step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self.step_count += 1
        data = self.env.unwrapped.data
        
        # Reset applied forces
        data.xfrc_applied[self.torso_id, :] = 0
        
        # Apply perturbations
        if self.perturb and self.step_count > WARMUP_STEPS and self.step_count % PUSH_INTERVAL == 0:
            print(f"[Step {self.step_count}] Applying push")
            impulse_x = np.random.uniform(-100, 100)
            impulse_y = np.random.uniform(-100, 100)
            data.xfrc_applied[self.torso_id, 0] = impulse_x
            data.xfrc_applied[self.torso_id, 1] = impulse_y
            # print(f"[Step {self.step_count}] Applied push: ({impulse_x:.1f}, {impulse_y:.1f}) N")

        # Step environment
        obs, _, terminated, truncated, info = self.env.step(action)
        
        # Get accurate state measurements
        torso_height = data.qpos[2]  # Z-coordinate
        torso_quat = data.qpos[3:7]  # Orientation quaternion
        roll, pitch, _ = self.quaternion_to_euler(*torso_quat)
        root_ang_vel = data.qvel[3:6]  # Angular velocity (roll, pitch, yaw)
        
        # Knee penalties
        lknee_angle = data.qpos[self.lknee_qadr]
        rknee_angle = data.qpos[self.rknee_qadr]
        lknee_excess = max(0, abs(lknee_angle) - KNEE_OK)
        rknee_excess = max(0, abs(rknee_angle) - KNEE_OK)
        knee_penalty = KNEE_PENALTY_W * (lknee_excess + rknee_excess)
        
        # COM penalties
        lfoot_pos = data.body(self.lfoot_id).xpos
        rfoot_pos = data.body(self.rfoot_id).xpos
        mid_feet = (lfoot_pos + rfoot_pos) / 2
        mid_feet[2] = 0  # Ignore height
        torso_pos = data.body(self.torso_id).xpos
        torso_horiz = torso_pos.copy()
        torso_horiz[2] = 0
        com_error = np.linalg.norm(torso_horiz - mid_feet)
        com_penalty = COM_PENALTY_W * max(0, com_error - COM_TOL)
        
        # Reward components
        upright_bonus = np.clip((torso_height - 1.2) * UPRIGHT_SCALE, 0, UPRIGHT_SCALE)
        posture_penalty = POSTURE_W * (abs(roll) + abs(pitch))
        stability_penalty = STABILITY_W * np.sum(np.square(root_ang_vel))
        
        reward = (
            ALIVE_BONUS + 
            upright_bonus - 
            posture_penalty - 
            stability_penalty - 
            knee_penalty - 
            com_penalty
        )
        
        return np.array(obs, dtype=np.float32), reward, terminated, truncated, info

# -------------------------------------------------------------------
# Environment creation
# -------------------------------------------------------------------
def make_env(perturb=True, render_mode=None):
    env = gym.make("Humanoid-v5", render_mode=render_mode)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_STEPS)
    return UprightHumanoidWrapper(env, perturb=perturb)

# -------------------------------------------------------------------
# Training with periodic evaluation
# -------------------------------------------------------------------
def train():
    log_dir = "./ppo_humanoid_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Create training environment
    train_env = DummyVecEnv([lambda: make_env(perturb=True)])
    
    # Create evaluation environment (no perturbations)
    eval_env = DummyVecEnv([lambda: make_env(perturb=False)])
    
    policy_kwargs = dict(
        activation_fn=nn.Tanh,
        net_arch=[dict(pi=[512, 512, 512], vf=[512, 512, 512])]
    )
    
    model = PPO(
        "MlpPolicy",
        train_env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=log_dir,
        n_steps=4096,
        batch_size=256,
        gamma=0.995,
        gae_lambda=0.98,
        learning_rate=3e-4,
        clip_range=0.3,
        n_epochs=10,
    )

    # Evaluation callback - runs every 50,000 steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=50000 // train_env.num_envs,  # Account for vectorized env
        deterministic=True,
        render=False,
        n_eval_episodes=5,  # Evaluate across 5 episodes
        verbose=1
    )

    print("Training with periodic evaluations...")
    model.learn(
        total_timesteps=2_000_000,
        callback=eval_callback,
        tb_log_name="PPO"
    )
    
    # Save final model
    model.save(os.path.join(log_dir, "ppo_humanoid_upright_final"))
    print(f"Saved final model → {os.path.join(log_dir, 'ppo_humanoid_upright_final')}")

# -------------------------------------------------------------------
# Test the trained model
# -------------------------------------------------------------------
def test(model_path="ppo_humanoid_logs/best_model",
         save_dir="videos", video_prefix="ppo_humanoid_test"):
    # 1) Load model
    try:
        model = PPO.load(model_path)
        print(f"Loaded model from {model_path}")
    except Exception:
        print(f"Could not load {model_path}, using final model")
        model = PPO.load("ppo_humanoid_logs/ppo_humanoid_upright_final")

    # 2) Create env with rgb frames (needed for recording)
    env = make_env(perturb=True, render_mode="rgb_array")

    # 3) Wrap with RecordVideo
    os.makedirs(save_dir, exist_ok=True)
    env = RecordVideo(
        env,
        video_folder=save_dir,
        name_prefix=video_prefix,
        episode_trigger=lambda ep_id: True,  # record the first episode
    )

    # Optional: set fps if your env metadata doesn't have it
    env.metadata.setdefault("render_fps", 50)

    # 4) Run one episode
    total_reward, steps = 0.0, 0
    obs, _ = env.reset()              # IMPORTANT: reset AFTER wrapping
    done = False
    while not done:
        steps += 1
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        done = terminated or truncated
        # don't sleep; fps is handled by the recorder

    env.close()
    print(f"Episode reward: {total_reward:.1f}  |  steps: {steps}")
    print(f"Saved video to: {os.path.abspath(save_dir)}")


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
if __name__ == "__main__":
    #train()
    test()