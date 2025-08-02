# How to Stand with PPO

Ever thought about how you *stand*? Probably not. That’s exactly why this project exists.
In reinforcement learning (RL), reward engineering is really  difficult, especially for tasks that feel trivial to humans. Standing upright, for instance, involves dynamic balancing, posture, and joint control that’s hard to define in code.

This project trains a humanoid agent using **Proximal Policy Optimization (PPO)** to learn how to stand—and keep standing—even under external disturbances. It uses MuJoCo's Humanoid environment and introduces custom reward shaping and perturbations to help the agent build robust, balanced behavior.

---

## Core Concepts

* **Environment**: `Humanoid-v5` (MuJoCo via Gymnasium)
* **Agent**: Trained with `stable-baselines3` PPO
* **Perturbations**: External pushes during training to test robustness
* **Reward Shaping**: Custom penalties and bonuses for posture, center-of-mass alignment, and knee joint positioning

---

## Project Structure

### `UprightHumanoidWrapper`

Custom `gym.Wrapper` for the MuJoCo humanoid. It:

* Applies random pushes to the torso at intervals
* Computes custom reward:

  * Penalizes excessive knee bending
  * Penalizes poor center-of-mass (COM) alignment
  * Rewards staying upright and balanced

### Reward Function Components

| Component        | Description                         |
| ---------------- | ----------------------------------- |
| `ALIVE_BONUS`    | Constant per-step survival bonus    |
| `UPRIGHT_SCALE`  | Reward for maintaining torso height |
| `KNEE_PENALTY_W` | Penalty for bending knees too much  |
| `COM_PENALTY_W`  | Penalty for center-of-mass drift    |
| `POSTURE_W`      | Penalizes bad roll/pitch posture    |
| `STABILITY_W`    | Penalizes angular velocity          |

---

## Training

To train the agent:

```python
if __name__ == "__main__":
    train()
```

Training setup:

* Total timesteps: **2,000,000**
* PPO architecture: 3 hidden layers with 512 units each
* Evaluation every 50,000 steps
* Logging with TensorBoard (see `./ppo_humanoid_logs/`)

---

## Testing

To test the agent and record a video:

```python
if __name__ == "__main__":
    test()
```

This will:

* Load the best or final PPO model
* Apply periodic torso pushes
* Save a video of the agent trying to stay upright in `./videos/`

---

## Dependencies

Install required packages:

```bash
pip install gymnasium[all] stable-baselines3[extra] mujoco torch numpy
```

---

## File Structure

```plaintext
ppo_standing/
│
├── ppo_humanoid_logs/        # Logs and saved models
├── videos/                   # Test run recordings
├── main.py                   # Full training & testing script
├── README.md                 # You're here!
```

---

## Notes

* Perturbations begin after a short warmup (`WARMUP_STEPS`) to ensure the agent has stabilized first.
* Reward shaping parameters are tuned for balance—feel free to experiment.

---

## Example Result

The agent starts standing still. Then, at regular intervals, random horizontal pushes are applied to test its balance. A well-trained policy will absorb the impact and regain posture—just like a human would.

---

## Demo

🎥 [Watch the demo](https://github.com/KaaustaaubShankar/HowToStandWithPPO/blob/main/videos/unflexible390_124.mp4)


---
