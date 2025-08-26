
# RL_Astar

This project combines Reinforcement Learning (PPO) and the interactive A* (only the single push-pull action is activate here) algorithm for intelligent path planning and solving the push-pull movable obstacles  problem.

## Project Structure
- `env.py`: Environment definition
- `ppo.py`: PPO algorithm implementation
- `train.py`: Training script
- `evaluation.py`: Evaluation script
- `utils.py`: Utility functions
- `results_train.py`: Training result analysis
- `checkpoints/`: Model weights
- `returns.pkl`: Training returns data

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python train.py

# Evaluate the model
python evaluation.py
```

## Dependencies
See `requirements.txt` for details.

## Visualization & Results
Training and evaluation results are saved in `results_train.py` and `returns.pkl`.

---

Feel free to open issues or pull requests for questions or contributions.
