# RL Early Learning Debugging Checklist

## 1. Reward Signal & Environment Setup
- [x] **Verify reward shaping:** Ensure positive rewards for safe moves and strong penalties for hitting mines.
- [x] **Check win reward:** Confirm that winning gives a significant positive reward.
- [x] **Check mine penalty:** Confirm that hitting a mine gives a strong negative reward.
- [x] **Reward for progress:** Consider small positive rewards for revealing safe cells or making progress.

## 2. Training Configuration
- [x] **Increase training timesteps:** Try running for more timesteps (e.g., 50,000 or 100,000) to allow learning.
- [x] **Batch size and learning rate:** Experiment with batch size and learning rate for stability.
- [x] **Simplify environment:** Start with the smallest board and fewest mines possible.

## 3. Environment Randomness & Difficulty
- [x] **Evaluation disconnect:** Fixed - environment was using hardcoded reward values instead of constants
  - **Issue:** `make_env()` function had hardcoded values that overrode the updated constants
  - **Fix:** Updated `make_env()` to use `REWARD_INVALID_ACTION`, `REWARD_HIT_MINE`, `REWARD_SAFE_REVEAL`, `REWARD_WIN`
  - **Result:** Training and evaluation now use consistent reward values
- [ ] **Check environment randomness:** Ensure the environment is properly randomized.
- [ ] **Verify difficulty progression:** Make sure the curriculum is working correctly.

## 4. Monitoring & Visualization
- [ ] **Use MLflow:** Visualize reward, loss, and win rate curves to spot learning trends.
- [ ] **Check logs:** Review logs for signs of improvement or stagnation.

## 5. Model & Algorithm
- [ ] **Network architecture:** Try a simpler or slightly larger network if learning is too slow.
- [ ] **Algorithm parameters:** Tune PPO or other RL algorithm parameters (entropy, gamma, etc.).

## 6. Sanity Checks
- [ ] **Test random agent:** Compare to a random agent's performance to ensure learning is possible.
- [ ] **Manual play:** Play the environment yourself to confirm it's winnable and rewards make sense. 