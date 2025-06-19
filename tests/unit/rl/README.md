# RL Test Suite

This directory contains tests for RL agent integration, early learning mode, and curriculum features.
 
- **Non-determinism is expected:** These tests do not assert specific board states or outcomes, only valid behaviors and API compliance.
- **Purpose:** Ensure the environment is robust for RL training, exploration, and curriculum learning.
- **Do not add deterministic/edge-case tests here:** Those belong in `tests/unit/core/`. 