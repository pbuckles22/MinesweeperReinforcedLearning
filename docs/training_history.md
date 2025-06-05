# Training History Log

This document tracks successful training runs, their configurations, and results. Use this as a reference to reproduce successful training runs or understand what configurations work best.

## Format
Each entry follows this structure:
```
### Run [DATE]_[TIME]
- **Board Config**: [size]x[size] with [mines] mines ([density]% density)
- **Win Rate**: [X]%
- **Training Steps**: [N]
- **Key Parameters**:
  - Learning Rate: [X]
  - Batch Size: [X]
  - Steps per Update: [X]
  - Network Architecture: [X]
  - Entropy Coefficient: [X]
- **Environment Settings**:
  - Curriculum Mode: [Yes/No]
  - Early Learning Assistance: [Yes/No]
  - Reward Structure: [Standard/Custom]
- **Notes**: [Any relevant observations or special conditions]
```

## Successful Runs

### Run 20250603_164515
- **Board Config**: 8x8 with 12 mines (18.75% density)
- **Win Rate**: 52%
- **Training Steps**: 150,000
- **Key Parameters**:
  - Learning Rate: 0.0001
  - Batch Size: 64
  - Steps per Update: 2048
  - Network Architecture: [256, 256] for both policy and value
  - Entropy Coefficient: 0.01
- **Environment Settings**:
  - Curriculum Mode: Yes
  - Early Learning Assistance: Yes (first 100 games)
  - Reward Structure: Standard
- **Notes**: 
  - Started with 4x4 board and 2 mines
  - Smooth curriculum progression
  - Stable learning curve
  - First move always safe

## Configuration Templates

### High Performance Template
Use this configuration for maximum performance:
```python
board_size = 8
max_mines = 12
learning_rate = 0.0001
batch_size = 64
n_steps = 2048
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.01
vf_coef = 0.5
max_grad_norm = 0.5
policy_kwargs = {
    "net_arch": [
        {
            "pi": [256, 256],
            "vf": [256, 256]
        }
    ]
}
```

### Quick Learning Template
Use this configuration for faster initial learning:
```python
board_size = 4
max_mines = 2
learning_rate = 0.0003
batch_size = 32
n_steps = 1024
n_epochs = 10
gamma = 0.99
gae_lambda = 0.95
clip_range = 0.2
ent_coef = 0.02
vf_coef = 0.5
max_grad_norm = 0.5
policy_kwargs = {
    "net_arch": [
        {
            "pi": [128, 128],
            "vf": [128, 128]
        }
    ]
}
```

## Failed Runs Analysis

### Common Failure Patterns
1. **Early Collapse**
   - Symptoms: Win rate drops to 0% and stays there
   - Likely Causes: 
     - Learning rate too high
     - Network too small
     - Insufficient exploration

2. **Unstable Learning**
   - Symptoms: Win rate fluctuates wildly
   - Likely Causes:
     - Batch size too small
     - Entropy coefficient too high
     - Reward structure too extreme

3. **Plateau**
   - Symptoms: Win rate stops improving
   - Likely Causes:
     - Learning rate too low
     - Network capacity insufficient
     - Curriculum progression too fast

## Best Practices

1. **Starting New Training**
   - Begin with the Quick Learning Template
   - Use curriculum learning
   - Enable early learning assistance
   - Start with smaller board size

2. **Monitoring Progress**
   - Check win rate every 10k steps
   - Look for steady improvement
   - Watch for signs of early collapse

3. **Adjusting Parameters**
   - If learning is too slow: increase learning rate
   - If unstable: increase batch size
   - If plateauing: increase network size
   - If overfitting: increase entropy coefficient

## Notes
- Always document any changes from the templates
- Record both successful and failed runs
- Note any special conditions or modifications
- Update this document after each significant training run 