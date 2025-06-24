# Performance Optimization TODO

## GPU vs CPU Performance Issues

### Current Problem
- Main training script (`train_agent.py`) automatically detects and uses M1 GPU (MPS)
- Our benchmark showed CPU is 10x faster than M1 GPU for PPO with MlpPolicy
- Environment variables don't effectively override GPU detection
- Training runs at ~200 steps/sec (GPU speed) instead of ~1200+ steps/sec (CPU speed)

### Temporary Fix Applied
- Added `FORCE_CPU=1` environment variable to bypass GPU detection
- Created `scripts/mac/curriculum_training_cpu.sh` to force CPU usage
- Modified `detect_optimal_device()` to check for `FORCE_CPU` environment variable

### TODO: Future Improvements

#### 1. Smart Device Selection Based on Board Size
- [ ] Implement board-size-based device selection in `detect_optimal_device()`
- [ ] Use CPU for small boards (4x4, 6x6, 8x8) where it's faster
- [ ] Use GPU for larger boards (16x16+) where it might be beneficial
- [ ] Add configuration option to override automatic selection

#### 2. Benchmark-Integrated Device Selection
- [ ] Run quick performance benchmark during startup
- [ ] Compare CPU vs GPU performance for current board size
- [ ] Automatically select faster device based on benchmark results
- [ ] Cache benchmark results to avoid repeated testing

#### 3. Policy-Based Device Selection
- [ ] Detect policy type (MlpPolicy vs CNN policy)
- [ ] Use CPU for MlpPolicy (as recommended by SB3)
- [ ] Use GPU for CNN policies where it's beneficial
- [ ] Add warning when using GPU with MlpPolicy

#### 4. Configuration File Support
- [ ] Add `config.yaml` or similar for device preferences
- [ ] Allow users to specify preferred device per board size
- [ ] Support for different devices per curriculum stage
- [ ] Environment variable overrides for all settings

#### 5. Performance Monitoring
- [ ] Add real-time performance monitoring during training
- [ ] Track steps/sec and automatically switch devices if performance degrades
- [ ] Log device performance metrics for analysis
- [ ] Alert when suboptimal device is being used

#### 6. Cross-Platform Optimization
- [ ] Test on different hardware (Intel Mac, Windows, Linux)
- [ ] Optimize device selection for CUDA GPUs
- [ ] Handle different GPU architectures (M1, M2, NVIDIA, AMD)
- [ ] Create platform-specific optimization profiles

### Current Workaround
Use the CPU-forced script for now:
```bash
./scripts/mac/curriculum_training_cpu.sh
```

### References
- [SB3 Issue #1245](https://github.com/DLR-RM/stable-baselines3/issues/1245) - GPU not recommended for MlpPolicy
- Benchmark results: CPU ~1200-1300 steps/sec vs M1 GPU ~119 steps/sec
- Conservative training achieved 30% win rate on CPU

### Priority
- **High**: Fix device selection for immediate performance improvement
- **Medium**: Add smart device selection based on board size
- **Low**: Comprehensive performance monitoring and optimization 