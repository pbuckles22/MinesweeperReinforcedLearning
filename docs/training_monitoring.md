# Training Progress Monitoring

This document explains how to monitor training progress and detect when training should be stopped due to lack of improvement.

## 🎯 **Quick Start Options**

### **Option 1: Simple Tail Monitoring (Recommended)**
```bash
# In one terminal: Start training
./scripts/mac/quick_training.sh

# In another terminal: Monitor with tail
tail -f training_stats.txt
```

### **Option 2: Quick Monitor Script**
```bash
# In one terminal: Start training
./scripts/mac/quick_training.sh

# In another terminal: Use quick monitor
./scripts/quick_monitor.sh
```

### **Option 3: Full Monitor Dashboard**
```bash
# In one terminal: Start training
./scripts/mac/quick_training.sh

# In another terminal: Use full monitor
./scripts/monitor_training.sh
```

### **Option 4: Smart Training with Auto-Stop**
```bash
# Automatically stops training if no progress
./scripts/smart_training.sh ./scripts/mac/quick_training.sh
```

## 🔧 **GPU Performance Monitoring**

### **Test GPU Performance Before Training**
```bash
# Check if your GPU is performing optimally
./scripts/test_gpu_performance.sh
```

### **What GPU Monitoring Shows**
- **Device Detection**: M1 GPU (MPS), NVIDIA GPU (CUDA), or CPU
- **Performance Assessment**: Excellent/Good/Fair/Poor based on matrix multiplication speed
- **Training Speed**: Iterations per measurement based on device type
- **Performance Hints**: Device-specific troubleshooting advice

### **Expected Performance Ranges**
| Device | Matrix Mult Speed | Training Speed | Status |
|--------|------------------|----------------|---------|
| M1 GPU | < 0.05s | 3+ iterations | 🚀 Excellent |
| M1 GPU | 0.05-0.1s | 2-3 iterations | ✅ Good |
| M1 GPU | 0.1-0.2s | 1-2 iterations | ⚠️ Fair |
| M1 GPU | > 0.2s | < 1 iteration | 🚨 Poor |
| NVIDIA | < 0.02s | 4+ iterations | 🚀 Excellent |
| NVIDIA | 0.02-0.05s | 3-4 iterations | ✅ Good |
| NVIDIA | 0.05-0.1s | 2-3 iterations | ⚠️ Fair |
| NVIDIA | > 0.1s | < 2 iterations | 🚨 Poor |
| CPU | < 0.1s | 1+ iteration | ✅ Good |
| CPU | 0.1-0.3s | 1 iteration | ⚠️ Fair |
| CPU | > 0.3s | < 1 iteration | 🚨 Slow |

## 📊 **What Gets Monitored**

The training system automatically writes progress to `training_stats.txt` with these metrics:

| Column | Description | Warning Threshold |
|--------|-------------|------------------|
| `timestamp` | Time of measurement | - |
| `iteration` | Training iteration number | - |
| `timesteps` | Total timesteps trained | - |
| `win_rate` | Current win percentage | < 5% after 200 iterations |
| `avg_reward` | Average reward per episode | - |
| `avg_length` | Average episode length | - |
| `stage` | Current curriculum stage | - |
| `phase` | Learning phase (Early Learning, etc.) | - |
| `stage_time` | Time spent in current stage | - |
| `no_improvement` | Iterations without improvement | > 50 iterations |

## 🚨 **Early Termination Triggers**

Training will be automatically stopped if:

1. **No Improvement**: No improvement in win rate or reward for 50+ iterations
2. **Low Win Rate**: Win rate < 5% after 200+ iterations
3. **Poor GPU Performance**: Training speed below expected for device type
4. **Manual Stop**: Press Ctrl+C in monitoring terminal

## 📈 **Expected Progress Patterns**

### **Good Progress (Keep Training):**
- Win rate increasing over time
- Average reward trending upward
- Curriculum stage progression
- Regular improvements every 10-20 iterations
- GPU performing at expected speed

### **Poor Progress (Consider Stopping):**
- Win rate stuck below 5% after 200 iterations
- No improvement for 50+ iterations
- Average reward not increasing
- Stuck in early curriculum stages
- GPU performance below expected range

## 🔧 **Customizing Monitoring**

### **Adjust Smart Training Thresholds:**
Edit `scripts/smart_training.sh`:
```bash
MAX_NO_IMPROVEMENT=50    # Stop after 50 iterations without improvement
MIN_WIN_RATE=5          # Stop if win rate < 5%
MIN_ITERATIONS=200      # Wait 200 iterations before checking win rate
CHECK_INTERVAL=30       # Check every 30 seconds
```

### **Custom One-Liner Monitoring:**
```bash
# Show only win rate and stage
tail -f training_stats.txt | awk -F',' '{print "Win: "$4"% Stage: "$7}'

# Show improvement tracking
tail -f training_stats.txt | awk -F',' '{print "NoImp: "$10" Iter: "$2}'

# Alert on low win rate
tail -f training_stats.txt | awk -F',' '$4 < 5 && $2 > 200 {print "🚨 LOW WIN RATE: "$4"%"}'

# Show GPU performance hints
tail -f training_stats.txt | awk -F',' '$2 > 100 && $4 < 5 {print "💡 Check GPU performance"}'
```

## 📝 **Example Monitoring Session**

```bash
# Terminal 1: Test GPU performance first
./scripts/test_gpu_performance.sh

# Terminal 2: Start training
./scripts/mac/quick_training.sh

# Terminal 3: Monitor progress
./scripts/quick_monitor.sh

# Output example:
# 🔧 Device: ✅ M1 GPU (MPS) - Expected: 2-4x faster than CPU
# [14:30:15] Iter:50 Win:12.5% Reward:8.45 Stage:1 (Early Learning) NoImp:3
# [14:30:25] Iter:51 Win:13.2% Reward:8.67 Stage:1 (Early Learning) NoImp:0
# [14:30:35] Iter:52 Win:14.1% Reward:9.12 Stage:1 (Early Learning) NoImp:0
# ✅ Good progress - win rate increasing, improvements happening
```

## 🎯 **Recommended Workflow**

1. **Test GPU Performance**: `./scripts/test_gpu_performance.sh`
2. **Start with Quick Training**: `./scripts/mac/quick_training.sh`
3. **Monitor with Tail**: `tail -f training_stats.txt`
4. **If Progress is Good**: Continue to medium/full training
5. **If No Progress**: Stop and investigate hyperparameters or GPU performance
6. **For Long Training**: Use smart training wrapper for auto-stop

## 🔍 **Troubleshooting**

### **No Stats File Created:**
- Check if training started properly
- Verify `training_stats.txt` is being written to
- Check for permission issues

### **Stats Not Updating:**
- Training might be stuck or crashed
- Check training process with `ps aux | grep python`
- Restart training if necessary

### **Poor GPU Performance:**
- **M1 Mac**: Check Activity Monitor for thermal throttling
- **NVIDIA**: Check `nvidia-smi` for GPU utilization
- **CPU**: Close other applications to free up resources
- **All**: Ensure good ventilation and cooling

### **False Positives:**
- Adjust thresholds in monitoring scripts
- Some plateaus are normal in RL training
- Consider longer patience for complex stages
- Check if GPU performance is optimal

## 🚀 **Performance Optimization Tips**

### **For M1 Mac Users:**
- Keep Activity Monitor open to watch for thermal throttling
- Ensure good ventilation (don't block air vents)
- Close unnecessary applications during training
- Consider using a laptop stand for better cooling

### **For NVIDIA GPU Users:**
- Monitor GPU utilization with `nvidia-smi`
- Ensure GPU drivers are up to date
- Check for other GPU-intensive applications
- Consider adjusting batch size based on GPU memory

### **For CPU Users:**
- Close other applications to free up CPU cores
- Consider using a cloud GPU service for faster training
- Reduce batch size and model complexity
- Use smaller board sizes for initial testing 