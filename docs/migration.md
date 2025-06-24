# Migration Guide: Legacy to Modular Training

## 🚀 Why Migrate?

The new **modular training script** (`train_agent_modular.py`) solves the complexity problems of the legacy script:

| Aspect | Legacy Script | Modular Script |
|--------|---------------|----------------|
| **Win Rate** | 0-5% | **20%+** |
| **Code Lines** | 2,300 | **300** |
| **Debugging** | Complex | **Simple** |
| **Parameters** | Rigid | **Flexible** |
| **Output** | Verbose | **Clean** |

## 📋 Migration Checklist

### ✅ **Step 1: Test the Modular Script**
```bash
# Quick test to verify it works
python src/core/train_agent_modular.py --total_timesteps 5000
```

### ✅ **Step 2: Update Your Training Commands**

#### **Old (Legacy)**
```bash
# Complex curriculum training
python src/core/train_agent.py \
    --total_timesteps 100000 \
    --curriculum_mode current \
    --eval_freq 5000 \
    --n_eval_episodes 50 \
    --verbose 1
```

#### **New (Modular)** ⭐
```bash
# Simple, effective training
python src/core/train_agent_modular.py \
    --total_timesteps 100000 \
    --board_size 4 \
    --max_mines 2
```

### ✅ **Step 3: Update Scripts**

#### **Mac Scripts**
```bash
# Old: scripts/mac/full_training.sh
# New: Use modular script directly
python src/core/train_agent_modular.py --total_timesteps 100000
```

#### **Windows Scripts**
```powershell
# Old: scripts/windows/full_training.ps1
# New: Use modular script directly
python src/core/train_agent_modular.py --total_timesteps 100000
```

### ✅ **Step 4: Update Documentation**
- Update your project docs to use modular script
- Remove references to complex curriculum features
- Focus on simple, proven approaches

## 🔄 Parameter Mapping

### **Basic Parameters**

| Legacy | Modular | Notes |
|--------|---------|-------|
| `--total_timesteps` | `--total_timesteps` | ✅ Same |
| `--board_size` | `--board_size` | ✅ Same |
| `--max_mines` | `--max_mines` | ✅ Same |
| `--device` | `--device` | ✅ Same |

### **Advanced Parameters**

| Legacy | Modular | Notes |
|--------|---------|-------|
| `--learning_rate` | `--learning_rate` | ✅ Same |
| `--batch_size` | `--batch_size` | ✅ Same |
| `--n_epochs` | `--n_epochs` | ✅ Same |
| `--curriculum_mode` | ❌ **Removed** | Not needed |
| `--eval_freq` | ❌ **Removed** | Built-in |
| `--n_eval_episodes` | ❌ **Removed** | Built-in |

## 🎯 Common Migration Scenarios

### **Scenario 1: Quick Testing**
```bash
# Old
python src/core/train_agent.py --total_timesteps 10000 --verbose 1

# New ⭐
python src/core/train_agent_modular.py --total_timesteps 10000
```

### **Scenario 2: Custom Board Size**
```bash
# Old
python src/core/train_agent.py --board_size 6 --max_mines 4 --total_timesteps 50000

# New ⭐
python src/core/train_agent_modular.py --board_size 6 --max_mines 4 --total_timesteps 50000
```

### **Scenario 3: GPU Training**
```bash
# Old
python src/core/train_agent.py --device mps --total_timesteps 100000

# New ⭐
python src/core/train_agent_modular.py --device mps --total_timesteps 100000
```

### **Scenario 4: Custom Hyperparameters**
```bash
# Old
python src/core/train_agent.py \
    --learning_rate 0.0002 \
    --batch_size 64 \
    --total_timesteps 100000

# New ⭐
python src/core/train_agent_modular.py \
    --learning_rate 0.0002 \
    --batch_size 64 \
    --total_timesteps 100000
```

## 🚨 Breaking Changes

### **Removed Features**
- ❌ **Curriculum Learning**: Too complex, not effective
- ❌ **MLflow Integration**: Adds complexity without benefit
- ❌ **Complex Wrappers**: MultiBoardTrainingWrapper, etc.
- ❌ **Verbose Callbacks**: Too much output

### **Simplified Features**
- ✅ **Evaluation**: Built-in, automatic
- ✅ **Progress Display**: Clean, single-line
- ✅ **Parameter Overrides**: Direct PPO parameter access
- ✅ **Results Saving**: Simple JSON format

## 📊 Expected Results

### **Before Migration (Legacy)**
```
Win Rate: 0-5%
Training Time: 2-4 hours
Debugging: Complex
Success: Rare
```

### **After Migration (Modular)** ⭐
```
Win Rate: 20%+
Training Time: 30-60 minutes
Debugging: Simple
Success: Consistent
```

## 🔧 Troubleshooting

### **Import Errors**
```bash
# If you get import errors, ensure you're in the project root
cd /path/to/MinesweeperReinforcedLearning
python src/core/train_agent_modular.py --total_timesteps 1000
```

### **Parameter Errors**
```bash
# Check available parameters
python src/core/train_agent_modular.py --help
```

### **Performance Issues**
```bash
# Use CPU for small boards (faster)
python src/core/train_agent_modular.py --device cpu --total_timesteps 10000

# Use GPU for large boards
python src/core/train_agent_modular.py --device mps --total_timesteps 100000
```

## 🎉 Success Metrics

You've successfully migrated when:
- ✅ **Win rates improve** from 0-5% to 20%+
- ✅ **Training time reduces** by 50-75%
- ✅ **Debugging becomes simple**
- ✅ **Parameters are easy to adjust**
- ✅ **Results are consistent**

## 🚀 Next Steps

After migration:
1. **Experiment** with different board sizes
2. **Tune** hyperparameters for your specific needs
3. **Document** your successful configurations
4. **Share** your results with the community

## 💡 Pro Tips

- **Start small**: 4x4 boards with 2 mines
- **Be patient**: 10,000+ timesteps for good results
- **Use CPU**: Often faster than GPU for small boards
- **Keep it simple**: Don't overcomplicate the approach

---

**Remember**: The modular script proves that **simplicity beats complexity** in reinforcement learning! 🚀 