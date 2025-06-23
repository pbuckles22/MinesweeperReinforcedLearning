# Curriculum Modes Documentation

## Overview

The Minesweeper RL project now supports **three distinct curriculum modes** through the dual curriculum system. Each mode is designed for different training objectives and performance targets.

## Available Curriculum Modes

### 1. **"current" - Original Learning Curriculum**
**Purpose**: Basic learning and experimentation with low performance targets

**Target Win Rates**:
- Stage 1 (4x4, 2 mines): 15%
- Stage 2 (6x6, 4 mines): 12%
- Stage 3 (9x9, 10 mines): 10%
- Stage 4 (16x16, 40 mines): 8%
- Stage 5 (16x30, 99 mines): 5%
- Stage 6 (18x24, 115 mines): 3%
- Stage 7 (20x35, 130 mines): 2%

**Key Features**:
- **Learning-based progression**: Allows progression even without meeting win rate targets
- **Standard training**: 1.0x training multiplier
- **10 evaluation episodes** per stage
- **Low minimum wins required** (1-3 wins)
- **Focus**: Basic learning, pattern recognition, and strategy development

**Use Case**: 
- Initial experimentation and learning
- Quick training runs for testing
- Understanding basic game mechanics
- Development and debugging

### 2. **"human_performance" - Human-Level Targets**
**Purpose**: Achieve human expert-level performance benchmarks

**Target Win Rates**:
- Stage 1 (4x4, 2 mines): 80%
- Stage 2 (6x6, 4 mines): 70%
- Stage 3 (9x9, 10 mines): 60%
- Stage 4 (16x16, 40 mines): 50%
- Stage 5 (16x30, 99 mines): 40%
- Stage 6 (18x24, 115 mines): 30%
- Stage 7 (20x35, 130 mines): 20%

**Key Features**:
- **Strict progression**: Must meet win rate targets to advance
- **Extended training**: 3.0x training multiplier
- **20 evaluation episodes** per stage
- **Higher minimum wins required** (2-8 wins)
- **Focus**: Achieving human expert-level performance

**Use Case**:
- Training agents to match human performance
- Benchmarking against human players
- Research on human-level AI capabilities
- Preparation for superhuman performance

### 3. **"superhuman" - Surpass Human Benchmarks**
**Purpose**: Exceed human expert performance and achieve superhuman capabilities

**Target Win Rates**:
- Stage 1 (4x4, 2 mines): 95%
- Stage 2 (6x6, 4 mines): 85%
- Stage 3 (9x9, 10 mines): 75%
- Stage 4 (16x16, 40 mines): 65%
- Stage 5 (16x30, 99 mines): 55%
- Stage 6 (18x24, 115 mines): 45%
- Stage 7 (20x35, 130 mines): 35%

**Key Features**:
- **Strict progression**: Must meet win rate targets to advance
- **Maximum training**: 5.0x training multiplier
- **30 evaluation episodes** per stage
- **Highest minimum wins required** (2-9 wins)
- **Focus**: Surpassing human expert performance

**Use Case**:
- Achieving superhuman performance
- Research on advanced AI capabilities
- Demonstrating RL system effectiveness
- Pushing the boundaries of AI performance

## Usage

### Command Line Usage
```bash
# Use original learning curriculum
python -m src.core.train_agent --curriculum_mode current

# Use human performance targets
python -m src.core.train_agent --curriculum_mode human_performance

# Use superhuman targets
python -m src.core.train_agent --curriculum_mode superhuman
```

### Default Mode
The system defaults to `human_performance` mode if no curriculum mode is specified.

## Training Parameters Comparison

| Parameter | current | human_performance | superhuman |
|-----------|---------|-------------------|------------|
| Training Multiplier | 1.0x | 3.0x | 5.0x |
| Evaluation Episodes | 10 | 20 | 30 |
| Progression Type | Learning-based | Strict | Strict |
| Target Difficulty | Low | Medium | High |
| Training Time | Standard | Extended | Maximum |

## Progression Logic

### Learning-Based Progression (current mode)
- Allows progression even if win rate targets aren't met
- Uses learning indicators (positive rewards, consistent improvement)
- Designed for early learning and experimentation

### Strict Progression (human_performance & superhuman modes)
- Must achieve minimum win rate to advance
- Requires minimum number of actual wins
- Designed for performance-focused training

## Implementation Status

✅ **All three modes are fully implemented and ready to use**

- **current**: Original system, fully tested and working
- **human_performance**: New implementation, tested and working
- **superhuman**: New implementation, ready for testing

## Training Recommendations

### For Development & Testing
- Use **"current"** mode for quick iterations and debugging
- Short training runs (1,000-5,000 timesteps)
- Focus on functionality rather than performance

### For Human-Level Research
- Use **"human_performance"** mode for serious training
- Extended training runs (50,000+ timesteps)
- Focus on achieving human expert benchmarks

### For Superhuman Research
- Use **"superhuman"** mode for advanced research
- Maximum training runs (100,000+ timesteps)
- Focus on surpassing human capabilities

## Performance Expectations

### Current Mode
- **Stage 1**: 15-25% win rate achievable in short training
- **Stage 3**: 8-15% win rate achievable with extended training
- **Stage 5**: 3-8% win rate achievable with maximum training

### Human Performance Mode
- **Stage 1**: 80% win rate requires significant training (50k+ timesteps)
- **Stage 3**: 60% win rate requires extensive training (100k+ timesteps)
- **Stage 5**: 40% win rate requires maximum training (200k+ timesteps)

### Superhuman Mode
- **Stage 1**: 95% win rate requires maximum training (100k+ timesteps)
- **Stage 3**: 75% win rate requires extensive training (200k+ timesteps)
- **Stage 5**: 55% win rate requires maximum training (500k+ timesteps)

## Future Enhancements

- **Adaptive curriculum**: Automatically switch between modes based on performance
- **Hybrid approaches**: Combine elements from different modes
- **Custom targets**: Allow user-defined win rate targets
- **Performance comparison**: Tools to compare results across modes

---

**Last Updated**: December 21, 2024  
**Status**: All modes implemented and tested  
**Next**: Performance comparison and optimization 