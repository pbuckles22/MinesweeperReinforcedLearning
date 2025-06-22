# Curriculum Strategy Backup

## Current Dual Curriculum System (Backed Up)

### Overview
This document backs up our current curriculum strategy before implementing the new dual curriculum system focused on achieving human-level win rates.

### Current Curriculum Stages

#### Stage 1: Beginner (4x4, 2 mines)
- **Target Win Rate**: 15%
- **Min Wins Required**: 1
- **Learning-Based Progression**: ✅ Allowed
- **Description**: Learning basic movement and safe cell identification

#### Stage 2: Intermediate (6x6, 4 mines)
- **Target Win Rate**: 12%
- **Min Wins Required**: 1
- **Learning-Based Progression**: ✅ Allowed
- **Description**: Developing pattern recognition and basic strategy

#### Stage 3: Easy (9x9, 10 mines)
- **Target Win Rate**: 10%
- **Min Wins Required**: 2
- **Learning-Based Progression**: ✅ Allowed
- **Description**: Standard easy difficulty, mastering basic gameplay

#### Stage 4: Normal (16x16, 40 mines)
- **Target Win Rate**: 8%
- **Min Wins Required**: 3
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Standard normal difficulty, developing advanced strategies

#### Stage 5: Hard (16x30, 99 mines)
- **Target Win Rate**: 5%
- **Min Wins Required**: 3
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Standard hard difficulty, mastering complex patterns

#### Stage 6: Expert (18x24, 115 mines)
- **Target Win Rate**: 3%
- **Min Wins Required**: 2
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Expert level, handling high mine density

#### Stage 7: Chaotic (20x35, 130 mines)
- **Target Win Rate**: 2%
- **Min Wins Required**: 1
- **Learning-Based Progression**: ❌ Not allowed
- **Description**: Ultimate challenge, maximum complexity

### Current Progression Modes

#### Learning-Based Progression (Default)
- Allows progression based on learning indicators
- Requires positive mean rewards (≥5.0)
- Requires learning progress over time
- Suitable for early exploration and learning

#### Strict Progression (--strict_progression)
- Requires actual win rate achievement
- Requires minimum wins per stage (1-3 wins)
- More realistic but slower progression
- Better for mastery-based learning

### Current Performance Results

#### Learning-Based Mode Results:
- ✅ Reached Stage 7 (Chaotic: 20x35, 130 mines)
- ✅ Positive learning progress throughout
- ✅ Mean rewards in 8-15 range
- ❌ Win rates: 0-34% (inconsistent)

#### Strict Mode Results:
- ❌ Got stuck on Stage 1 (4x4, 2 mines)
- ❌ Required 15% win rate not achieved
- ❌ 0% win rate in evaluation
- ✅ Demonstrated realistic requirements

### Current Issues Identified

1. **Win Rate Discrepancy**: Monitor shows 32-34% win rates, but evaluation shows 0%
2. **Training vs Evaluation Gap**: Different performance between training and evaluation
3. **Human Performance Gap**: Current win rates far below human performance
4. **Progression Logic**: May be too permissive or too strict

### Current Configuration

#### Training Parameters:
- **Total Timesteps**: 1,000,000
- **Evaluation Frequency**: 10,000
- **Evaluation Episodes**: 100
- **Verbose Level**: 0 (optimized)
- **Device**: M1 GPU (MPS)

#### Reward System:
- **Safe Reveal**: +15
- **Mine Hit**: -20
- **Win**: +500
- **Invalid Action**: -1

#### State Representation:
- **4 Channels**: Game state, safety hints, revealed count, progress indicators
- **Action Masking**: Smart masking for obviously bad moves

### Backup Date: December 21, 2024
### Branch: feature/dual-curriculum-system
### Status: Fully functional, needs human performance optimization 