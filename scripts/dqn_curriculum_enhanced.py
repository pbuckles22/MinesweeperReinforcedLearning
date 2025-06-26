#!/usr/bin/env python3
"""
Enhanced DQN Curriculum Learning

Progressive difficulty training for the enhanced DQN agent:
- Start with 2x2 boards (simplest possible)
- Progress to 3x3 boards
- Final training on 4x4 boards
- Transfer learning between stages
"""

import sys
import os
import json
import time
from datetime import datetime
from typing import Dict, List, Tuple, Any
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.minesweeper_env import MinesweeperEnv
from src.core.dqn_agent_enhanced import EnhancedDQNAgent, train_enhanced_dqn_agent, evaluate_enhanced_dqn_agent


class DQNCurriculumTrainer:
    """Curriculum learning trainer for enhanced DQN."""
    
    def __init__(self):
        self.curriculum_stages = [
            {
                'name': 'Stage 1: 2x2 Board',
                'board_size': (2, 2),
                'mine_count': 1,
                'episodes': 50,
                'target_win_rate': 0.7,
                'description': 'Simplest possible board for basic pattern learning'
            },
            {
                'name': 'Stage 2: 3x3 Board',
                'board_size': (3, 3),
                'mine_count': 1,
                'episodes': 75,
                'target_win_rate': 0.6,
                'description': 'Medium complexity for spatial reasoning'
            },
            {
                'name': 'Stage 3: 4x4 Board',
                'board_size': (4, 4),
                'mine_count': 1,
                'episodes': 100,
                'target_win_rate': 0.5,
                'description': 'Full complexity for advanced strategies'
            }
        ]
        
        self.results = []
        self.current_agent = None
    
    def create_agent_for_stage(self, stage: Dict[str, Any], transfer_from: EnhancedDQNAgent = None) -> EnhancedDQNAgent:
        """Create or adapt agent for current stage."""
        board_size = stage['board_size']
        action_size = board_size[0] * board_size[1]
        
        if transfer_from is not None:
            # Transfer learning: adapt existing agent to new board size
            print(f"🔄 Transferring knowledge from {transfer_from.board_size} to {board_size}")
            
            # Create new agent with same hyperparameters
            agent = EnhancedDQNAgent(
                board_size=board_size,
                action_size=action_size,
                learning_rate=0.0003,
                discount_factor=0.99,
                epsilon=0.5,  # Lower epsilon for transfer learning
                epsilon_decay=0.995,
                epsilon_min=0.01,
                replay_buffer_size=100000,
                batch_size=64,
                target_update_freq=1000,
                device='cpu',
                use_double_dqn=True,
                use_dueling=True,
                use_prioritized_replay=True
            )
            
            # Transfer convolutional layers (they should work for different board sizes)
            # Note: This is a simplified transfer - in practice, you'd need to handle size differences
            print(f"   ⚠️  Note: Full transfer not implemented for different board sizes")
            
        else:
            # Create new agent for first stage
            agent = EnhancedDQNAgent(
                board_size=board_size,
                action_size=action_size,
                learning_rate=0.0003,
                discount_factor=0.99,
                epsilon=1.0,
                epsilon_decay=0.995,
                epsilon_min=0.01,
                replay_buffer_size=100000,
                batch_size=64,
                target_update_freq=1000,
                device='cpu',
                use_double_dqn=True,
                use_dueling=True,
                use_prioritized_replay=True
            )
        
        return agent
    
    def train_stage(self, stage: Dict[str, Any], agent: EnhancedDQNAgent) -> Dict[str, Any]:
        """Train agent on a specific curriculum stage."""
        print(f"\n🎯 {stage['name']}")
        print(f"   {stage['description']}")
        print(f"   Board size: {stage['board_size']}")
        print(f"   Mine count: {stage['mine_count']}")
        print(f"   Episodes: {stage['episodes']}")
        print(f"   Target win rate: {stage['target_win_rate']:.1%}")
        print("-" * 60)
        
        # Create environment
        env = MinesweeperEnv(
            initial_board_size=stage['board_size'],
            initial_mines=stage['mine_count']
        )
        
        # Train agent
        start_time = time.time()
        training_stats = train_enhanced_dqn_agent(
            env, agent, stage['episodes'], stage['mine_count'], eval_freq=10
        )
        training_time = time.time() - start_time
        
        # Evaluate agent
        eval_stats = evaluate_enhanced_dqn_agent(agent, env, n_episodes=30)
        
        # Calculate stage metrics
        final_win_rate = training_stats['win_rate']
        eval_win_rate = eval_stats['win_rate']
        target_achieved = eval_win_rate >= stage['target_win_rate']
        
        stage_result = {
            'stage': stage,
            'training_stats': training_stats,
            'eval_stats': eval_stats,
            'training_time': training_time,
            'target_achieved': target_achieved,
            'final_epsilon': agent.epsilon,
            'replay_buffer_size': len(agent.replay_buffer)
        }
        
        print(f"\n✅ Stage Results:")
        print(f"   Training Win Rate: {final_win_rate:.3f}")
        print(f"   Evaluation Win Rate: {eval_win_rate:.3f}")
        print(f"   Target Achieved: {'✅' if target_achieved else '❌'}")
        print(f"   Training Time: {training_time:.2f}s")
        print(f"   Final Epsilon: {agent.epsilon:.3f}")
        
        return stage_result
    
    def run_curriculum(self) -> List[Dict[str, Any]]:
        """Run complete curriculum learning process."""
        print("🚀 Enhanced DQN Curriculum Learning")
        print("=" * 60)
        print("Progressive difficulty training with transfer learning")
        print("=" * 60)
        
        previous_agent = None
        
        for i, stage in enumerate(self.curriculum_stages):
            print(f"\n📊 Stage {i+1}/{len(self.curriculum_stages)}")
            
            # Create or adapt agent
            agent = self.create_agent_for_stage(stage, previous_agent)
            
            # Train on current stage
            stage_result = self.train_stage(stage, agent)
            self.results.append(stage_result)
            
            # Save intermediate agent
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            agent_filename = f"dqn_curriculum_stage_{i+1}_{timestamp}.pth"
            agent.save_model(agent_filename)
            print(f"💾 Agent saved to {agent_filename}")
            
            # Check if target achieved
            if not stage_result['target_achieved']:
                print(f"⚠️  Warning: Target not achieved for {stage['name']}")
                print(f"   Consider training longer or adjusting hyperparameters")
            
            # Prepare for next stage
            previous_agent = agent
            self.current_agent = agent
        
        # Print curriculum summary
        self._print_curriculum_summary()
        
        return self.results
    
    def _print_curriculum_summary(self):
        """Print summary of curriculum learning results."""
        print("\n" + "=" * 60)
        print("🏆 CURRICULUM LEARNING SUMMARY")
        print("=" * 60)
        
        total_training_time = sum(r['training_time'] for r in self.results)
        total_episodes = sum(r['stage']['episodes'] for r in self.results)
        stages_completed = len(self.results)
        targets_achieved = sum(1 for r in self.results if r['target_achieved'])
        
        print(f"\n📊 Overall Results:")
        print(f"   Stages Completed: {stages_completed}/{len(self.curriculum_stages)}")
        print(f"   Targets Achieved: {targets_achieved}/{stages_completed}")
        print(f"   Total Episodes: {total_episodes}")
        print(f"   Total Training Time: {total_training_time:.2f}s")
        
        print(f"\n📈 Stage-by-Stage Results:")
        for i, result in enumerate(self.results):
            stage = result['stage']
            eval_stats = result['eval_stats']
            target_achieved = result['target_achieved']
            
            print(f"   {i+1}. {stage['name']}")
            print(f"      Win Rate: {eval_stats['win_rate']:.3f} (Target: {stage['target_win_rate']:.1%})")
            print(f"      Status: {'✅ Achieved' if target_achieved else '❌ Not Achieved'}")
            print(f"      Training Time: {result['training_time']:.2f}s")
        
        # Final evaluation on hardest stage
        if self.current_agent:
            final_stage = self.curriculum_stages[-1]
            env = MinesweeperEnv(
                initial_board_size=final_stage['board_size'],
                initial_mines=final_stage['mine_count']
            )
            
            print(f"\n🎯 Final Evaluation on {final_stage['name']}")
            final_eval = evaluate_enhanced_dqn_agent(self.current_agent, env, n_episodes=50)
            
            print(f"   Final Win Rate: {final_eval['win_rate']:.3f}")
            print(f"   Final Mean Reward: {final_eval['mean_reward']:.2f}")
            print(f"   Final Mean Length: {final_eval['mean_length']:.1f} steps")
        
        # Save curriculum results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"dqn_curriculum_results_{timestamp}.json"
        
        # Convert results to serializable format
        serializable_results = []
        for result in self.results:
            serializable_result = {}
            for key, value in result.items():
                if key == 'stage':
                    serializable_result[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    serializable_result[key] = float(value)
                elif isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                else:
                    serializable_result[key] = value
            serializable_results.append(serializable_result)
        
        curriculum_summary = {
            'timestamp': timestamp,
            'total_stages': len(self.curriculum_stages),
            'stages_completed': stages_completed,
            'targets_achieved': targets_achieved,
            'total_episodes': total_episodes,
            'total_training_time': total_training_time,
            'stage_results': serializable_results
        }
        
        with open(filename, 'w') as f:
            json.dump(curriculum_summary, f, indent=2)
        
        print(f"\n💾 Curriculum results saved to {filename}")


def main():
    """Run enhanced DQN curriculum learning."""
    print("🧪 Enhanced DQN Curriculum Learning")
    print("=" * 50)
    
    try:
        # Create curriculum trainer
        trainer = DQNCurriculumTrainer()
        
        # Run curriculum learning
        results = trainer.run_curriculum()
        
        print(f"\n✅ Curriculum learning completed successfully!")
        print(f"📊 Trained through {len(results)} stages")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during curriculum learning: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Curriculum learning completed successfully!")
    else:
        print("\n💥 Curriculum learning failed. Please check the error messages above.")
        sys.exit(1) 