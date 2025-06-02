import unittest
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from minesweeper_env import MinesweeperEnv

class TestTrainAgent(unittest.TestCase):
    def setUp(self):
        """Set up test environment and directories."""
        self.log_dir = "test_logs/"
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Create a small test environment
        def make_test_env():
            env = MinesweeperEnv(board_size=3, num_mines=1)  # Small board for testing
            env = Monitor(env, self.log_dir)
            return env
        
        self.env = DummyVecEnv([make_test_env])

    def tearDown(self):
        """Clean up after tests."""
        # Close the environment
        self.env.close()
        
        # Remove test logs directory
        if os.path.exists(self.log_dir):
            try:
                for file in os.listdir(self.log_dir):
                    file_path = os.path.join(self.log_dir, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(self.log_dir)
            except Exception as e:
                print(f"Warning: Could not clean up test directory: {e}")

    def test_environment_creation(self):
        """Test that the environment is created correctly."""
        # Test environment dimensions
        obs = self.env.reset()
        self.assertEqual(obs.shape[1:], (3, 3))  # 3x3 board
        
        # Test that environment is properly wrapped
        self.assertIsInstance(self.env.envs[0], Monitor)
        self.assertIsInstance(self.env.envs[0].env, MinesweeperEnv)

    def test_model_initialization(self):
        """Test that the PPO model initializes correctly."""
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_rate=0.0003,
            n_steps=128,  # Smaller for testing
            batch_size=32,
            n_epochs=2,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            tensorboard_log=self.log_dir
        )
        
        # Test model attributes
        self.assertEqual(model.learning_rate, 0.0003)
        self.assertEqual(model.n_steps, 128)
        self.assertEqual(model.batch_size, 32)
        self.assertEqual(model.n_epochs, 2)

    def test_model_prediction(self):
        """Test that the model can make predictions."""
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=128,
            batch_size=32,
            n_epochs=2
        )
        
        # Test prediction
        obs = self.env.reset()
        action, _states = model.predict(obs, deterministic=True)
        
        # Check action is within valid range
        self.assertGreaterEqual(action[0], 0)
        self.assertLess(action[0], 18)  # 3x3 board * 2 actions (reveal/flag)

    def test_training_step(self):
        """Test that the model can perform a single training step and save a model file."""
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=128,
            batch_size=32,
            n_epochs=2
        )
        
        # Train for a few steps
        model.learn(total_timesteps=256, progress_bar=False)
        # Save the model
        model_path = os.path.join(self.log_dir, "test_model")
        model.save(model_path)
        # Check that model file was created
        self.assertTrue(os.path.exists(model_path + ".zip"))

    def test_model_saving_loading(self):
        """Test that the model can be saved and loaded."""
        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=128,
            batch_size=32,
            n_epochs=2
        )
        
        # Save model
        model_path = os.path.join(self.log_dir, "test_model")
        model.save(model_path)
        
        # Load model
        loaded_model = PPO.load(model_path)
        
        # Compare predictions
        obs = self.env.reset()
        action1, _ = model.predict(obs, deterministic=True)
        action2, _ = loaded_model.predict(obs, deterministic=True)
        
        self.assertEqual(action1[0], action2[0])

    def test_environment_interaction(self):
        """Test that the environment responds correctly to actions."""
        obs = self.env.reset()
        
        # Take a random action
        action = np.array([0])  # First cell
        obs, reward, done, info = self.env.step(action)
        
        # Check observation shape
        self.assertEqual(obs.shape[1:], (3, 3))
        # Check reward is a number (including numpy types)
        self.assertTrue(np.issubdtype(type(reward[0]), np.number))
        # Check done is boolean or numpy bool
        self.assertTrue(isinstance(done[0], (bool, np.bool_)))

if __name__ == '__main__':
    unittest.main() 