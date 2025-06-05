from gymnasium.vector import SyncVectorEnv

# Export SyncVectorEnv as DummyVecEnv for backward compatibility
DummyVecEnv = SyncVectorEnv 