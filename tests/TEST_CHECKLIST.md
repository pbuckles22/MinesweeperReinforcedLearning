# Minesweeper Test Checklist (Updated)

## Priority 1: Core State Management & Mechanics
- [x] Safe cell reveal: `assert not terminated` fails
- [x] Safe cell cascade: Not explicitly listed, but likely related to state update issues
- [ ] Safe cell adjacent mines: `assert np.int64(9) == 0`
- [ ] Mine placement: `assert mine_count == self.env.current_mines`
- [ ] Board initialization: `assert np.all(env.board == CELL_UNREVEALED)`
- [ ] Flag placement on mine/safe cell: `assert state[1, 1] == CELL_FLAGGED`
- [ ] Flag removal: `assert state[1, 1] == CELL_UNREVEALED`
- [ ] Flag count in info: `KeyError: 'flags_remaining'`
- [ ] Flag on revealed cell: `assert not terminated`
- [ ] Reveal flagged cell: `assert not terminated`
- [ ] Reveal already revealed cell: `assert not terminated`
- [ ] State transitions: `assert state[y, x] == CELL_UNREVEALED`
- [ ] State representation: `assert state[y2, x2] == CELL_UNREVEALED`

## Priority 2: Game Logic & Win/Loss
- [ ] Mine hit termination: `assert terminated`
- [ ] Mine hit state update: `assert state[1, 1] == CELL_MINE_HIT`
- [ ] Mine hit reward breakdown: `assert terminated`
- [ ] First move mine hit reset: `assert reward == 0`
- [ ] First move behavior: `assert reward == 0`
- [ ] Win condition: `assert reward > 0`
- [ ] Game over condition: `KeyError: 'won'`
- [ ] Win condition (rectangular): `KeyError: 'won'`
- [ ] Curriculum progression: `assert (3 > 3 or 3 > 3 or 1 > 1)`
- [ ] Difficulty levels: `assert 96 == 99`
- [ ] Rectangular board actions: `assert not terminated`
- [ ] Board boundary actions: `assert not terminated`

## Priority 3: Initialization & Parameter Validation
- [ ] Invalid board size: Regex mismatch (`'Board size must be positive'` vs `'Board dimensions must be positive'`)
- [ ] Invalid mine count: Regex mismatch (`'Mine count cannot exceed board size squared'` vs `'Mine count cannot exceed board area'`)
- [ ] Invalid initial parameters: Regex mismatch (`'Initial board size cannot exceed max board size'` vs `'Mine count cannot exceed board area'`)
- [ ] Invalid reward parameters: Did not raise ValueError

## Priority 4: Action Space & Masking
- [ ] Action space boundaries: `assert np.int64(32) != np.int64(32)`
- [ ] Action space mapping: Not explicitly listed, but related to above
- [ ] Invalid action handling: Did not raise ValueError/IndexError
- [ ] Reveal after game over: `assert 'won' in info`
- [ ] Flag already flagged cell: `assert 'won' in {}`

## Priority 5: Regression (Previously Passing, Now Failing)
- [ ] None detected in this run

## Priority 6: Cleared Issues (Now Passing)
- [x] Mine placement method called in reset
- [x] Board and state arrays initialized with CELL_UNREVEALED
- [x] Action space and observation space updated on reset

## Priority 7: Core Functionality (Already Implemented)
### Environment Initialization
- [x] Test with default parameters
- [x] Test with custom parameters
- [x] Test board dimensions
- [x] Test mine placement
- [x] Test adjacent mine counting
- [x] Test board state initialization

### Basic Game Mechanics
- [x] Test safe cell reveal
- [x] Test safe cell cascade
- [x] Test mine hits
- [x] Test adjacent mine counting
- [x] Test state updates

## Priority 8: Game Rules and Actions
### Action Tests
- [ ] Test reveal action
- [ ] Test flag action
- [ ] Test unflag action
- [ ] Test invalid actions
- [ ] Test actions on rectangular boards
- [ ] Test actions at board boundaries

### Win/Loss Conditions
- [ ] Test game over condition
- [ ] Test win condition
- [ ] Test loss condition
- [ ] Test state transitions
- [ ] Test state representation

## Priority 9: Difficulty Levels
### Board Size Tests
- [ ] Test Easy (9x9, 10 mines)
- [ ] Test Normal (16x16, 40 mines)
- [ ] Test Hard (16x30, 99 mines)
- [ ] Test Expert (18x24, 115 mines)
- [ ] Test Chaotic (20x35, 130 mines)

### Rectangular Board Tests
- [ ] Test 16x30 board
- [ ] Test 18x24 board
- [ ] Test 20x35 board
- [ ] Test board layouts
- [ ] Test mine placement

## Priority 10: Reward System
### Reward Tests
- [ ] Test first move reward
- [ ] Test reveal reward
- [ ] Test flag reward
- [ ] Test unflag reward
- [ ] Test win reward
- [ ] Test loss penalty
- [ ] Test reward scaling with board size

## Priority 11: Curriculum Learning
### Progression Tests
- [ ] Test progression through difficulty levels
- [ ] Test win rate thresholds
- [ ] Test board size progression
- [ ] Test mine count progression
- [ ] Test stage transitions
- [ ] Test performance metrics

## Priority 12: Edge Cases
### Boundary Tests
- [ ] Test minimum board size
- [ ] Test maximum board size
- [ ] Test minimum mine count
- [ ] Test maximum mine count
- [ ] Test invalid configurations
- [ ] Test boundary conditions

## Priority 13: Performance
### Performance Tests
- [ ] Test large board performance
- [ ] Test many mines performance
- [ ] Test action speed
- [ ] Test state updates
- [ ] Test memory usage
- [ ] Test CPU usage

## Priority 14: Integration
### Integration Tests
- [ ] Test with different agents
- [ ] Test with different learning algorithms
- [ ] Test with different reward structures
- [ ] Test with different observation spaces
- [ ] Test with different action spaces
- [ ] Test with different training configurations

## Priority 15: Rendering
### Rendering Tests
- [ ] Test board visualization
- [ ] Test state visualization
- [ ] Test flag visualization
- [ ] Test rectangular board rendering
- [ ] Test different color schemes
- [ ] Test display scaling

## Priority 16: Early Learning Mode
### Early Learning Tests
- [ ] Test early learning mode initialization
- [ ] Test corner safety
- [ ] Test edge safety
- [ ] Test initial board size
- [ ] Test transition out of early learning
- [ ] Test threshold behavior
- [ ] Test parameter updates
- [ ] Test state preservation

## Priority 17: State Management
### State Tests
- [ ] Test state reset
- [ ] Test mine placement on reset
- [ ] Test flag clearing
- [ ] Test counter reset
- [ ] Test state persistence between actions
- [ ] Test flag persistence
- [ ] Test revealed cell persistence
- [ ] Test game over state
- [ ] Test game counter
- [ ] Test win counter
- [ ] Test consecutive hits
- [ ] Test win rate calculation

## Priority 18: Error Handling
### Error Tests
- [ ] Test invalid board size errors
- [ ] Test invalid mine count errors
- [ ] Test invalid spacing errors
- [ ] Test invalid threshold errors
- [ ] Test invalid reward errors
- [ ] Test error message clarity
- [ ] Test error recovery

## Priority 19: Functional Tests
### Game Flow Tests
- [ ] Test complete game win scenario
- [ ] Test complete game loss scenario
- [ ] Test game with flags
- [ ] Test game with wrong flags

### Difficulty Progression Tests
- [ ] Test early learning progression
- [ ] Test different difficulty levels
- [ ] Test curriculum limits
- [ ] Test win rate tracking

### Performance Tests
- [ ] Test large board performance
- [ ] Test many mines performance
- [ ] Test memory usage
- [ ] Test reset performance
- [ ] Test state update performance
- [ ] Test rapid actions

## Priority 20: Script Tests
### Installation Script Tests
- [ ] Test script existence
- [ ] Test script permissions
- [ ] Test script syntax
- [ ] Test script dependencies
- [ ] Test environment setup

### Run Script Tests
- [ ] Test script existence
- [ ] Test script permissions
- [ ] Test script syntax
- [ ] Test script parameters
- [ ] Test environment check
- [ ] Test output handling
- [ ] Test error handling

## Test Implementation Status
- [x] Basic environment tests implemented
- [ ] Difficulty level tests implemented
- [ ] Rectangular board tests implemented
- [ ] Curriculum learning tests implemented
- [ ] Performance tests implemented
- [ ] Integration tests implemented

## Notes
- Tests should be implemented in appropriate test files under the new directory structure
- Each test should have clear documentation
- Tests should be independent and repeatable
- Tests should cover both success and failure cases
- Tests should verify all edge cases
- Tests should be efficient and not take too long to run 