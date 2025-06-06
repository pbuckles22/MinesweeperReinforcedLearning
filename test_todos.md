# Test TODOs

## High Priority Issues (Core Functionality)

1. **Action Space and Board Size Issues**
   - Fix `current_board_size` attribute references (use `current_board_width` and `current_board_height`)
   - Update action space calculations to use correct dimensions
   - Fix board size validation in initialization tests
   - Priority: HIGH (affects multiple test files)

2. **Mine Hit and Game Termination**
   - Fix mine hit state updates and rewards
   - Ensure proper game termination on mine hits
   - Fix first move behavior and reset logic
   - Priority: HIGH (core gameplay mechanics)

3. **Invalid Action Handling**
   - Implement proper error handling for invalid actions
   - Add reward breakdown in info dictionary
   - Fix invalid action penalties
   - Priority: HIGH (affects agent training)

## Medium Priority Issues

4. **Core Mechanics Tests**
   - Fix board state initialization in test fixtures
   - Correct adjacent mine count calculations
   - Fix safe cell reveal and cascade logic
   - Priority: MEDIUM (basic game mechanics)

5. **Flag Placement Tests**
   - Fix flag placement on mines and safe cells
   - Implement proper flag removal
   - Update flag action validation
   - Priority: MEDIUM (game mechanics)

## Low Priority Issues

6. **Test Infrastructure**
   - Update test fixtures to use correct board sizes
   - Fix test environment setup
   - Add more comprehensive test coverage
   - Priority: LOW (test quality)

## Test Status Summary
- Total Tests: 57
- Passed: 32
- Failed: 19
- Errors: 6

## Next Steps
1. Fix action space and board size issues first
2. Address mine hit and game termination logic
3. Implement proper invalid action handling
4. Update core mechanics tests
5. Fix flag placement tests
6. Improve test infrastructure

## Notes
- Many failures are related to the `current_board_size` attribute being replaced with `current_board_width` and `current_board_height`
- Several tests need to be updated to match the new error message format
- Some tests are failing due to incorrect assumptions about game behavior
- Need to ensure consistent behavior across all test files 