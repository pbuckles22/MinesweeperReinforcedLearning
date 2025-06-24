# Quick Training Script for Windows
# Uses the new modular training script for best results

Write-Host "🚀 Starting Quick Training (Modular Script)" -ForegroundColor Green
Write-Host "Board: 4x4, Mines: 2, Timesteps: 10,000" -ForegroundColor Yellow
Write-Host "Expected: 20%+ win rate in ~10 minutes" -ForegroundColor Yellow
Write-Host ""

# Use modular script for best results
python src/core/train_agent_modular.py `
    --board_size 4 `
    --max_mines 2 `
    --total_timesteps 10000

Write-Host ""
Write-Host "✅ Training complete! Check results in modular_results_*.json" -ForegroundColor Green
Write-Host ""
Write-Host "💡 For legacy script (advanced features), use:" -ForegroundColor Cyan
Write-Host "   python src/core/train_agent.py --simple_training_mode --total_timesteps 10000" -ForegroundColor Gray 