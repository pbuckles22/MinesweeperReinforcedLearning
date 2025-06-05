param(
    [Parameter(Mandatory=$true)]
    [string]$DestinationPath,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# List of files to copy
$filesToCopy = @(
    "src\scripts\install_and_run.ps1",
    "src\scripts\install_and_run.sh",
    "requirements.txt",
    "src\core\train_agent.py",
    "tests\test_train_agent.py",
    "src\core\minesweeper_env.py"
)

# Convert destination path to absolute path
$DestinationPath = (Resolve-Path $DestinationPath).Path

# Check if any files exist in destination
$existingFiles = $filesToCopy | Where-Object { Test-Path (Join-Path $DestinationPath (Split-Path $_ -Leaf)) }
if ($existingFiles) {
    if ($Force) {
        Write-Host "Removing existing files in: $DestinationPath"
        foreach ($file in $existingFiles) {
            Remove-Item -Force (Join-Path $DestinationPath (Split-Path $file -Leaf))
        }
    } else {
        $response = Read-Host "Files already exist in destination. Overwrite? (y/n)"
        if ($response -ne "y") {
            Write-Host "Exiting script."
            exit
        }
        Write-Host "Removing existing files in: $DestinationPath"
        foreach ($file in $existingFiles) {
            Remove-Item -Force (Join-Path $DestinationPath (Split-Path $file -Leaf))
        }
    }
}

# Create destination directory if it doesn't exist
if (-not (Test-Path $DestinationPath)) {
    Write-Host "Creating directory: $DestinationPath"
    New-Item -ItemType Directory -Path $DestinationPath | Out-Null
}

# Copy files
foreach ($file in $filesToCopy) {
    if (Test-Path $file) {
        Write-Host "Copying $file to $DestinationPath"
        Copy-Item $file -Destination $DestinationPath
    } else {
        Write-Host "Warning: $file not found in source directory"
    }
}

Write-Host "`nFiles copied successfully to: $DestinationPath"
Write-Host "To run the installation script, navigate to the directory and run:"
Write-Host "Navigate to: $DestinationPath"
Write-Host ".\install_and_run.ps1" 