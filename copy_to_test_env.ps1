param(
    [Parameter(Mandatory=$true)]
    [string]$DestinationPath,
    
    [Parameter(Mandatory=$false)]
    [switch]$Force
)

# Get the script's directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

# List of files to copy
$filesToCopy = @(
    "install_and_run.ps1",
    "install_and_run.sh",
    "requirements.txt",
    "train_agent.py",
    "test_train_agent.py",
    "minesweeper_env.py"
)

# Convert destination path to absolute path
$DestinationPath = (Resolve-Path $DestinationPath).Path

# Check if any files exist in destination
$existingFiles = $filesToCopy | Where-Object { Test-Path (Join-Path $DestinationPath $_) }
if ($existingFiles) {
    if ($Force) {
        Write-Host "Removing existing files in: $DestinationPath"
        foreach ($file in $existingFiles) {
            Remove-Item -Force (Join-Path $DestinationPath $file)
        }
    } else {
        $response = Read-Host "Files already exist in destination. Overwrite? (y/n)"
        if ($response -ne "y") {
            Write-Host "Exiting script."
            exit
        }
        Write-Host "Removing existing files in: $DestinationPath"
        foreach ($file in $existingFiles) {
            Remove-Item -Force (Join-Path $DestinationPath $file)
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
    $sourceFile = Join-Path $scriptDir $file
    if (Test-Path $sourceFile) {
        Write-Host "Copying $file to $DestinationPath"
        Copy-Item $sourceFile -Destination $DestinationPath
    } else {
        Write-Host "Warning: $file not found in source directory"
    }
}

Write-Host "`nFiles copied successfully to: $DestinationPath"
Write-Host "To run the installation script, navigate to the directory and run:"
Write-Host "cd $DestinationPath"
Write-Host ".\install_and_run.ps1" 