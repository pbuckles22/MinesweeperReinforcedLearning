#!/bin/bash

# Cursor Settings Updater Script
# This script locates and updates Cursor's settings.json with useful settings

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Cursor Settings Updater${NC}"
echo "================================"

# Function to find Cursor settings directory
find_cursor_settings() {
    local settings_dir=""
    
    echo -e "${YELLOW}🔍 Detecting operating system...${NC}" >&2
    
    # Check different possible locations based on OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        settings_dir="$HOME/Library/Application Support/Cursor/User"
        echo -e "${GREEN}✅ Detected macOS${NC}" >&2
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        settings_dir="$HOME/.config/Cursor/User"
        echo -e "${GREEN}✅ Detected Linux${NC}" >&2
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "win32" ]]; then
        # Windows
        settings_dir="$APPDATA/Cursor/User"
        echo -e "${GREEN}✅ Detected Windows${NC}" >&2
    else
        echo -e "${RED}❌ Unsupported operating system: $OSTYPE${NC}" >&2
        exit 1
    fi
    
    # Escape spaces for display
    local display_path=$(echo "$settings_dir" | sed 's/ /\\ /g')
    echo -e "${YELLOW}📁 Looking for Cursor settings in: $display_path${NC}" >&2
    
    if [[ ! -d "$settings_dir" ]]; then
        echo -e "${RED}❌ Cursor settings directory not found at: $display_path${NC}" >&2
        echo -e "${YELLOW}💡 Make sure Cursor is installed and has been run at least once${NC}" >&2
        echo -e "${YELLOW}💡 Expected directory: $display_path${NC}" >&2
        exit 1
    fi
    
    echo -e "${GREEN}✅ Found Cursor settings directory${NC}" >&2
    printf "%s" "$settings_dir"
}

# Function to backup existing settings
backup_settings() {
    local settings_file="$1"
    local backup_file="${settings_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    echo -e "${YELLOW}📋 Checking for existing settings...${NC}"
    
    if [[ -f "$settings_file" ]]; then
        echo -e "${YELLOW}📋 Creating backup of existing settings...${NC}"
        cp "$settings_file" "$backup_file"
        echo -e "${GREEN}✅ Backup created: $(basename "$backup_file")${NC}"
    else
        echo -e "${YELLOW}📝 No existing settings.json found, will create new one${NC}"
    fi
}

# Function to generate new settings content
generate_new_settings() {
    local current_dir="$(pwd)"
    
    echo -e "${YELLOW}⚙️  Generating new settings for: $current_dir${NC}"
    
    cat << EOF
{
    "terminal.integrated.cwd": "${current_dir}",
    "terminal.integrated.defaultProfile.osx": "zsh",
    "terminal.integrated.defaultProfile.linux": "bash",
    "terminal.integrated.defaultProfile.windows": "PowerShell",
    "python.defaultInterpreterPath": "${current_dir}/venv/bin/python",
    "python.terminal.activateEnvironment": true,
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
        "source.organizeImports": true
    },
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/venv": false,
        "**/.pytest_cache": true,
        "**/mlruns": true,
        "**/logs": true,
        "**/models": true,
        "**/tensorboard": true
    },
    "search.exclude": {
        "**/venv": true,
        "**/__pycache__": true,
        "**/*.pyc": true,
        "**/mlruns": true,
        "**/logs": true,
        "**/models": true,
        "**/tensorboard": true
    },
    "files.watcherExclude": {
        "**/venv/**": true,
        "**/__pycache__/**": true,
        "**/mlruns/**": true,
        "**/logs/**": true,
        "**/models/**": true,
        "**/tensorboard/**": true
    },
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.unittestEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "workbench.colorTheme": "Default Dark+",
    "editor.rulers": [88],
    "editor.tabSize": 4,
    "editor.insertSpaces": true,
    "files.trimTrailingWhitespace": true,
    "files.insertFinalNewline": true,
    "emmet.includeLanguages": {
        "python": "html"
    }
}
EOF
}

# Function to update settings
update_settings() {
    local settings_file="$1"
    
    echo -e "${YELLOW}⚙️  Updating Cursor settings...${NC}"
    
    # Create settings directory if it doesn't exist
    mkdir -p "$(dirname "$settings_file")"
    
    # Generate and write new settings
    generate_new_settings > "$settings_file"
    
    echo -e "${GREEN}✅ Settings updated successfully!${NC}"
}

# Function to show current settings
show_settings() {
    local settings_file="$1"
    
    # Escape spaces for display
    local display_path=$(echo "$settings_file" | sed 's/ /\\ /g')
    echo -e "${YELLOW}📖 Reading current settings from: $display_path${NC}"
    
    if [[ -f "$settings_file" ]]; then
        echo -e "${BLUE}📖 Current settings:${NC}"
        echo "================================"
        # Use python3 to format JSON, fallback to cat if python3 fails
        if command -v python3 >/dev/null 2>&1; then
            python3 -m json.tool "$settings_file" 2>/dev/null || cat "$settings_file"
        else
            cat "$settings_file"
        fi
    else
        echo -e "${YELLOW}📝 No settings.json file found at: $display_path${NC}"
    fi
}

# Function to show differences between current and new settings
show_differences() {
    local settings_file="$1"
    local temp_new_settings="/tmp/cursor_new_settings_$$.json"
    
    echo -e "${YELLOW}📊 Preparing settings comparison...${NC}"
    
    # Generate new settings to temp file
    generate_new_settings > "$temp_new_settings"
    
    echo -e "${BLUE}📊 Settings Comparison${NC}"
    echo "================================"
    echo ""
    
    if [[ -f "$settings_file" ]]; then
        echo -e "${YELLOW}📖 Current settings:${NC}"
        echo "================================"
        # Use python3 to format JSON, fallback to cat if python3 fails
        if command -v python3 >/dev/null 2>&1; then
            python3 -m json.tool "$settings_file" 2>/dev/null || cat "$settings_file"
        else
            cat "$settings_file"
        fi
        echo ""
        echo -e "${GREEN}🆕 New settings:${NC}"
        echo "================================"
        if command -v python3 >/dev/null 2>&1; then
            python3 -m json.tool "$temp_new_settings" 2>/dev/null || cat "$temp_new_settings"
        else
            cat "$temp_new_settings"
        fi
        echo ""
        
        # Show diff if available
        if command -v diff >/dev/null 2>&1; then
            echo -e "${BLUE}🔍 Differences:${NC}"
            echo "================================"
            if diff -u "$settings_file" "$temp_new_settings" 2>/dev/null; then
                echo -e "${GREEN}✅ No differences found - settings are already up to date${NC}"
            fi
        else
            echo -e "${YELLOW}💡 Install 'diff' command to see detailed differences${NC}"
        fi
    else
        # Escape spaces for display
        local display_path=$(echo "$settings_file" | sed 's/ /\\ /g')
        echo -e "${YELLOW}📝 No existing settings.json found at: $display_path${NC}"
        echo ""
        echo -e "${GREEN}🆕 New settings that would be created:${NC}"
        echo "================================"
        if command -v python3 >/dev/null 2>&1; then
            python3 -m json.tool "$temp_new_settings" 2>/dev/null || cat "$temp_new_settings"
        else
            cat "$temp_new_settings"
        fi
    fi
    
    # Clean up temp file
    rm -f "$temp_new_settings"
    echo -e "${GREEN}✅ Comparison complete${NC}"
}

# Function to restore backup
restore_backup() {
    local settings_file="$1"
    local backup_files=($(ls -t "${settings_file}.backup."* 2>/dev/null || true))
    
    echo -e "${YELLOW}🔄 Looking for backup files...${NC}"
    
    if [[ ${#backup_files[@]} -eq 0 ]]; then
        echo -e "${RED}❌ No backup files found${NC}"
        return 1
    fi
    
    local latest_backup="${backup_files[0]}"
    echo -e "${YELLOW}🔄 Restoring from backup: $(basename "$latest_backup")${NC}"
    
    cp "$latest_backup" "$settings_file"
    echo -e "${GREEN}✅ Settings restored successfully!${NC}"
}

# Main script logic
main() {
    echo -e "${YELLOW}🚀 Starting Cursor Settings Updater...${NC}"
    
    # Parse command line arguments
    local action="update"
    local show_current=false
    local restore=false
    local preview=false
    
    echo -e "${YELLOW}📝 Parsing command line arguments...${NC}"
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --show|-s)
                show_current=true
                echo -e "${GREEN}✅ Show current settings mode${NC}"
                shift
                ;;
            --preview|-p)
                preview=true
                echo -e "${GREEN}✅ Preview mode${NC}"
                shift
                ;;
            --restore|-r)
                restore=true
                echo -e "${GREEN}✅ Restore mode${NC}"
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [OPTIONS]"
                echo ""
                echo "Options:"
                echo "  --show, -s     Show current settings without updating"
                echo "  --preview, -p  Show differences between current and new settings"
                echo "  --restore, -r  Restore from latest backup"
                echo "  --help, -h     Show this help message"
                echo ""
                echo "Default action: Update settings with optimized configuration"
                exit 0
                ;;
            *)
                echo -e "${RED}❌ Unknown option: $1${NC}"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    echo -e "${YELLOW}🔍 Finding Cursor settings directory...${NC}"
    
    # Find Cursor settings directory
    local settings_dir=$(find_cursor_settings)
    local settings_file="$settings_dir/settings.json"
    
    # Escape spaces for display
    local display_path=$(echo "$settings_file" | sed 's/ /\\ /g')
    echo -e "${BLUE}🎯 Target settings file: $display_path${NC}"
    echo ""
    
    if [[ "$show_current" == true ]]; then
        show_settings "$settings_file"
        exit 0
    fi
    
    if [[ "$preview" == true ]]; then
        show_differences "$settings_file"
        exit 0
    fi
    
    if [[ "$restore" == true ]]; then
        restore_backup "$settings_file"
        exit 0
    fi
    
    echo -e "${YELLOW}📋 Proceeding with settings update...${NC}"
    
    # Backup existing settings
    backup_settings "$settings_file"
    
    # Update settings
    update_settings "$settings_file"
    
    echo ""
    echo -e "${GREEN}🎉 Cursor settings updated successfully!${NC}"
    echo ""
    echo -e "${BLUE}📋 What was updated:${NC}"
    echo "  • Terminal working directory set to current project"
    echo "  • Python interpreter set to project's virtual environment"
    echo "  • Pytest testing configuration enabled"
    echo "  • Code formatting and linting settings"
    echo "  • File exclusions for better performance"
    echo "  • Editor preferences for Python development"
    echo ""
    echo -e "${YELLOW}💡 Restart Cursor for all changes to take effect${NC}"
    echo ""
    echo -e "${BLUE}🔄 To restore previous settings, run: $0 --restore${NC}"
}

# Run main function with all arguments
main "$@" 