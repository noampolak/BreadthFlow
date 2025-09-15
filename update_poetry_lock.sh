#!/bin/bash

# 🔒 Update Poetry Lock File Script
# This script regenerates the poetry.lock file to match pyproject.toml

echo "🔒 Updating Poetry Lock File..."
echo "================================"

# Check if poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry is not installed. Installing..."
    pip install poetry
fi

echo "📋 Current pyproject.toml and poetry.lock status:"
echo "  - pyproject.toml: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" pyproject.toml 2>/dev/null || stat -c "%y" pyproject.toml 2>/dev/null)"
echo "  - poetry.lock: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" poetry.lock 2>/dev/null || stat -c "%y" poetry.lock 2>/dev/null)"

echo ""
echo "🔄 Regenerating poetry.lock file..."

# Update the lock file
poetry lock

if [ $? -eq 0 ]; then
    echo "✅ Poetry lock file updated successfully!"
    echo ""
    echo "📊 Updated files:"
    echo "  - pyproject.toml: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" pyproject.toml 2>/dev/null || stat -c "%y" pyproject.toml 2>/dev/null)"
    echo "  - poetry.lock: $(stat -f "%Sm" -t "%Y-%m-%d %H:%M:%S" poetry.lock 2>/dev/null || stat -c "%y" poetry.lock 2>/dev/null)"
    echo ""
    echo "💡 Next steps:"
    echo "  1. git add poetry.lock"
    echo "  2. git commit -m 'Update poetry.lock file'"
    echo "  3. git push"
else
    echo "❌ Failed to update poetry.lock file"
    echo "   Please check the error messages above"
    exit 1
fi
