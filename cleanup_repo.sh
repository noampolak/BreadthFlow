#!/bin/bash

# 🧹 BreadthFlow Repository Cleanup Script
# This script removes all Python cache files and other unnecessary files

echo "🧹 Starting BreadthFlow Repository Cleanup..."
echo "=============================================="

# Count files before cleanup
echo "📊 Files to be removed:"
echo "  - .pyc files: $(find . -name "*.pyc" -type f | wc -l)"
echo "  - __pycache__ directories: $(find . -name "__pycache__" -type d | wc -l)"
echo "  - .pyo files: $(find . -name "*.pyo" -type f | wc -l)"
echo "  - .DS_Store files: $(find . -name ".DS_Store" -type f | wc -l)"
echo "  - .log files: $(find . -name "*.log" -type f | wc -l)"
echo ""

# Ask for confirmation
read -p "🤔 Do you want to proceed with cleanup? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cleanup cancelled."
    exit 1
fi

echo "🚀 Starting cleanup..."

# Remove Python cache files
echo "  🐍 Removing .pyc files..."
find . -name "*.pyc" -type f -delete

echo "  📁 Removing __pycache__ directories..."
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "  🗑️  Removing .pyo files..."
find . -name "*.pyo" -type f -delete

echo "  🍎 Removing .DS_Store files..."
find . -name ".DS_Store" -type f -delete

echo "  📝 Removing .log files..."
find . -name "*.log" -type f -delete

echo "  🧽 Removing temporary files..."
find . -name "*.tmp" -type f -delete
find . -name "*.temp" -type f -delete

# Remove Jupyter checkpoints
echo "  📓 Removing Jupyter checkpoints..."
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} + 2>/dev/null || true

# Remove IDE files
echo "  💻 Removing IDE files..."
find . -name ".vscode" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name ".idea" -type d -exec rm -rf {} + 2>/dev/null || true
find . -name "*.swp" -type f -delete
find . -name "*.swo" -type f -delete

echo ""
echo "✅ Cleanup completed!"
echo ""

# Count remaining files
echo "📊 Remaining files:"
echo "  - .pyc files: $(find . -name "*.pyc" -type f | wc -l)"
echo "  - __pycache__ directories: $(find . -name "__pycache__" -type d | wc -l)"
echo "  - .pyo files: $(find . -name "*.pyo" -type f | wc -l)"
echo "  - .DS_Store files: $(find . -name ".DS_Store" -type f | wc -l)"
echo "  - .log files: $(find . -name "*.log" -type f | wc -l)"
echo ""

echo "🎉 Repository is now clean!"
echo "💡 The .gitignore file has been created to prevent this in the future."
echo ""
echo "📝 Next steps:"
echo "  1. git add .gitignore"
echo "  2. git commit -m 'Add .gitignore and cleanup repository'"
echo "  3. git add ."
echo "  4. git commit -m 'Remove cache files and clean repository'"
