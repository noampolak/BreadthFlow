#!/bin/bash
# Breadth/Thrust Signals POC - Setup Script

set -e  # Exit on any error

echo "🚀 Breadth/Thrust Signals POC - Setup Script"
echo "=============================================="

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Poetry not found. Installing Poetry..."
    curl -sSL https://install.python-poetry.org | python3 -
    echo "✅ Poetry installed successfully"
    echo "💡 You may need to restart your terminal or run: source ~/.bashrc"
else
    echo "✅ Poetry is already installed"
fi

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.9"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" = "$required_version" ]; then
    echo "✅ Python version $python_version is compatible"
else
    echo "❌ Python version $python_version is too old. Please upgrade to Python 3.9+"
    exit 1
fi

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker: https://docs.docker.com/get-docker/"
    exit 1
else
    echo "✅ Docker is installed"
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose"
    exit 1
else
    echo "✅ Docker Compose is available"
fi

# Install dependencies
echo "📦 Installing Python dependencies with Poetry..."
poetry install

# Install ML dependencies separately for faster CI
echo "📦 Installing ML dependencies for testing..."
pip install -r requirements-ci.txt

# Setup environment
echo "⚙️  Setting up environment..."
if [ ! -f .env ]; then
    cp env.example .env
    echo "✅ Environment file created (.env)"
    echo "📝 Please edit .env with your configuration"
else
    echo "ℹ️  Environment file already exists"
fi

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Edit .env file with your configuration"
echo "  2. Start infrastructure: poetry run bf infra start"
echo "  3. Run demo: poetry run bf demo"
echo ""
echo "💡 Useful commands:"
echo "  • poetry shell                    # Activate virtual environment"
echo "  • poetry run bf --help           # Show CLI help"
echo "  • poetry run bf infra status     # Check infrastructure status"
echo ""
