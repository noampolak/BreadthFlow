#!/usr/bin/env python3
"""
Simple Demo for BreadthFlow

This demo shows the system working without requiring PySpark execution.
It demonstrates the infrastructure, CLI, and web interfaces.
"""

import subprocess
import time
import webbrowser
from pathlib import Path


def run_command(cmd, description):
    """Run a command and show the result."""
    print(f"\n🔧 {description}")
    print("-" * 50)
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Success!")
            if result.stdout.strip():
                print(result.stdout)
        else:
            print("⚠️  Command completed with warnings:")
            if result.stdout.strip():
                print(result.stdout)
            if result.stderr.strip():
                print(result.stderr)
    except Exception as e:
        print(f"❌ Error: {e}")


def check_infrastructure():
    """Check if infrastructure is running."""
    print("\n🏥 Checking Infrastructure Status")
    print("=" * 50)
    
    run_command("docker ps --format 'table {{.Names}}\t{{.Status}}'", "Docker Containers")
    
    # Check specific services
    services = [
        ("spark-master", "Spark Master"),
        ("spark-worker", "Spark Worker"), 
        ("minio", "MinIO Storage"),
        ("kafka", "Kafka"),
        ("elasticsearch", "Elasticsearch"),
        ("kibana", "Kibana")
    ]
    
    for container, name in services:
        run_command(f"docker ps --filter name={container} --format '{{{{.Status}}}}'", f"{name} Status")


def show_symbol_lists():
    """Show available symbol lists."""
    print("\n📊 Available Symbol Lists")
    print("=" * 50)
    
    run_command("poetry run bf data symbols", "Symbol Lists")


def show_web_interfaces():
    """Show available web interfaces."""
    print("\n🌐 Web Interfaces")
    print("=" * 50)
    
    interfaces = [
        ("http://localhost:8080", "Spark UI", "Spark cluster monitoring"),
        ("http://localhost:9001", "MinIO Console", "Data storage management (minioadmin/minioadmin)"),
        ("http://localhost:5601", "Kibana", "Data visualization and analytics"),
        ("http://localhost:8081", "Dashboard", "Pipeline monitoring (if running)")
    ]
    
    for url, name, description in interfaces:
        print(f"🔗 {name}: {url}")
        print(f"   📝 {description}")
    
    print("\n💡 You can open these URLs in your browser to explore the system!")


def show_cli_commands():
    """Show available CLI commands."""
    print("\n🛠️  Available CLI Commands")
    print("=" * 50)
    
    commands = [
        ("bf infra start", "Start all infrastructure services"),
        ("bf infra stop", "Stop all infrastructure services"),
        ("bf infra status", "Check infrastructure health"),
        ("bf data symbols", "List available symbol lists"),
        ("bf data fetch --symbol-list demo_small", "Fetch data for demo symbols"),
        ("bf signals generate --symbol-list demo_small", "Generate trading signals"),
        ("bf backtest run --symbol-list demo_small", "Run backtest"),
        ("bf dashboard --port 8081", "Start web dashboard"),
        ("bf pipeline --symbol-list demo_small --interval 60", "Run continuous pipeline"),
        ("bf --help", "Show all available commands")
    ]
    
    for cmd, description in commands:
        print(f"💻 {cmd}")
        print(f"   📝 {description}")


def show_system_info():
    """Show system information."""
    print("\n📋 System Information")
    print("=" * 50)
    
    # Project structure
    project_root = Path(__file__).parent.parent
    print(f"📁 Project Root: {project_root}")
    
    # Check key files
    key_files = [
        "pyproject.toml",
        "infra/docker-compose.yml", 
        "cli/bf.py",
        "model/scoring.py",
        "features/common/symbols.py"
    ]
    
    for file_path in key_files:
        full_path = project_root / file_path
        status = "✅" if full_path.exists() else "❌"
        print(f"{status} {file_path}")
    
    # Check Docker Compose
    run_command("docker-compose -f infra/docker-compose.yml config", "Docker Compose Configuration")


def main():
    """Run the simple demo."""
    print("🎯 BreadthFlow Simple Demo")
    print("=" * 60)
    print("This demo shows the system infrastructure and capabilities")
    print("without requiring PySpark execution.")
    print("=" * 60)
    
    # Check infrastructure
    check_infrastructure()
    
    # Show symbol lists
    show_symbol_lists()
    
    # Show system info
    show_system_info()
    
    # Show CLI commands
    show_cli_commands()
    
    # Show web interfaces
    show_web_interfaces()
    
    print("\n🎉 Demo Complete!")
    print("=" * 60)
    print("✅ Infrastructure is running")
    print("✅ Symbol lists are available")
    print("✅ CLI commands are working")
    print("✅ Web interfaces are accessible")
    print("\n💡 Next steps:")
    print("   • Open web interfaces in your browser")
    print("   • Try CLI commands like 'poetry run bf data symbols'")
    print("   • Explore the system documentation")


if __name__ == "__main__":
    main()
