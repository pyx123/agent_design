#!/usr/bin/env python3
"""
Setup script for Cline Recorder MCP Server
"""

import os
import sys
import subprocess
import sqlite3
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_dependencies():
    """Install Python dependencies"""
    print("Installing Python dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Error installing dependencies: {e}")
        sys.exit(1)

def create_env_file():
    """Create .env file if it doesn't exist"""
    env_file = Path(".env")
    if not env_file.exists():
        print("Creating .env file...")
        with open(env_file, "w") as f:
            f.write("""# Database Configuration
DATABASE_URL=postgresql://cline_user:cline_password@localhost/cline_records

# Server Configuration
MCP_PORT=8000
API_PORT=8001
API_HOST=0.0.0.0

# Logging
LOG_LEVEL=INFO
""")
        print("✓ .env file created")
    else:
        print("✓ .env file already exists")

def setup_sqlite_database():
    """Setup SQLite database for development/testing"""
    print("Setting up SQLite database for development...")
    
    # Update .env to use SQLite
    env_file = Path(".env")
    if env_file.exists():
        with open(env_file, "r") as f:
            content = f.read()
        
        # Replace PostgreSQL URL with SQLite
        content = content.replace(
            "DATABASE_URL=postgresql://cline_user:cline_password@localhost/cline_records",
            "DATABASE_URL=sqlite:///cline_records.db"
        )
        
        with open(env_file, "w") as f:
            f.write(content)
        
        print("✓ Updated .env to use SQLite database")
    
    # Create SQLite database
    try:
        conn = sqlite3.connect("cline_records.db")
        conn.close()
        print("✓ SQLite database created")
    except Exception as e:
        print(f"Error creating SQLite database: {e}")

def setup_postgres_database():
    """Setup PostgreSQL database"""
    print("Setting up PostgreSQL database...")
    print("Note: This requires PostgreSQL to be installed and running")
    
    # Check if psql is available
    try:
        subprocess.run(["psql", "--version"], capture_output=True, check=True)
        print("✓ PostgreSQL client detected")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: PostgreSQL client not found")
        print("Please install PostgreSQL and ensure 'psql' is in your PATH")
        return False
    
    # Create database
    try:
        subprocess.run([
            "psql", "-U", "postgres", "-c", 
            "CREATE DATABASE cline_records;"
        ], capture_output=True, check=True)
        print("✓ Database 'cline_records' created")
        
        subprocess.run([
            "psql", "-U", "postgres", "-c", 
            "CREATE USER cline_user WITH PASSWORD 'cline_password';"
        ], capture_output=True, check=True)
        print("✓ User 'cline_user' created")
        
        subprocess.run([
            "psql", "-U", "postgres", "-c", 
            "GRANT ALL PRIVILEGES ON DATABASE cline_records TO cline_user;"
        ], capture_output=True, check=True)
        print("✓ Privileges granted")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error setting up PostgreSQL: {e}")
        print("You may need to run as a user with sufficient privileges")
        return False

def initialize_database():
    """Initialize database tables"""
    print("Initializing database tables...")
    try:
        subprocess.check_call([sys.executable, "main.py", "--init-db"])
        print("✓ Database tables initialized")
    except subprocess.CalledProcessError as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

def run_tests():
    """Run basic tests"""
    print("Running basic tests...")
    try:
        subprocess.check_call([sys.executable, "test_server.py"])
        print("✓ Tests completed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Tests failed: {e}")
        print("You can run tests manually later with: python test_server.py")

def main():
    """Main setup function"""
    print("Cline Recorder MCP Server - Setup")
    print("=" * 40)
    
    # Check Python version
    check_python_version()
    
    # Install dependencies
    install_dependencies()
    
    # Create environment file
    create_env_file()
    
    # Database setup
    print("\nDatabase Setup:")
    print("1. SQLite (recommended for development)")
    print("2. PostgreSQL (recommended for production)")
    
    choice = input("Choose database type (1 or 2): ").strip()
    
    if choice == "1":
        setup_sqlite_database()
    elif choice == "2":
        if not setup_postgres_database():
            print("Falling back to SQLite...")
            setup_sqlite_database()
    else:
        print("Invalid choice, using SQLite...")
        setup_sqlite_database()
    
    # Initialize database
    initialize_database()
    
    # Run tests
    run_tests()
    
    print("\n" + "=" * 40)
    print("Setup completed successfully!")
    print("\nTo start the servers:")
    print("  MCP Server: python main.py --mode mcp")
    print("  API Server: python main.py --mode api")
    print("  Both: python main.py --mode both")
    print("\nFor Docker deployment:")
    print("  docker-compose up -d")
    print("\nDocumentation: README.md and USAGE.md")

if __name__ == "__main__":
    main()