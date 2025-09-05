#!/usr/bin/env python3
"""
Main entry point for the Cline Recorder MCP Server
Can run either the MCP server or the REST API server
"""

import asyncio
import argparse
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_mcp_server():
    """Run the MCP server"""
    from mcp_server import main as mcp_main
    logger.info("Starting MCP server...")
    await mcp_main()

def run_api_server():
    """Run the REST API server"""
    import uvicorn
    from api_server import app
    
    port = int(os.getenv("API_PORT", 8001))
    host = os.getenv("API_HOST", "0.0.0.0")
    
    logger.info(f"Starting REST API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)

def run_both():
    """Run both MCP server and REST API server"""
    import threading
    import time
    
    # Start API server in a separate thread
    api_thread = threading.Thread(target=run_api_server, daemon=True)
    api_thread.start()
    
    # Give API server time to start
    time.sleep(2)
    
    # Run MCP server in main thread
    logger.info("Starting both MCP server and REST API server...")
    asyncio.run(run_mcp_server())

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Cline Recorder MCP Server")
    parser.add_argument(
        "--mode",
        choices=["mcp", "api", "both"],
        default="mcp",
        help="Server mode to run (default: mcp)"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database tables"
    )
    
    args = parser.parse_args()
    
    if args.init_db:
        from database import init_db
        logger.info("Initializing database...")
        init_db()
        logger.info("Database initialization complete")
        return
    
    if args.mode == "mcp":
        asyncio.run(run_mcp_server())
    elif args.mode == "api":
        run_api_server()
    elif args.mode == "both":
        run_both()
    else:
        logger.error(f"Unknown mode: {args.mode}")
        return 1

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise