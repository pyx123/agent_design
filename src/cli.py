"""Command-line interface for DevOps Agent."""

import asyncio
import json
import sys
from datetime import datetime
from pathlib import Path

import click
import httpx
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.markdown import Markdown

from src.config import settings

console = Console()


@click.group()
def cli():
    """DevOps Agent CLI for troubleshooting."""
    pass


@cli.command()
@click.option("--title", "-t", required=True, help="Issue title")
@click.option("--service", "-s", required=True, help="Affected service")
@click.option("--description", "-d", help="Detailed description")
@click.option("--environment", "-e", default="prod", help="Environment (prod/staging/dev)")
@click.option("--severity", default="medium", help="Severity (critical/high/medium/low)")
@click.option("--sync", is_flag=True, help="Run synchronously and wait for result")
def troubleshoot(title, service, description, environment, severity, sync):
    """Create a new troubleshooting request."""
    
    # Build request
    request_data = {
        "title": title,
        "service": service,
        "description": description,
        "environment": environment,
        "severity": severity,
        "mode": "sync" if sync else "async"
    }
    
    # Make API call
    url = f"http://{settings.server.host}:{settings.server.port}/api/v1/troubleshoot"
    
    try:
        with console.status("[bold green]Creating troubleshooting request..."):
            response = httpx.post(url, json=request_data, timeout=300 if sync else 30)
            response.raise_for_status()
            
        result = response.json()
        console.print(f"[green]✓[/green] Request created: {result['id']}")
        console.print(f"Status: {result['status']}")
        
        if sync and result["status"] == "completed":
            # Get full result
            asyncio.run(_show_result(result["id"]))
        elif not sync:
            console.print(f"\n[yellow]Run 'devops-agent status {result['id']}' to check progress[/yellow]")
            
    except httpx.HTTPError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.option("--status", "-s", help="Filter by status")
@click.option("--service", help="Filter by service")
@click.option("--limit", "-l", default=10, help="Number of results")
def list(status, service, limit):
    """List troubleshooting requests."""
    
    url = f"http://{settings.server.host}:{settings.server.port}/api/v1/troubleshoot"
    params = {"size": limit}
    if status:
        params["status"] = status
    if service:
        params["service"] = service
    
    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()
        
        requests = response.json()
        
        if not requests:
            console.print("[yellow]No requests found[/yellow]")
            return
        
        # Create table
        table = Table(title="Troubleshooting Requests")
        table.add_column("ID", style="cyan")
        table.add_column("Title", style="white")
        table.add_column("Service", style="blue")
        table.add_column("Status", style="green")
        table.add_column("Created", style="yellow")
        
        for req in requests:
            table.add_row(
                req["id"],
                req["title"][:50] + "..." if len(req["title"]) > 50 else req["title"],
                req["service"],
                req["status"],
                req["created_at"][:16]
            )
        
        console.print(table)
        
    except httpx.HTTPError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
@click.argument("request_id")
def status(request_id):
    """Check status of a troubleshooting request."""
    asyncio.run(_show_result(request_id))


@cli.command()
@click.argument("request_id")
@click.option("--format", "-f", default="markdown", help="Report format (markdown/html/pdf)")
@click.option("--output", "-o", help="Output file")
def report(request_id, format, output):
    """Generate troubleshooting report."""
    
    url = f"http://{settings.server.host}:{settings.server.port}/api/v1/troubleshoot/{request_id}/report"
    params = {"format": format}
    
    try:
        response = httpx.get(url, params=params)
        response.raise_for_status()
        
        if output:
            # Save to file
            Path(output).write_text(response.text)
            console.print(f"[green]✓[/green] Report saved to {output}")
        else:
            # Display in console
            if format == "markdown":
                md = Markdown(response.text)
                console.print(md)
            else:
                console.print(response.text)
                
    except httpx.HTTPError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


@cli.command()
def serve():
    """Start the API server."""
    from src.app import main
    main()


async def _show_result(request_id: str):
    """Show detailed result for a request."""
    url = f"http://{settings.server.host}:{settings.server.port}/api/v1/troubleshoot/{request_id}"
    
    try:
        response = httpx.get(url)
        response.raise_for_status()
        
        result = response.json()
        
        # Basic info
        console.print(f"\n[bold]Request Details[/bold]")
        console.print(f"ID: {result['id']}")
        console.print(f"Title: {result['request']['title']}")
        console.print(f"Service: {result['request']['service']}")
        console.print(f"Status: [bold {_status_color(result['status'])}]{result['status']}[/bold {_status_color(result['status'])}]")
        
        # Evidence
        if result.get("evidence"):
            console.print(f"\n[bold]Evidence Collected[/bold] ({len(result['evidence'])} items)")
            for i, ev in enumerate(result["evidence"][:5], 1):
                console.print(f"{i}. [{ev['source']}] {ev['summary']}")
        
        # Root cause
        if result.get("root_cause"):
            rc = result["root_cause"]
            console.print(f"\n[bold]Root Cause Analysis[/bold]")
            console.print(f"Hypothesis: {rc['hypothesis']}")
            console.print(f"Confidence: {rc['confidence']:.1%}")
            if rc.get("affected_components"):
                console.print(f"Affected: {', '.join(rc['affected_components'])}")
        
        # Remediation
        if result.get("remediation"):
            rem = result["remediation"]
            console.print(f"\n[bold]Remediation Steps[/bold]")
            for i, action in enumerate(rem.get("actions", []), 1):
                console.print(f"{i}. {action['description']}")
                if action.get("command"):
                    console.print(f"   Command: [cyan]{action['command']}[/cyan]")
        
    except httpx.HTTPError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)


def _status_color(status: str) -> str:
    """Get color for status."""
    colors = {
        "init": "yellow",
        "planning": "yellow",
        "investigating": "blue",
        "analyzing": "blue",
        "summarizing": "cyan",
        "completed": "green",
        "failed": "red",
        "cancelled": "gray"
    }
    return colors.get(status.lower(), "white")


if __name__ == "__main__":
    cli()