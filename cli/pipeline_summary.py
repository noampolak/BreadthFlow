#!/usr/bin/env python3
"""
BreadthFlow Pipeline Summary & Analytics

Provides comprehensive visibility across all pipeline runs with:
- Run status and performance metrics
- Symbol processing statistics
- Earnings and financial insights
- Historical trends and patterns
- Exportable JSON/CSV reports
"""

import click
import json
import sqlite3
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PipelineRunSummary:
    """Summary of a single pipeline run"""
    run_id: str
    command: str
    status: str
    start_time: str
    end_time: Optional[str]
    duration: Optional[float]
    symbols_processed: int
    symbols_successful: int
    symbols_failed: int
    success_rate: float
    data_size_mb: float
    errors_count: int
    warnings_count: int
    metadata: Dict[str, Any]

@dataclass
class SystemSummary:
    """Overall system summary"""
    total_runs: int
    successful_runs: int
    failed_runs: int
    overall_success_rate: float
    total_symbols_processed: int
    unique_symbols: List[str]
    total_data_mb: float
    avg_duration: float
    last_24h_runs: int
    most_processed_symbols: List[Dict[str, int]]
    error_patterns: List[Dict[str, Any]]

class PipelineSummaryGenerator:
    """Generates comprehensive pipeline summaries from multiple data sources"""
    
    def __init__(self):
        self.dashboard_db_path = "dashboard.db"
        self.elasticsearch_url = "http://localhost:9200"
        self.minio_url = "http://localhost:9000"
        
    def get_sqlite_summary(self) -> List[PipelineRunSummary]:
        """Get pipeline runs from SQLite dashboard database"""
        runs = []
        
        try:
            conn = sqlite3.connect(self.dashboard_db_path)
            cursor = conn.cursor()
            
            # Get pipeline runs with logs
            cursor.execute('''
                SELECT pr.run_id, pr.command, pr.status, pr.start_time, 
                       pr.end_time, pr.duration, pr.metadata,
                       COUNT(CASE WHEN l.level = 'ERROR' THEN 1 END) as error_count,
                       COUNT(CASE WHEN l.level = 'WARN' THEN 1 END) as warn_count
                FROM pipeline_runs pr
                LEFT JOIN logs l ON pr.run_id = l.run_id
                GROUP BY pr.run_id
                ORDER BY pr.start_time DESC
            ''')
            
            for row in cursor.fetchall():
                run_id, command, status, start_time, end_time, duration, metadata_json, error_count, warn_count = row
                
                # Parse metadata
                metadata = {}
                if metadata_json:
                    try:
                        metadata = json.loads(metadata_json)
                    except:
                        metadata = {}
                
                # Extract symbol statistics
                symbols_processed = metadata.get('symbols_count', 0)
                symbols_successful = metadata.get('successful_symbols', 0)
                symbols_failed = metadata.get('failed_symbols', 0)
                
                # Calculate success rate
                success_rate = 0.0
                if symbols_processed > 0:
                    success_rate = (symbols_successful / symbols_processed) * 100
                
                runs.append(PipelineRunSummary(
                    run_id=run_id,
                    command=command,
                    status=status,
                    start_time=start_time,
                    end_time=end_time,
                    duration=duration or 0.0,
                    symbols_processed=symbols_processed,
                    symbols_successful=symbols_successful,
                    symbols_failed=symbols_failed,
                    success_rate=success_rate,
                    data_size_mb=metadata.get('total_size_mb', 0.0),
                    errors_count=error_count or 0,
                    warnings_count=warn_count or 0,
                    metadata=metadata
                ))
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error reading SQLite database: {e}")
            
        return runs
    
    def get_elasticsearch_summary(self) -> Dict[str, Any]:
        """Get additional metrics from Elasticsearch"""
        try:
            # Get aggregated data from Elasticsearch
            query = {
                "size": 0,
                "aggs": {
                    "total_runs": {
                        "filter": {"term": {"status.keyword": "started"}}
                    },
                    "commands": {
                        "terms": {"field": "command.keyword", "size": 20}
                    },
                    "symbols": {
                        "terms": {"field": "metadata.symbol.keyword", "size": 100}
                    },
                    "durations": {
                        "filter": {"exists": {"field": "duration"}},
                        "aggs": {
                            "avg_duration": {"avg": {"field": "duration"}},
                            "max_duration": {"max": {"field": "duration"}},
                            "min_duration": {"min": {"field": "duration"}}
                        }
                    },
                    "recent_activity": {
                        "filter": {
                            "range": {
                                "@timestamp": {
                                    "gte": "now-24h"
                                }
                            }
                        }
                    },
                    "error_patterns": {
                        "filter": {"term": {"level.keyword": "ERROR"}},
                        "aggs": {
                            "top_errors": {
                                "terms": {"field": "message.keyword", "size": 10}
                            }
                        }
                    }
                }
            }
            
            response = requests.post(
                f"{self.elasticsearch_url}/breadthflow-logs/_search",
                json=query,
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.warning(f"Elasticsearch query failed: {response.status_code}")
                
        except Exception as e:
            logger.warning(f"Could not connect to Elasticsearch: {e}")
            
        return {}
    
    def get_minio_data_summary(self) -> Dict[str, Any]:
        """Get data storage statistics from MinIO"""
        try:
            import boto3
            
            s3_client = boto3.client(
                's3',
                endpoint_url='http://minio:9000',
                aws_access_key_id='minioadmin',
                aws_secret_access_key='minioadmin',
                region_name='us-east-1'
            )
            
            # Get OHLCV data statistics
            response = s3_client.list_objects_v2(Bucket='breadthflow', Prefix='ohlcv/')
            
            symbols_data = {}
            total_size = 0
            total_files = 0
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    size = obj['Size']
                    modified = obj['LastModified']
                    
                    # Extract symbol from path
                    path_parts = key.split('/')
                    if len(path_parts) >= 2 and path_parts[1]:
                        symbol = path_parts[1]
                        if symbol not in symbols_data:
                            symbols_data[symbol] = {
                                'files': 0,
                                'size_mb': 0,
                                'latest_update': modified
                            }
                        
                        symbols_data[symbol]['files'] += 1
                        symbols_data[symbol]['size_mb'] += size / (1024*1024)
                        if modified > symbols_data[symbol]['latest_update']:
                            symbols_data[symbol]['latest_update'] = modified
                    
                    total_size += size
                    total_files += 1
            
            return {
                'total_symbols': len(symbols_data),
                'total_files': total_files,
                'total_size_mb': total_size / (1024*1024),
                'symbols_data': symbols_data
            }
            
        except Exception as e:
            logger.warning(f"Could not connect to MinIO: {e}")
            return {}
    
    def generate_system_summary(self, runs: List[PipelineRunSummary], es_data: Dict, minio_data: Dict) -> SystemSummary:
        """Generate overall system summary"""
        
        if not runs:
            return SystemSummary(
                total_runs=0, successful_runs=0, failed_runs=0, overall_success_rate=0.0,
                total_symbols_processed=0, unique_symbols=[], total_data_mb=0.0,
                avg_duration=0.0, last_24h_runs=0, most_processed_symbols=[],
                error_patterns=[]
            )
        
        # Calculate basic statistics
        total_runs = len(runs)
        successful_runs = len([r for r in runs if r.status == 'completed'])
        failed_runs = len([r for r in runs if r.status == 'failed'])
        overall_success_rate = (successful_runs / total_runs * 100) if total_runs > 0 else 0.0
        
        # Symbol statistics
        all_symbols = set()
        total_symbols_processed = 0
        symbol_counts = {}
        
        for run in runs:
            total_symbols_processed += run.symbols_processed
            
            # Extract symbols from metadata
            if 'symbols' in run.metadata:
                symbols = run.metadata['symbols']
                if isinstance(symbols, list):
                    for symbol in symbols:
                        all_symbols.add(symbol)
                        symbol_counts[symbol] = symbol_counts.get(symbol, 0) + 1
        
        # Most processed symbols
        most_processed = sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        most_processed_symbols = [{'symbol': symbol, 'count': count} for symbol, count in most_processed]
        
        # Duration statistics
        durations = [r.duration for r in runs if r.duration and r.duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0.0
        
        # Recent activity (last 24 hours)
        yesterday = datetime.now() - timedelta(days=1)
        last_24h_runs = len([r for r in runs if datetime.fromisoformat(r.start_time.replace('Z', '')) > yesterday])
        
        # Data size from MinIO
        total_data_mb = minio_data.get('total_size_mb', 0.0)
        
        # Error patterns from runs
        error_patterns = []
        error_runs = [r for r in runs if r.errors_count > 0]
        if error_runs:
            error_patterns = [
                {
                    'pattern': 'Runs with errors',
                    'count': len(error_runs),
                    'percentage': len(error_runs) / total_runs * 100
                }
            ]
        
        return SystemSummary(
            total_runs=total_runs,
            successful_runs=successful_runs,
            failed_runs=failed_runs,
            overall_success_rate=overall_success_rate,
            total_symbols_processed=total_symbols_processed,
            unique_symbols=list(all_symbols),
            total_data_mb=total_data_mb,
            avg_duration=avg_duration,
            last_24h_runs=last_24h_runs,
            most_processed_symbols=most_processed_symbols,
            error_patterns=error_patterns
        )
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate complete pipeline summary report"""
        
        print("🔍 Gathering pipeline data from multiple sources...")
        
        # Collect data from all sources
        runs = self.get_sqlite_summary()
        es_data = self.get_elasticsearch_summary()
        minio_data = self.get_minio_data_summary()
        
        # Generate system summary
        system_summary = self.generate_system_summary(runs, es_data, minio_data)
        
        # Create comprehensive report
        report = {
            "report_generated": datetime.now().isoformat(),
            "system_summary": asdict(system_summary),
            "recent_runs": [asdict(run) for run in runs[:20]],  # Last 20 runs
            "data_storage": minio_data,
            "elasticsearch_metrics": {
                "total_log_entries": es_data.get('hits', {}).get('total', {}).get('value', 0),
                "aggregations_available": bool(es_data.get('aggregations'))
            }
        }
        
        return report

@click.group()
def cli():
    """🚀 BreadthFlow Pipeline Summary & Analytics"""
    pass

@cli.command()
@click.option('--format', type=click.Choice(['json', 'table', 'detailed']), default='detailed', help='Output format')
@click.option('--save', help='Save report to file (e.g., report.json)')
@click.option('--last', type=int, help='Show only last N runs')
def status(format, save, last):
    """Show comprehensive pipeline status and summary"""
    
    generator = PipelineSummaryGenerator()
    
    try:
        print("📊 BreadthFlow Pipeline Summary")
        print("=" * 60)
        
        report = generator.generate_comprehensive_report()
        
        if format == 'json':
            output = json.dumps(report, indent=2, default=str)
            print(output)
            
        elif format == 'table':
            _display_table_summary(report)
            
        else:  # detailed
            _display_detailed_summary(report, last)
        
        # Save to file if requested
        if save:
            with open(save, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n💾 Report saved to: {save}")
        
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        click.echo(f"❌ Error: {str(e)}")

@cli.command()
@click.option('--symbol', help='Filter by specific symbol')
@click.option('--status', type=click.Choice(['completed', 'failed', 'running']), help='Filter by status')
@click.option('--days', type=int, default=7, help='Days to look back')
def analyze(symbol, status, days):
    """Analyze pipeline performance and trends"""
    
    generator = PipelineSummaryGenerator()
    report = generator.generate_comprehensive_report()
    
    print(f"📈 BreadthFlow Performance Analysis (Last {days} days)")
    print("=" * 60)
    
    runs = report['recent_runs']
    
    # Filter runs
    cutoff_date = datetime.now() - timedelta(days=days)
    filtered_runs = []
    
    for run in runs:
        run_date = datetime.fromisoformat(run['start_time'].replace('Z', ''))
        if run_date >= cutoff_date:
            if not status or run['status'] == status:
                if not symbol or symbol in run.get('metadata', {}).get('symbols', []):
                    filtered_runs.append(run)
    
    if not filtered_runs:
        print("❌ No runs found matching criteria")
        return
    
    # Performance metrics
    durations = [r['duration'] for r in filtered_runs if r['duration'] and r['duration'] > 0]
    success_rates = [r['success_rate'] for r in filtered_runs if r['success_rate'] >= 0]
    
    print(f"📊 Found {len(filtered_runs)} matching runs")
    print()
    
    if durations:
        print("⏱️ Duration Statistics:")
        print(f"   • Average: {sum(durations)/len(durations):.1f} seconds")
        print(f"   • Fastest: {min(durations):.1f} seconds")
        print(f"   • Slowest: {max(durations):.1f} seconds")
        print()
    
    if success_rates:
        print("🎯 Success Rate Statistics:")
        print(f"   • Average: {sum(success_rates)/len(success_rates):.1f}%")
        print(f"   • Best: {max(success_rates):.1f}%")
        print(f"   • Worst: {min(success_rates):.1f}%")
        print()
    
    # Top symbols processed
    symbol_counts = {}
    for run in filtered_runs:
        symbols = run.get('metadata', {}).get('symbols', [])
        if isinstance(symbols, list):
            for sym in symbols:
                symbol_counts[sym] = symbol_counts.get(sym, 0) + 1
    
    if symbol_counts:
        print("📈 Most Processed Symbols:")
        for sym, count in sorted(symbol_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"   • {sym}: {count} times")

@cli.command()
def export():
    """Export comprehensive data for external analysis"""
    
    generator = PipelineSummaryGenerator()
    report = generator.generate_comprehensive_report()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Export JSON report
    json_file = f"breadthflow_report_{timestamp}.json"
    with open(json_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    # Export CSV of runs
    csv_file = f"breadthflow_runs_{timestamp}.csv"
    runs_df = pd.DataFrame(report['recent_runs'])
    runs_df.to_csv(csv_file, index=False)
    
    print(f"📊 Exported comprehensive report:")
    print(f"   • JSON: {json_file}")
    print(f"   • CSV:  {csv_file}")
    print()
    print("💡 Use these files for:")
    print("   • Excel analysis and charts")
    print("   • Business intelligence tools")
    print("   • Custom data science analysis")
    print("   • Archival and compliance")

def _display_detailed_summary(report: Dict[str, Any], last: Optional[int]):
    """Display detailed summary in a nice format"""
    
    system = report['system_summary']
    
    print("🎯 System Overview:")
    print(f"   • Total Runs: {system['total_runs']}")
    print(f"   • Success Rate: {system['overall_success_rate']:.1f}%")
    print(f"   • Symbols Processed: {system['total_symbols_processed']}")
    print(f"   • Unique Symbols: {len(system['unique_symbols'])}")
    print(f"   • Data Stored: {system['total_data_mb']:.2f} MB")
    print(f"   • Avg Duration: {system['avg_duration']:.1f} seconds")
    print(f"   • Last 24h Activity: {system['last_24h_runs']} runs")
    print()
    
    # Top symbols
    if system['most_processed_symbols']:
        print("📈 Most Processed Symbols:")
        for symbol_data in system['most_processed_symbols'][:5]:
            print(f"   • {symbol_data['symbol']}: {symbol_data['count']} times")
        print()
    
    # Recent runs
    runs = report['recent_runs']
    if last:
        runs = runs[:last]
    
    print(f"📋 Recent Runs (showing {len(runs)}):")
    print()
    
    for i, run in enumerate(runs, 1):
        status_emoji = {
            'completed': '✅',
            'failed': '❌', 
            'running': '🔄'
        }.get(run['status'], '❓')
        
        start_time = datetime.fromisoformat(run['start_time'].replace('Z', '')).strftime('%m-%d %H:%M')
        
        print(f"{i:2}. {status_emoji} {run['command'][:40]:<40} | {start_time} | {run['duration']:>5.1f}s | {run['symbols_processed']:>2} symbols | {run['success_rate']:>5.1f}%")
    
    print()
    print("💡 Use --format json for machine-readable output")
    print("💡 Use --save filename.json to export data")

def _display_table_summary(report: Dict[str, Any]):
    """Display summary in table format"""
    
    runs = report['recent_runs'][:10]  # Top 10 recent runs
    
    print("Run ID       | Command           | Status    | Duration | Symbols | Success%")
    print("-" * 75)
    
    for run in runs:
        run_id = run['run_id'][:8] + '...'
        command = run['command'][:15] + '...' if len(run['command']) > 15 else run['command']
        status = run['status'][:8]
        duration = f"{run['duration']:.1f}s"
        symbols = str(run['symbols_processed'])
        success_rate = f"{run['success_rate']:.1f}%"
        
        print(f"{run_id:<12} | {command:<17} | {status:<9} | {duration:<8} | {symbols:<7} | {success_rate}")

if __name__ == '__main__':
    cli()
