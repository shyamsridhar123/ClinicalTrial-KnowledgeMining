"""
Evaluation Metrics CLI

Command-line interface for running evaluation metrics on the clinical knowledge graph.
Supports comprehensive evaluation and individual component assessment.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional
import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from docintel.config import get_config

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from docintel.knowledge_graph.evaluation_metrics import (
    ClinicalEvaluationFramework,
    evaluate_clinical_knowledge_graph
)

console = Console()


@click.group()
def evaluation_cli():
    """Clinical Knowledge Graph Evaluation Metrics"""
    pass


@evaluation_cli.command()
@click.option('--connection-string', '-c', 
              default=None,
              show_default=False,
              help='Database connection string (defaults to DOCINTEL_DSN).')
@click.option('--output', '-o', 
              help='Output file for evaluation report (JSON)')
@click.option('--verbose', '-v', is_flag=True, 
              help='Enable verbose output')
def comprehensive(connection_string: str, output: Optional[str], verbose: bool):
    """Run comprehensive evaluation across all components"""
    
    console.print(Panel.fit(
        "🚀 Clinical Knowledge Graph Comprehensive Evaluation",
        style="bold blue"
    ))
    
    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO)
    
    resolved_dsn = connection_string or get_config().docintel_dsn
    
    async def run_evaluation():
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            
            task = progress.add_task("Running comprehensive evaluation...", total=None)
            
            try:
                report = await evaluate_clinical_knowledge_graph(
                    connection_string=resolved_dsn,
                    output_file=output
                )
                
                progress.update(task, description="✅ Evaluation completed!")
                
                # Display results
                display_comprehensive_results(report)
                
                if output:
                    console.print(f"\n📄 Detailed report saved to: [bold green]{output}[/bold green]")
                
            except Exception as e:
                progress.update(task, description="❌ Evaluation failed!")
                console.print(f"[bold red]Error: {e}[/bold red]")
                raise
    
    asyncio.run(run_evaluation())


@evaluation_cli.command()
@click.option('--connection-string', '-c',
              default=None,
              show_default=False,
              help='Database connection string (defaults to DOCINTEL_DSN).')
def entities(connection_string: str):
    """Evaluate entity extraction quality"""
    
    console.print(Panel.fit(
        "👥 Entity Extraction Evaluation",
        style="bold green"
    ))
    
    resolved_dsn = connection_string or get_config().docintel_dsn
    
    async def run_entity_evaluation():
        evaluator = ClinicalEvaluationFramework(resolved_dsn)
        await evaluator.connect()
        
        try:
            metrics = await evaluator.evaluate_entity_extraction()
            display_entity_results(metrics)
        finally:
            await evaluator.close()
    
    asyncio.run(run_entity_evaluation())


@evaluation_cli.command()
@click.option('--connection-string', '-c',
              default=None,
              show_default=False,
              help='Database connection string (defaults to DOCINTEL_DSN).')
def relations(connection_string: str):
    """Evaluate relation extraction quality"""
    
    console.print(Panel.fit(
        "🔗 Relation Extraction Evaluation",
        style="bold yellow"
    ))
    
    resolved_dsn = connection_string or get_config().docintel_dsn
    
    async def run_relation_evaluation():
        evaluator = ClinicalEvaluationFramework(resolved_dsn)
        await evaluator.connect()
        
        try:
            metrics = await evaluator.evaluate_relation_extraction()
            display_relation_results(metrics)
        finally:
            await evaluator.close()
    
    asyncio.run(run_relation_evaluation())


@evaluation_cli.command()
@click.option('--connection-string', '-c',
              default=None,
              show_default=False,
              help='Database connection string (defaults to DOCINTEL_DSN).')
def communities(connection_string: str):
    """Evaluate community detection quality"""
    
    console.print(Panel.fit(
        "🏘️ Community Detection Evaluation",
        style="bold purple"
    ))
    
    resolved_dsn = connection_string or get_config().docintel_dsn
    
    async def run_community_evaluation():
        evaluator = ClinicalEvaluationFramework(resolved_dsn)
        await evaluator.connect()
        
        try:
            metrics = await evaluator.evaluate_community_detection()
            display_community_results(metrics)
        finally:
            await evaluator.close()
    
    asyncio.run(run_community_evaluation())


@evaluation_cli.command()
@click.option('--connection-string', '-c',
              default=None,
              show_default=False,
              help='Database connection string (defaults to DOCINTEL_DSN).')
@click.option('--queries', '-q',
              help='JSON file with test queries')
def retrieval(connection_string: str, queries: Optional[str]):
    """Evaluate U-Retrieval system performance"""
    
    console.print(Panel.fit(
        "🔍 U-Retrieval System Evaluation",
        style="bold cyan"
    ))
    
    resolved_dsn = connection_string or get_config().docintel_dsn
    
    # Load test queries if provided
    test_queries = None
    if queries:
        try:
            with open(queries, 'r') as f:
                test_queries = json.load(f)
        except Exception as e:
            console.print(f"[bold red]Error loading queries file: {e}[/bold red]")
            return
    
    async def run_retrieval_evaluation():
        evaluator = ClinicalEvaluationFramework(resolved_dsn)
        await evaluator.connect()
        
        try:
            metrics = await evaluator.evaluate_retrieval_system(test_queries)
            display_retrieval_results(metrics)
        finally:
            await evaluator.close()
    
    asyncio.run(run_retrieval_evaluation())


def display_comprehensive_results(report):
    """Display comprehensive evaluation results"""
    
    # Overall scores
    overall_table = Table(title="🎯 Overall Evaluation Scores")
    overall_table.add_column("Metric", style="bold")
    overall_table.add_column("Score", justify="center")
    overall_table.add_column("Assessment", justify="center")
    
    def get_assessment(score: float) -> tuple:
        if score >= 0.8:
            return "🎉 Excellent", "bold green"
        elif score >= 0.6:
            return "✅ Good", "green"
        elif score >= 0.4:
            return "⚠️ Fair", "yellow"
        else:
            return "❌ Poor", "red"
    
    clinical_relevance = report.overall_clinical_relevance
    rag_compliance = report.medical_graph_rag_compliance
    
    assessment, style = get_assessment(clinical_relevance)
    overall_table.add_row("Clinical Relevance", f"{clinical_relevance:.3f}", assessment)
    
    assessment, style = get_assessment(rag_compliance)
    overall_table.add_row("Medical-Graph-RAG Compliance", f"{rag_compliance:.3f}", assessment)
    
    console.print(overall_table)
    console.print()
    
    # Component scores
    components_table = Table(title="📊 Component Evaluation Summary")
    components_table.add_column("Component", style="bold")
    components_table.add_column("Precision", justify="center")
    components_table.add_column("Recall", justify="center")
    components_table.add_column("F1-Score", justify="center")
    components_table.add_column("Quality Score", justify="center")
    
    # Entity metrics
    e_metrics = report.entity_metrics
    components_table.add_row(
        "👥 Entities",
        f"{e_metrics.precision:.3f}",
        f"{e_metrics.recall:.3f}",
        f"{e_metrics.f1_score:.3f}",
        f"{e_metrics.clinical_relevance_score:.3f}"
    )
    
    # Relation metrics
    r_metrics = report.relation_metrics
    components_table.add_row(
        "🔗 Relations",
        f"{r_metrics.precision:.3f}",
        f"{r_metrics.recall:.3f}",
        f"{r_metrics.f1_score:.3f}",
        f"{r_metrics.semantic_coherence:.3f}"
    )
    
    # Community metrics
    c_metrics = report.community_metrics
    components_table.add_row(
        "🏘️ Communities",
        f"{c_metrics.modularity:.3f}",
        f"{c_metrics.coverage:.3f}",
        f"{c_metrics.silhouette_score:.3f}",
        f"{c_metrics.clinical_clustering_quality:.3f}"
    )
    
    # Retrieval metrics
    ret_metrics = report.retrieval_metrics
    components_table.add_row(
        "🔍 Retrieval",
        f"{ret_metrics.precision_at_k.get(5, 0.0):.3f}",
        f"{ret_metrics.recall_at_k.get(5, 0.0):.3f}",
        f"{ret_metrics.ndcg_at_k.get(5, 0.0):.3f}",
        f"{ret_metrics.mean_reciprocal_rank:.3f}"
    )
    
    console.print(components_table)
    console.print()
    
    # Dataset info
    dataset_info = report.dataset_info
    dataset_panel = Panel(
        f"📈 Dataset Statistics\n\n" +
        f"Entities: {dataset_info['total_entities']:,}\n" +
        f"Relations: {dataset_info['total_relations']:,}\n" +
        f"Communities: {dataset_info['total_communities']:,}\n" +
        f"Documents: {dataset_info['total_chunks']:,}\n" +
        f"Entities per Document: {dataset_info['entities_per_chunk']:.1f}",
        title="Dataset Overview",
        expand=False
    )
    console.print(dataset_panel)


def display_entity_results(metrics):
    """Display entity evaluation results"""
    
    # Main metrics
    main_table = Table(title="👥 Entity Extraction Metrics")
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", justify="center")
    main_table.add_column("Description")
    
    main_table.add_row("Precision", f"{metrics.precision:.3f}", "Fraction of extracted entities that are correct")
    main_table.add_row("Recall", f"{metrics.recall:.3f}", "Fraction of actual entities that were extracted")
    main_table.add_row("F1-Score", f"{metrics.f1_score:.3f}", "Harmonic mean of precision and recall")
    main_table.add_row("Accuracy", f"{metrics.accuracy:.3f}", "Overall correctness")
    main_table.add_row("Coverage Rate", f"{metrics.coverage_rate:.1f}", "Entities per document")
    main_table.add_row("Clinical Relevance", f"{metrics.clinical_relevance_score:.3f}", "Domain-specific quality score")
    
    console.print(main_table)
    console.print()
    
    # Entity type breakdown
    if metrics.entity_type_breakdown:
        type_table = Table(title="📋 Entity Type Breakdown")
        type_table.add_column("Entity Type", style="bold")
        type_table.add_column("Count", justify="center")
        type_table.add_column("Percentage", justify="center")
        type_table.add_column("Avg Confidence", justify="center")
        type_table.add_column("Normalization Rate", justify="center")
        
        for entity_type, stats in sorted(metrics.entity_type_breakdown.items(), 
                                       key=lambda x: x[1]['count'], reverse=True):
            type_table.add_row(
                entity_type,
                str(stats['count']),
                f"{stats['percentage']:.1f}%",
                f"{stats['avg_confidence']:.3f}",
                f"{stats['normalization_rate']:.3f}"
            )
        
        console.print(type_table)


def display_relation_results(metrics):
    """Display relation evaluation results"""
    
    main_table = Table(title="🔗 Relation Extraction Metrics")
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", justify="center")
    main_table.add_column("Description")
    
    main_table.add_row("Precision", f"{metrics.precision:.3f}", "Fraction of extracted relations that are correct")
    main_table.add_row("Recall", f"{metrics.recall:.3f}", "Fraction of actual relations that were extracted")
    main_table.add_row("F1-Score", f"{metrics.f1_score:.3f}", "Harmonic mean of precision and recall")
    main_table.add_row("Graph Connectivity", f"{metrics.graph_connectivity:.3f}", "How well-connected the graph is")
    main_table.add_row("Semantic Coherence", f"{metrics.semantic_coherence:.3f}", "Clinical meaningfulness of relations")
    
    console.print(main_table)
    console.print()
    
    # Relation type breakdown
    if metrics.relation_type_breakdown:
        type_table = Table(title="📋 Relation Type Breakdown")
        type_table.add_column("Relation Type", style="bold")
        type_table.add_column("Count", justify="center")
        type_table.add_column("Percentage", justify="center")
        type_table.add_column("Avg Confidence", justify="center")
        
        for rel_type, stats in sorted(metrics.relation_type_breakdown.items(),
                                    key=lambda x: x[1]['count'], reverse=True):
            type_table.add_row(
                rel_type,
                str(stats['count']),
                f"{stats['percentage']:.1f}%",
                f"{stats['avg_confidence']:.3f}"
            )
        
        console.print(type_table)


def display_community_results(metrics):
    """Display community evaluation results"""
    
    main_table = Table(title="🏘️ Community Detection Metrics")
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", justify="center")
    main_table.add_column("Description")
    
    main_table.add_row("Number of Communities", str(metrics.num_communities), "Total communities detected")
    main_table.add_row("Average Community Size", f"{metrics.average_community_size:.1f}", "Mean entities per community")
    main_table.add_row("Modularity", f"{metrics.modularity:.3f}", "Community structure quality")
    main_table.add_row("Silhouette Score", f"{metrics.silhouette_score:.3f}", "Clustering separation quality")
    main_table.add_row("Coverage", f"{metrics.coverage:.3f}", "Fraction of entities in communities")
    main_table.add_row("Community Coherence", f"{metrics.community_coherence:.3f}", "Internal consistency")
    main_table.add_row("Clinical Clustering Quality", f"{metrics.clinical_clustering_quality:.3f}", "Domain-specific clustering assessment")
    
    console.print(main_table)


def display_retrieval_results(metrics):
    """Display retrieval evaluation results"""
    
    main_table = Table(title="🔍 U-Retrieval System Metrics")
    main_table.add_column("Metric", style="bold")
    main_table.add_column("Value", justify="center")
    main_table.add_column("Description")
    
    main_table.add_row("Mean Reciprocal Rank", f"{metrics.mean_reciprocal_rank:.3f}", "Average ranking quality")
    main_table.add_row("NDCG@5", f"{metrics.ndcg_at_k.get(5, 0.0):.3f}", "Normalized Discounted Cumulative Gain at 5")
    main_table.add_row("Precision@5", f"{metrics.precision_at_k.get(5, 0.0):.3f}", "Precision at top 5 results")
    main_table.add_row("Recall@5", f"{metrics.recall_at_k.get(5, 0.0):.3f}", "Recall at top 5 results")
    main_table.add_row("Average Query Time", f"{metrics.average_query_time:.1f}ms", "Response time performance")
    main_table.add_row("Clinical Relevance Correlation", f"{metrics.clinical_relevance_correlation:.3f}", "Alignment with clinical importance")
    
    console.print(main_table)
    console.print()
    
    # Ranking metrics breakdown
    ranking_table = Table(title="📈 Ranking Metrics by K")
    ranking_table.add_column("K", justify="center", style="bold")
    ranking_table.add_column("NDCG@K", justify="center")
    ranking_table.add_column("Precision@K", justify="center")
    ranking_table.add_column("Recall@K", justify="center")
    
    for k in [1, 5, 10]:
        ranking_table.add_row(
            str(k),
            f"{metrics.ndcg_at_k.get(k, 0.0):.3f}",
            f"{metrics.precision_at_k.get(k, 0.0):.3f}",
            f"{metrics.recall_at_k.get(k, 0.0):.3f}"
        )
    
    console.print(ranking_table)


if __name__ == "__main__":
    evaluation_cli()