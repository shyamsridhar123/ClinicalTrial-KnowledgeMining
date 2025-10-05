"""Interactive command-line interface for DocIntel pipelines."""

from __future__ import annotations

import json
import sys
import textwrap
import traceback
from typing import Any, Dict, Optional

from . import embed, ingest, parse

_PROMPT_BANNER = textwrap.dedent(
    """
    ================================================================================
                      Clinical Trial Knowledge Mining Toolkit
    ================================================================================
    
    📥 INGESTION & PROCESSING:
      1.  Download clinical trial documents from ClinicalTrials.gov
      2.  Parse documents (extract text, tables, figures with Docling)
      3.  Generate embeddings (BiomedCLIP + pgvector)
    
    🧠 KNOWLEDGE EXTRACTION:
      4.  Extract entities & relations (GPT-4.1 + medspaCy + context-aware NLP)
      5.  Build knowledge graph (PostgreSQL + AGE graph database)
      6.  View graph statistics (nodes, edges, entity types)
    
    🔍 QUERY & SEARCH:
      7.  Semantic search (Ask questions about clinical trials)
      8.  Advanced graph query (Cypher + U-Retrieval)
      9.  Test context-aware extraction (negation, historical, etc.)
    
    🔧 UTILITIES:
      10. Run full pipeline (1→2→3→4→5)
      11. Show system status & configuration
      12. Show last run summaries
      
      0.  Exit
    ================================================================================
    """
)


def _prompt(message: str) -> str:
    try:
        return input(message)
    except EOFError:
        print("\n[EOF detected, exiting...]")
        sys.exit(0)
    except KeyboardInterrupt:
        print("\n[Interrupted, exiting...]")
        sys.exit(0)


def _prompt_int(prompt: str, *, default: int, minimum: Optional[int] = None) -> int:
    while True:
        raw = _prompt(f"{prompt} [{default}]: ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be ≥ {minimum}.")
            continue
        return value


def _prompt_optional_int(prompt: str, *, default: Optional[int] = None, minimum: Optional[int] = None) -> Optional[int]:
    label = f"{prompt} [{'' if default is None else default}]: "
    while True:
        raw = _prompt(label).strip()
        if not raw:
            return default
        if raw.lower() in {"none", "null"}:
            return None
        try:
            value = int(raw)
        except ValueError:
            print("Please enter an integer, 'none', or press enter for default.")
            continue
        if minimum is not None and value < minimum:
            print(f"Value must be ≥ {minimum}.")
            continue
        return value


def _prompt_bool(prompt: str, *, default: bool = False) -> bool:
    suffix = "Y/n" if default else "y/N"
    while True:
        raw = _prompt(f"{prompt} [{suffix}]: ").strip().lower()
        if not raw:
            return default
        if raw in {"y", "yes", "1", "true"}:
            return True
        if raw in {"n", "no", "0", "false"}:
            return False
        print("Please respond with 'y' or 'n'.")


def _prompt_choice(prompt: str, choices: Dict[str, str], *, default: Optional[str] = None) -> Optional[str]:
    choice_display = ", ".join(f"{key}={label}" for key, label in choices.items())
    message = f"{prompt} ({choice_display})"
    if default is not None:
        message += f" [{default}]"
    message += ": "
    while True:
        raw = _prompt(message).strip().lower()
        if not raw:
            return default
        if raw in choices:
            return raw
        if raw in {"none", "null"}:
            return None
        print(f"Please choose one of: {', '.join(choices)} or press enter for default.")


def _print_report(name: str, report: Dict[str, Any]) -> None:
    print(f"\n{name} report:")
    try:
        formatted = json.dumps(report, indent=2, sort_keys=True, default=str)
        print(formatted)
    except TypeError:
        print(report)
    print()


def _run_ingestion_interactive() -> Dict[str, Any]:
    max_studies = _prompt_int("Maximum studies to ingest", default=25, minimum=1)
    print("\n➡️  Running ingestion...\n")
    report = ingest.run(max_studies=max_studies)
    _print_report("Ingestion", report)
    return report


def _run_parsing_interactive() -> Dict[str, Any]:
    force_reparse = _prompt_bool("Force reparse existing documents?", default=False)
    max_workers = _prompt_optional_int("Maximum concurrent parsing workers", default=None, minimum=1)
    print("\n➡️  Running parsing...\n")
    report = parse.run(force_reparse=force_reparse, max_workers=max_workers)
    _print_report("Parsing", report)
    return report


def _run_embedding_interactive() -> Dict[str, Any]:
    force_reembed = _prompt_bool("Force re-embed existing chunks?", default=False)
    batch_size = _prompt_optional_int("Embedding batch size", default=None, minimum=1)
    quant_choice = _prompt_choice(
        "Quantization encoding",
        {"none": "float32", "bfloat16": "bfloat16", "int8": "int8"},
        default=None,
    )
    store_float32 = None
    if quant_choice and quant_choice != "none":
        store_float32 = _prompt_bool("Store float32 copies alongside quantized payloads?", default=True)
    print("\n➡️  Running embedding...\n")
    report = embed.run(
        force_reembed=force_reembed,
        batch_size=batch_size,
        quantization_encoding=quant_choice,
        store_float32=store_float32,
    )
    _print_report("Embedding", report)
    return report


def _run_full_pipeline() -> Dict[str, Dict[str, Any]]:
    results: Dict[str, Dict[str, Any]] = {}
    try:
        results["ingestion"] = _run_ingestion_interactive()
        results["parsing"] = _run_parsing_interactive()
        results["embedding"] = _run_embedding_interactive()
    except Exception:
        print("An error occurred while running the pipeline. See traceback below:\n")
        traceback.print_exc()
    return results


def _run_extraction_interactive() -> Dict[str, Any]:
    """Run entity extraction without prompts."""
    print("\n➡️  Running clinical entity extraction...\n")
    
    from . import extract
    report = extract.run(nct_id=None, limit=5, output_file="logs/extraction_results.json")
    _print_report("Entity Extraction", report)
    return report


def _run_graph_construction_interactive() -> Dict[str, Any]:
    """Run knowledge graph construction without prompts."""
    extraction_file = "logs/extraction_results.json"
    clear_existing = False
    
    print("\n➡️  Building knowledge graph...\n")
    print(f"Using extraction file: {extraction_file}")
    print(f"Clear existing data: {clear_existing}")
    
    from . import graph
    report = graph.run("build", extraction_file=extraction_file, clear_existing=clear_existing)
    _print_report("Graph Construction", report)
    return report


def _run_graph_stats_interactive() -> None:
    """Run knowledge graph statistics query without prompts."""
    print("\n� Knowledge Graph Statistics")
    print("=" * 50)
    
    from . import query
    
    print("\n➡️  Getting knowledge graph statistics...\n")
    result = query.run("stats")
    _print_report("Graph Statistics", result)


def _run_graph_query_interactive() -> None:
    """Advanced Cypher query interface for the knowledge graph."""
    print("\n🔍 Advanced Graph Query (Cypher)")
    print("=" * 50)
    
    cypher_query = _prompt("Enter Cypher query (or press Enter to skip): ").strip()
    if not cypher_query:
        print("No query entered. Returning to menu.\n")
        return
    
    print(f"\n➡️  Executing: {cypher_query}\n")
    
    try:
        from docintel.knowledge_graph.builder import KnowledgeGraphBuilder
        import psycopg
        from docintel.core.config import settings
        
        conn = psycopg.connect(settings.VECTOR_DB_DSN)
        builder = KnowledgeGraphBuilder(conn)
        
        # Execute Cypher query
        cursor = conn.cursor()
        cursor.execute(f"SELECT * FROM ag_catalog.cypher('clinical_graph', $$ {cypher_query} $$) as (result agtype);")
        results = cursor.fetchall()
        
        if results:
            print(f"✓ Found {len(results)} results:\n")
            for i, row in enumerate(results[:20], 1):  # Show first 20
                print(f"  {i}. {row[0]}")
            if len(results) > 20:
                print(f"  ... and {len(results) - 20} more")
        else:
            print("No results found.")
        
        cursor.close()
        conn.close()
        
    except Exception as e:
        print(f"✗ Query failed: {e}")
        traceback.print_exc()
    
    print()


def _run_semantic_search_interactive() -> None:
    """Interactive semantic Q&A using the enhanced context-aware query system."""
    print("\n🔍 Semantic Search (Context-Aware Q&A)")
    print("=" * 50)
    print("Ask questions about clinical trials. Context flags (negation,")
    print("historical, hypothetical, etc.) are automatically applied.\n")
    
    # Get query from user
    query_text = _prompt("Your question: ").strip()
    if not query_text:
        print("No question entered. Returning to menu.\n")
        return
    
    # Optional parameters
    max_results = _prompt_int("Maximum results", default=5, minimum=1)
    
    # Run the query using the enhanced query system
    print(f"\n➡️  Searching for: {query_text}")
    print(f"   Max results: {max_results}\n")
    
    try:
        # Import and run query_clinical_trials.py functionality
        import asyncio
        from pathlib import Path
        import sys
        
        # Add project root to path
        project_root = Path(__file__).parent.parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        # Import the query module
        import query_clinical_trials
        
        # Create QA instance and run query
        async def run_query():
            qa = query_clinical_trials.ClinicalTrialQA()
            await qa.initialize()
            return await qa.answer_question(query_text, top_k=max_results)
        
        # Run async query
        result = asyncio.run(run_query())
        
        # Display results
        print("\n" + "=" * 50)
        print("ANSWER:")
        print("=" * 50)
        print(result["answer"])
        print()
        
        # Display statistics
        print("=" * 50)
        print("QUERY STATISTICS:")
        print("=" * 50)
        print(f"  Entities found: {result.get('entities_found', 0)}")
        print(f"  Graph expansions: {result.get('graph_expanded_count', 0)}")
        print(f"  NCTs searched: {', '.join(result.get('ncts_searched', []))}")
        print(f"  Processing time: {result.get('processing_time_ms', 0):.2f} ms")
        print()
        
        if result.get("sources"):
            print("=" * 50)
            print("SOURCE DOCUMENTS:")
            print("=" * 50)
            for i, source in enumerate(result["sources"][:10], 1):
                print(f"  {i}. {source['document']} (NCT: {source['nct_id']})")
                print(f"     Similarity: {source['similarity']}, Entities: {source['entities']}")
            print()
        
        print("✓ Query complete.")
        
    except ImportError as e:
        print(f"✗ Failed to import query module: {e}")
        print("Make sure query_clinical_trials.py is in the project root.")
        traceback.print_exc()
    except Exception as e:
        print(f"✗ Query failed: {e}")
        traceback.print_exc()
    
    print()


def _run_context_test_interactive() -> None:
    """Run the context-aware extraction test suite."""
    print("\n🧪 Context-Aware Extraction Test Suite")
    print("=" * 50)
    print("Testing: negation, historical, hypothetical, family history, etc.\n")
    
    try:
        import subprocess
        from pathlib import Path
        
        # Find test script
        test_script = Path(__file__).parent.parent.parent / "test_context_aware_extraction.py"
        
        if not test_script.exists():
            print(f"✗ Test script not found: {test_script}")
            return
        
        print(f"➡️  Running: {test_script.name}\n")
        
        # Run test with pixi
        result = subprocess.run(
            ["pixi", "run", "python", str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(test_script.parent)
        )
        
        # Display output
        if result.stdout:
            print(result.stdout)
        if result.stderr and result.returncode != 0:
            print(f"[STDERR]\n{result.stderr}")
        
        if result.returncode == 0:
            print("\n✓ All tests completed successfully.")
        else:
            print(f"\n✗ Tests failed with exit code {result.returncode}")
        
    except Exception as e:
        print(f"✗ Test execution failed: {e}")
        traceback.print_exc()
    
    print()


def _show_system_status() -> None:
    """Show comprehensive system health check."""
    import asyncio
    from docintel.health.runner import HealthCheckRunner
    
    print("\n📊 DocIntel System Health Check")
    print("=" * 70)
    print("Running checks (this may take ~10 seconds)...\n")
    
    runner = HealthCheckRunner()
    results = asyncio.run(runner.run_all())
    summary = runner.summarize(results)
    
    # Display results
    status_symbols = {
        "healthy": "✅",
        "warning": "⚠️ ",
        "error": "❌",
        "unknown": "❓"
    }
    
    for result in results:
        symbol = status_symbols[result.status.value]
        print(f"{symbol} {result.name}")
        print(f"   {result.message}")
        
        # Show details
        if result.details:
            for key, value in result.details.items():
                if isinstance(value, dict):
                    print(f"   {key}:")
                    for k, v in value.items():
                        print(f"      • {k}: {v}")
                elif isinstance(value, list):
                    print(f"   {key}: {', '.join(str(v) for v in value)}")
                else:
                    print(f"   {key}: {value}")
        
        print(f"   ⏱  {result.duration_ms:.0f}ms")
        print()
    
    # Summary
    print("=" * 70)
    overall_symbol = status_symbols[summary['overall_status'].value]
    print(f"{overall_symbol} Overall Status: {summary['overall_status'].value.upper()}")
    print(f"   ✅ Healthy: {summary['healthy']}/{summary['total']}")
    print(f"   ⚠️  Warning: {summary['warning']}/{summary['total']}")
    print(f"   ❌ Error: {summary['error']}/{summary['total']}")
    print()


def main() -> None:
    last_reports: Dict[str, Dict[str, Any]] = {}
    while True:
        print(_PROMPT_BANNER)
        try:
            choice = _prompt("Enter option: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n👋 Bye!")
            sys.exit(0)
        if choice == "1":
            try:
                last_reports["ingestion"] = _run_ingestion_interactive()
            except Exception:
                print("Ingestion failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "2":
            try:
                last_reports["parsing"] = _run_parsing_interactive()
            except Exception:
                print("Parsing failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "3":
            try:
                last_reports["embedding"] = _run_embedding_interactive()
            except Exception:
                print("Embedding failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "4":
            try:
                last_reports["extraction"] = _run_extraction_interactive()
            except Exception:
                print("Knowledge extraction failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "5":
            try:
                last_reports["graph_construction"] = _run_graph_construction_interactive()
            except Exception:
                print("Graph construction failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "6":
            try:
                _run_graph_stats_interactive()
            except Exception:
                print("Graph statistics failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "7":
            try:
                _run_semantic_search_interactive()
            except Exception:
                print("Semantic search failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "8":
            try:
                _run_graph_query_interactive()
            except Exception:
                print("Graph query failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "9":
            try:
                _run_context_test_interactive()
            except Exception:
                print("Context test failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "10":
            pipeline_results = _run_full_pipeline()
            last_reports.update(pipeline_results)
        elif choice == "11":
            try:
                _show_system_status()
            except Exception:
                print("Status check failed. Traceback:\n")
                traceback.print_exc()
        elif choice == "12":
            if not last_reports:
                print("\n📄 No runs recorded yet in this session.\n")
            else:
                print(f"\n📊 Showing summaries for {len(last_reports)} completed operations:\n")
                for name, report in last_reports.items():
                    _print_report(name.capitalize(), report)
        elif choice in {"0", "q", "quit", "exit", ""}:
            print("👋 Bye!")
            sys.exit(0)
        else:
            print(f"Unknown option: '{choice}'. Please try again.\n")
            # If we're getting repeated unknown options (like from piped input), exit gracefully
            if not sys.stdin.isatty():
                print("[Non-interactive mode detected with invalid input, exiting...]")
                sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
