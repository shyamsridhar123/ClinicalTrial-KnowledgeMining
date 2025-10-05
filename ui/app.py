"""Gradio UI for DocIntel Clinical Trial Knowledge Mining Platform."""

import gradio as gr
import asyncio
import json
import traceback
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from docintel import ingest, parse, embed
from docintel.health.runner import HealthCheckRunner


def format_health_check_results(results, summary):
    """Format health check results as HTML."""
    status_symbols = {
        "healthy": "✅",
        "warning": "⚠️",
        "error": "❌",
        "unknown": "❓"
    }
    
    html = "<div style='font-family: monospace; padding: 20px; background: #1a1a1a; color: #f0f0f0; border-radius: 8px;'>"
    html += "<h2 style='color: #4a9eff;'>📊 System Health Check</h2>"
    html += "<hr style='border-color: #444;'>"
    
    for result in results:
        symbol = status_symbols.get(result.status.value, "❓")
        color = {"healthy": "#4ade80", "warning": "#fbbf24", "error": "#ef4444"}.get(result.status.value, "#888")
        
        html += f"<div style='margin: 15px 0; padding: 10px; background: #2a2a2a; border-radius: 5px;'>"
        html += f"<div style='font-size: 16px; font-weight: bold; color: {color};'>{symbol} {result.name}</div>"
        html += f"<div style='margin-left: 30px; margin-top: 5px; color: #ccc;'>{result.message}</div>"
        
        if result.details:
            html += "<div style='margin-left: 30px; margin-top: 8px; font-size: 12px; color: #999;'>"
            for key, value in list(result.details.items())[:5]:
                if isinstance(value, dict):
                    html += f"<div><strong>{key}:</strong></div>"
                    for k, v in list(value.items())[:3]:
                        html += f"<div style='margin-left: 15px;'>• {k}: {v}</div>"
                elif not isinstance(value, list):
                    html += f"<div>• {key}: {value}</div>"
            html += "</div>"
        
        html += f"<div style='margin-left: 30px; margin-top: 5px; font-size: 11px; color: #666;'>⏱ {result.duration_ms:.0f}ms</div>"
        html += "</div>"
    
    # Summary
    overall_symbol = status_symbols.get(summary['overall_status'].value, "❓")
    overall_color = {"healthy": "#4ade80", "warning": "#fbbf24", "error": "#ef4444"}.get(summary['overall_status'].value, "#888")
    
    html += "<hr style='border-color: #444; margin-top: 20px;'>"
    html += f"<div style='font-size: 18px; font-weight: bold; color: {overall_color}; margin-top: 15px;'>"
    html += f"{overall_symbol} Overall Status: {summary['overall_status'].value.upper()}</div>"
    html += f"<div style='margin-top: 10px; color: #ccc;'>"
    html += f"✅ Healthy: {summary['healthy']}/{summary['total']} &nbsp;&nbsp; "
    html += f"⚠️ Warning: {summary['warning']}/{summary['total']} &nbsp;&nbsp; "
    html += f"❌ Error: {summary['error']}/{summary['total']}"
    html += "</div></div>"
    
    return html


async def run_health_check():
    """Run system health check."""
    runner = HealthCheckRunner()
    results = await runner.run_all()
    summary = runner.summarize(results)
    return format_health_check_results(results, summary)


def run_ingestion(max_studies):
    """Run document ingestion."""
    try:
        report = ingest.run(max_studies=max_studies)
        return f"✅ Ingestion completed!\n\n{json.dumps(report, indent=2)}"
    except Exception as e:
        return f"❌ Ingestion failed: {str(e)}"


def run_parsing(force_reparse, max_workers):
    """Run document parsing."""
    try:
        report = parse.run(
            force_reparse=force_reparse,
            max_workers=max_workers if max_workers else None
        )
        return f"✅ Parsing completed!\n\n{json.dumps(report, indent=2)}"
    except Exception as e:
        return f"❌ Parsing failed: {str(e)}"


def run_embedding(force_reembed, batch_size):
    """Run embedding generation."""
    try:
        report = embed.run(
            force_reembed=force_reembed,
            batch_size=batch_size if batch_size else None
        )
        return f"✅ Embedding completed!\n\n{json.dumps(report, indent=2)}"
    except Exception as e:
        return f"❌ Embedding failed: {str(e)}"


def run_graph_stats():
    """Get knowledge graph statistics."""
    try:
        import psycopg
        from psycopg.rows import dict_row
        
        conn = psycopg.connect("postgresql://dbuser:dbpass123@localhost:5432/docintel")
        
        with conn.cursor(row_factory=dict_row) as cur:
            # Get comprehensive statistics in one query
            cur.execute("""
                SELECT 
                    (SELECT COUNT(*) FROM docintel.entities) as total_entities,
                    (SELECT COUNT(*) FROM docintel.relations) as total_relations,
                    (SELECT COUNT(*) FROM docintel.embeddings) as total_embeddings,
                    (SELECT COUNT(DISTINCT nct_id) FROM docintel.embeddings) as unique_nct_studies,
                    (SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type = 'figure_image') as figure_embeddings,
                    (SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type = 'figure_caption') as figure_captions,
                    (SELECT COUNT(*) FROM docintel.embeddings WHERE artefact_type = 'table') as table_embeddings,
                    (SELECT COUNT(*) FROM docintel.meta_graphs) as meta_graphs,
                    (SELECT COUNT(*) FROM docintel.repo_nodes) as vocabulary_nodes
            """)
            stats = cur.fetchone()
        
        conn.close()
        
        # Format results nicely
        result = {
            "📊 Core Knowledge Graph": {
                "Entities": f"{stats['total_entities']:,}",
                "Relations": f"{stats['total_relations']:,}",
                "Meta Graphs": f"{stats['meta_graphs']:,}"
            },
            "🔍 Semantic Search": {
                "Total Embeddings": f"{stats['total_embeddings']:,}",
                "NCT Studies": f"{stats['unique_nct_studies']:,}",
                "Figure Images": f"{stats['figure_embeddings']:,}",
                "Figure Captions": f"{stats['figure_captions']:,}",
                "Table Embeddings": f"{stats['table_embeddings']:,}"
            },
            "📚 Vocabulary Repository": {
                "Normalized Terms": f"{stats['vocabulary_nodes']:,}"
            }
        }
        
        return f"📊 **Knowledge Graph Statistics**\n\n```json\n{json.dumps(result, indent=2)}\n```"
    except Exception as e:
        return f"❌ Failed to get stats: {str(e)}\n\nLINE 1: {str(e)}"


async def run_semantic_search(query_text, max_results):
    """Run semantic search query."""
    if not query_text.strip():
        return "⚠️ Please enter a question"
    
    try:
        # Import query_clinical_trials
        import query_clinical_trials
        
        qa = query_clinical_trials.ClinicalTrialQA()
        await qa.initialize()
        result = await qa.answer_question(query_text, top_k=max_results)
        
        # Format results
        if result and 'answer' in result:
            output = f"### Answer\n{result['answer']}\n\n"
            
            # Show metadata
            output += "### Query Statistics\n"
            output += f"- Entities found: {result.get('entities_found', 0)}\n"
            output += f"- Graph expansions: {result.get('graph_expanded_count', 0)}\n"
            output += f"- NCTs searched: {', '.join(result.get('ncts_searched', []))}\n"
            output += f"- Model used: {result.get('model_used', 'unknown')}\n"
            output += f"- Routing reason: {result.get('routing_reason', 'unknown')}\n"
            output += f"- Processing time: {result.get('processing_time_ms', 0):.2f} ms\n\n"
            
            if 'sources' in result and result['sources']:
                output += "### Source Documents\n"
                for source in result['sources'][:10]:  # Show top 10
                    output += f"- **{source['document']}** (NCT: {source['nct_id']}, Similarity: {source['similarity']}, Entities: {source['entities']})\n"
        
        return output
    except Exception as e:
        return f"❌ Query failed: {str(e)}\n\n{traceback.format_exc()}"


def run_context_tests():
    """Run context-aware extraction tests."""
    try:
        import subprocess
        test_script = Path(__file__).parent.parent / "test_context_aware_extraction.py"
        
        if not test_script.exists():
            return f"❌ Test script not found: {test_script}"
        
        result = subprocess.run(
            ["pixi", "run", "python", str(test_script)],
            capture_output=True,
            text=True,
            cwd=str(test_script.parent)
        )
        
        output = result.stdout if result.stdout else result.stderr
        
        if result.returncode == 0:
            return f"✅ Tests completed!\n\n{output}"
        else:
            return f"⚠️ Tests finished with exit code {result.returncode}\n\n{output}"
    except Exception as e:
        return f"❌ Test execution failed: {str(e)}"


# Create Gradio interface with sidebar navigation
with gr.Blocks(title="DocIntel - Clinical Trial Knowledge Mining", theme=gr.themes.Soft()) as app:
    gr.Markdown("""
    # 🏥 DocIntel - Clinical Trial Knowledge Mining Platform
    
    A comprehensive system for extracting, analyzing, and querying clinical trial knowledge with context-aware AI.
    """)
    
    with gr.Row():
        # Left sidebar navigation
        with gr.Column(scale=1, min_width=220):
            gr.Markdown("### 📋 Navigation")
            
            search_nav = gr.Button("🔍 Semantic Search", variant="primary", size="lg")
            health_nav = gr.Button("📊 System Health", size="lg")
            ingest_nav = gr.Button("📥 Data Ingestion", size="lg")
            parse_nav = gr.Button("📄 Document Parsing", size="lg")
            embed_nav = gr.Button("🧠 Generate Embeddings", size="lg")
            stats_nav = gr.Button("📈 Graph Statistics", size="lg")
            test_nav = gr.Button("🧪 Context Tests", size="lg")
        
        # Main content area
        with gr.Column(scale=4):
            # Section 1: Semantic Search (default)
            with gr.Group(visible=True) as search_section:
                gr.Markdown("## 🔍 Semantic Search")
                gr.Markdown("""
                Ask questions about clinical trials with context-aware Q&A and intelligent model routing.
                
                **GPT-4.1** for simple queries | **GPT-5-mini** for statistical analysis and reasoning
                """)
                
                query_text = gr.Textbox(
                    label="Your Question",
                    placeholder="What are the observed statistics for niraparib efficacy?",
                    lines=2
                )
                
                max_results = gr.Slider(
                    minimum=1, maximum=100, value=50, step=1,
                    label="Maximum Results"
                )
                
                search_btn = gr.Button("Search", variant="primary", size="lg")
                search_output = gr.Markdown()
                
                search_btn.click(
                    fn=lambda q, m: asyncio.run(run_semantic_search(q, m)),
                    inputs=[query_text, max_results],
                    outputs=search_output
                )
                
                gr.Markdown("""
                #### Context Flags:
                - ❌ **NEGATED**: "no evidence of X"
                - 📅 **HISTORICAL**: past conditions
                - 🤔 **HYPOTHETICAL**: potential scenarios
                - ❓ **UNCERTAIN**: "possible", "may be"
                - 👨‍👩‍👧 **FAMILY**: family history
                """)
            
            # Section 2: System Health
            with gr.Group(visible=False) as health_section:
                gr.Markdown("## 📊 System Health")
                gr.Markdown("### Check system status and connectivity")
                
                health_check_btn = gr.Button("Run Health Check", variant="primary", size="lg")
                health_output = gr.HTML()
                
                health_check_btn.click(
                    fn=lambda: asyncio.run(run_health_check()),
                    outputs=health_output
                )
            
            # Section 3: Data Ingestion
            with gr.Group(visible=False) as ingest_section:
                gr.Markdown("## 📥 Data Ingestion")
                gr.Markdown("### Download clinical trial documents from ClinicalTrials.gov")
                
                max_studies = gr.Slider(
                    minimum=1, maximum=100, value=25, step=1,
                    label="Maximum Studies to Download"
                )
                
                ingest_btn = gr.Button("Start Ingestion", variant="primary")
                ingest_output = gr.Textbox(label="Ingestion Results", lines=15)
                
                ingest_btn.click(
                    fn=run_ingestion,
                    inputs=max_studies,
                    outputs=ingest_output
                )
            
            # Section 4: Document Parsing
            with gr.Group(visible=False) as parse_section:
                gr.Markdown("## 📄 Document Parsing")
                gr.Markdown("### Parse documents with Docling (GPU-accelerated)")
                
                with gr.Row():
                    force_reparse = gr.Checkbox(label="Force Reparse", value=False)
                    max_workers = gr.Number(label="Max Workers (optional)", value=None, precision=0)
                
                parse_btn = gr.Button("Start Parsing", variant="primary")
                parse_output = gr.Textbox(label="Parsing Results", lines=15)
                
                parse_btn.click(
                    fn=run_parsing,
                    inputs=[force_reparse, max_workers],
                    outputs=parse_output
                )
            
            # Section 5: Generate Embeddings
            with gr.Group(visible=False) as embed_section:
                gr.Markdown("## 🧠 Generate Embeddings")
                gr.Markdown("### Create semantic embeddings with BiomedCLIP")
                
                with gr.Row():
                    force_reembed = gr.Checkbox(label="Force Re-embed", value=False)
                    batch_size = gr.Number(label="Batch Size (optional)", value=None, precision=0)
                
                embed_btn = gr.Button("Generate Embeddings", variant="primary")
                embed_output = gr.Textbox(label="Embedding Results", lines=15)
                
                embed_btn.click(
                    fn=run_embedding,
                    inputs=[force_reembed, batch_size],
                    outputs=embed_output
                )
            
            # Section 6: Graph Statistics
            with gr.Group(visible=False) as stats_section:
                gr.Markdown("## 📈 Graph Statistics")
                gr.Markdown("### View knowledge graph metrics")
                
                stats_btn = gr.Button("Get Statistics", variant="primary")
                stats_output = gr.Textbox(label="Graph Statistics", lines=20)
                
                stats_btn.click(
                    fn=run_graph_stats,
                    outputs=stats_output
                )
            
            # Section 7: Context Tests
            with gr.Group(visible=False) as test_section:
                gr.Markdown("## 🧪 Context Tests")
                gr.Markdown("""
                ### Validate context-aware extraction
                
                Tests negation, historical conditions, hypothetical scenarios, family history, and uncertainty detection.
                """)
                
                test_btn = gr.Button("Run Tests", variant="primary")
                test_output = gr.Textbox(label="Test Results", lines=20)
                
                test_btn.click(
                    fn=run_context_tests,
                    outputs=test_output
                )
    
    # Navigation logic
    def show_search():
        return {
            search_section: gr.update(visible=True),
            health_section: gr.update(visible=False),
            ingest_section: gr.update(visible=False),
            parse_section: gr.update(visible=False),
            embed_section: gr.update(visible=False),
            stats_section: gr.update(visible=False),
            test_section: gr.update(visible=False)
        }
    
    def show_health():
        return {
            search_section: gr.update(visible=False),
            health_section: gr.update(visible=True),
            ingest_section: gr.update(visible=False),
            parse_section: gr.update(visible=False),
            embed_section: gr.update(visible=False),
            stats_section: gr.update(visible=False),
            test_section: gr.update(visible=False)
        }
    
    def show_ingest():
        return {
            search_section: gr.update(visible=False),
            health_section: gr.update(visible=False),
            ingest_section: gr.update(visible=True),
            parse_section: gr.update(visible=False),
            embed_section: gr.update(visible=False),
            stats_section: gr.update(visible=False),
            test_section: gr.update(visible=False)
        }
    
    def show_parse():
        return {
            search_section: gr.update(visible=False),
            health_section: gr.update(visible=False),
            ingest_section: gr.update(visible=False),
            parse_section: gr.update(visible=True),
            embed_section: gr.update(visible=False),
            stats_section: gr.update(visible=False),
            test_section: gr.update(visible=False)
        }
    
    def show_embed():
        return {
            search_section: gr.update(visible=False),
            health_section: gr.update(visible=False),
            ingest_section: gr.update(visible=False),
            parse_section: gr.update(visible=False),
            embed_section: gr.update(visible=True),
            stats_section: gr.update(visible=False),
            test_section: gr.update(visible=False)
        }
    
    def show_stats():
        return {
            search_section: gr.update(visible=False),
            health_section: gr.update(visible=False),
            ingest_section: gr.update(visible=False),
            parse_section: gr.update(visible=False),
            embed_section: gr.update(visible=False),
            stats_section: gr.update(visible=True),
            test_section: gr.update(visible=False)
        }
    
    def show_tests():
        return {
            search_section: gr.update(visible=False),
            health_section: gr.update(visible=False),
            ingest_section: gr.update(visible=False),
            parse_section: gr.update(visible=False),
            embed_section: gr.update(visible=False),
            stats_section: gr.update(visible=False),
            test_section: gr.update(visible=True)
        }
    
    # Wire navigation buttons
    search_nav.click(show_search, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    health_nav.click(show_health, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    ingest_nav.click(show_ingest, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    parse_nav.click(show_parse, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    embed_nav.click(show_embed, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    stats_nav.click(show_stats, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    test_nav.click(show_tests, outputs=[search_section, health_section, ingest_section, parse_section, embed_section, stats_section, test_section])
    
    # Footer
    # gr.Markdown("""
    # ---
    # 
    # ### 📚 Documentation
    # - **Demo Questions**: See `DEMO_QUESTIONS.md`
    # - **Architecture**: See `docs/Clinical_Trial_Knowledge_Mining_TRD_Modular.md`
    # 
    # ### ⚡ Current Status
    # - **Model Routing**: GPT-4.1 ⟷ GPT-5-mini
    # - **Entities**: 37,657 extracted
    # - **Relations**: 5,266 identified
    # - **Embeddings**: 3,735 chunks + 212 figures (BiomedCLIP)
    # """)


if __name__ == "__main__":
    app.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
