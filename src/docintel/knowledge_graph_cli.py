"""CLI entrypoint for knowledge graph operations."""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any, Dict, Optional

from .knowledge_graph.triple_extraction import ClinicalTripleExtractor
from .knowledge_graph.community_detection import CommunityDetector
from .knowledge_graph.evaluation_metrics import ClinicalEvaluationFramework, evaluate_clinical_knowledge_graph
from .knowledge_graph.enhanced_extraction import extract_and_normalize_clinical_data
from .config import get_config
import psycopg


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


async def _process_real_embeddings() -> Dict[str, Any]:
    """Process clinical trial chunk files produced by the parsing pipeline."""
    from pathlib import Path
    import json
    from uuid import uuid4
    from .knowledge_graph.enhanced_extraction import extract_and_normalize_clinical_data
    
    print("🔍 Processing REAL clinical trial chunk files...")
    
    # Find actual chunk files that were processed into embeddings
    chunk_base_path = Path("data/processing/chunks")
    if not chunk_base_path.exists():
        print(f"❌ Chunk directory not found: {chunk_base_path}")
        return {"status": "failed", "error": f"Directory not found: {chunk_base_path}"}
    
    chunk_files = list(chunk_base_path.glob("**/*.json"))
    if not chunk_files:
        print(f"❌ No chunk files found in {chunk_base_path}")
        return {"status": "failed", "error": f"No chunk files found in {chunk_base_path}"}
    
    print(f"📁 Found {len(chunk_files)} chunk files to process")
    
    total_entities = 0
    total_relations = 0
    total_chunks = 0
    processed_files = 0
    
    # Process every available chunk file
    for chunk_file in chunk_files:
        print(f"📝 Processing {chunk_file.name}...")
        
        try:
            # Load chunk data
            with chunk_file.open('r', encoding='utf-8') as f:
                chunk_data = json.load(f)
            
            # Extract chunks from file
            chunks = chunk_data.get("chunks", []) if isinstance(chunk_data, dict) else chunk_data
            if not chunks:
                print(f"  ⚠️ No chunks found in {chunk_file.name}")
                continue
            
            file_entities = 0
            file_relations = 0
            
            # Process each chunk in the file
            for i, chunk in enumerate(chunks, start=1):
                if not isinstance(chunk, dict):
                    continue
                
                chunk_text = chunk.get("text", "")
                if not chunk_text or len(chunk_text.strip()) < 50:
                    continue
                
                # Generate a unique chunk ID
                chunk_id = uuid4()
                
                # Use the sophisticated extraction system on REAL clinical text
                print(f"  🧬 Extracting from chunk {i} (length: {len(chunk_text)} chars)")
                
                try:
                    result = await extract_and_normalize_clinical_data(chunk_text, chunk_id)
                    
                    file_entities += len(result.entities)
                    file_relations += len(result.relations)
                    total_chunks += 1
                    
                    print(f"    ✅ Extracted {len(result.entities)} entities, {len(result.relations)} relations")
                    
                    # Show some sample entities
                    if result.entities:
                        sample_entities = result.entities[:3]
                        entity_samples = [f"{e.entity_type}:{e.text}" for e in sample_entities]
                        print(f"    📋 Sample entities: {', '.join(entity_samples)}")
                    
                except Exception as e:
                    print(f"    ❌ Failed to process chunk {i}: {e}")
                    continue
            
            total_entities += file_entities
            total_relations += file_relations
            processed_files += 1
            
            print(f"  📊 {chunk_file.name}: {file_entities} entities, {file_relations} relations")
            
        except Exception as e:
            print(f"❌ Failed to process file {chunk_file.name}: {e}")
            continue
    
    print(f"\n✅ Real clinical document processing complete!")
    print(f"📈 Summary:")
    print(f"  Files processed: {processed_files}/{len(chunk_files)}")
    print(f"  Chunks processed: {total_chunks}")
    print(f"  Total entities extracted: {total_entities}")
    print(f"  Total relations extracted: {total_relations}")
    print(f"  Available chunk files: {len(chunk_files)}")
    
    return {
        "status": "real_processing_complete", 
        "files_processed": processed_files,
        "chunks_processed": total_chunks,
        "entities_extracted": total_entities,
        "relations_extracted": total_relations,
        "source": "real_clinical_trial_chunks",
        "chunk_files_available": len(chunk_files)
    }


async def _run_entity_extraction() -> Dict[str, Any]:
    """Run enhanced entity and relation extraction with normalization on processed documents"""
    # Extract entities and relations from all processed documents using the real system
    print("🔍 Starting enhanced entity extraction with normalization from embeddings...")
    
    # Process real embeddings exported by the parsing pipeline
    return await _process_real_embeddings()


async def _run_community_detection() -> Dict[str, Any]:
    """Run community detection on the knowledge graph"""
    config = get_config()
    connection_string = config.docintel_dsn
    
    detector = CommunityDetector(connection_string)
    
    try:
        communities = await detector.run_community_detection(
            max_cluster_size=10,
            random_seed=0xDEADBEEF
        )
        
        return {
            "status": "complete",
            "communities_created": len(communities),
            "levels": len(set(c.level for c in communities.values()))
        }
        
    except Exception as e:
        logging.error(f"Community detection failed: {e}")
        return {"status": "failed", "error": str(e)}


async def _run_sync_age() -> Dict[str, Any]:
    """Sync extracted entities and relations to Apache AGE property graph"""
    from scripts.production_age_sync import ProductionAGESync
    
    connection_string = get_config().docintel_dsn
    
    sync = ProductionAGESync(connection_string)
    
    try:
        await sync.connect()
        
        # Run sync
        entity_result = await sync.sync_entities_chunked()
        relation_result = await sync.sync_relations_chunked()
        
        # Verify sync
        verification = await sync.verify_sync()
        
        return {
            "status": "complete",
            "entities_synced": entity_result.get("synced", 0) if isinstance(entity_result, dict) else 0,
            "relations_synced": relation_result.get("synced", 0) if isinstance(relation_result, dict) else 0,
            "verification": verification
        }
        
    finally:
        await sync.close()


def run_entity_extraction() -> Dict[str, Any]:
    """Run entity extraction"""
    _configure_logging()
    return asyncio.run(_run_entity_extraction())


def run_community_detection() -> Dict[str, Any]:
    """Run community detection"""
    _configure_logging()
    return asyncio.run(_run_community_detection())


def run_sync_age() -> Dict[str, Any]:
    """Sync to Apache AGE"""
    _configure_logging()
    return asyncio.run(_run_sync_age())


async def _run_comprehensive_evaluation(output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive evaluation across all components"""
    connection_string = get_config().docintel_dsn
    
    print("🚀 Starting comprehensive evaluation...")
    
    try:
        report = await evaluate_clinical_knowledge_graph(
            connection_string=connection_string,
            output_file=output_file
        )
        
        # Display summary results
        print("\n📊 EVALUATION RESULTS SUMMARY")
        print("=" * 50)
        print(f"✅ Overall Clinical Relevance: {report.overall_clinical_relevance:.3f}")
        print(f"✅ Medical-Graph-RAG Compliance: {report.medical_graph_rag_compliance:.3f}")
        print()
        print(f"👥 Entity Metrics - P:{report.entity_metrics.precision:.3f} R:{report.entity_metrics.recall:.3f} F1:{report.entity_metrics.f1_score:.3f}")
        print(f"🔗 Relation Metrics - P:{report.relation_metrics.precision:.3f} R:{report.relation_metrics.recall:.3f} F1:{report.relation_metrics.f1_score:.3f}")
        print(f"🏘️ Community Metrics - Modularity:{report.community_metrics.modularity:.3f} Communities:{report.community_metrics.num_communities}")
        print(f"🔍 Retrieval Metrics - MRR:{report.retrieval_metrics.mean_reciprocal_rank:.3f} NDCG@5:{report.retrieval_metrics.ndcg_at_k.get(5, 0.0):.3f}")
        print()
        print(f"📈 Dataset: {report.dataset_info['total_entities']} entities, {report.dataset_info['total_relations']} relations, {report.dataset_info['total_communities']} communities")
        
        if output_file:
            print(f"📄 Detailed report saved to: {output_file}")
        
        return {
            "status": "complete",
            "clinical_relevance": report.overall_clinical_relevance,
            "rag_compliance": report.medical_graph_rag_compliance,
            "entity_f1": report.entity_metrics.f1_score,
            "relation_f1": report.relation_metrics.f1_score,
            "communities": report.community_metrics.num_communities,
            "retrieval_mrr": report.retrieval_metrics.mean_reciprocal_rank
        }
        
    except Exception as e:
        print(f"❌ Comprehensive evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "failed", "error": str(e)}


async def _run_entity_evaluation() -> Dict[str, Any]:
    """Run entity extraction evaluation"""
    connection_string = get_config().docintel_dsn
    
    print("👥 Evaluating entity extraction...")
    
    try:
        evaluator = ClinicalEvaluationFramework(connection_string)
        await evaluator.connect()
        
        metrics = await evaluator.evaluate_entity_extraction()
        
        print(f"\n📊 ENTITY EVALUATION RESULTS")
        print("=" * 40)
        print(f"✅ Precision: {metrics.precision:.3f}")
        print(f"✅ Recall: {metrics.recall:.3f}")
        print(f"✅ F1-Score: {metrics.f1_score:.3f}")
        print(f"✅ Clinical Relevance: {metrics.clinical_relevance_score:.3f}")
        print(f"📈 Total Entities: {metrics.total_extracted}")
        print(f"📈 Coverage Rate: {metrics.coverage_rate:.1f} entities/document")
        
        await evaluator.close()
        
        return {
            "status": "complete",
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "clinical_relevance": metrics.clinical_relevance_score,
            "total_entities": metrics.total_extracted
        }
        
    except Exception as e:
        print(f"❌ Entity evaluation failed: {e}")
        return {"status": "failed", "error": str(e)}


async def _run_relation_evaluation() -> Dict[str, Any]:
    """Run relation extraction evaluation"""
    connection_string = get_config().docintel_dsn
    
    print("🔗 Evaluating relation extraction...")
    
    try:
        evaluator = ClinicalEvaluationFramework(connection_string)
        await evaluator.connect()
        
        metrics = await evaluator.evaluate_relation_extraction()
        
        print(f"\n📊 RELATION EVALUATION RESULTS")
        print("=" * 40)
        print(f"✅ Precision: {metrics.precision:.3f}")
        print(f"✅ Recall: {metrics.recall:.3f}")
        print(f"✅ F1-Score: {metrics.f1_score:.3f}")
        print(f"✅ Graph Connectivity: {metrics.graph_connectivity:.3f}")
        print(f"✅ Semantic Coherence: {metrics.semantic_coherence:.3f}")
        print(f"📈 Total Relations: {metrics.total_extracted}")
        
        await evaluator.close()
        
        return {
            "status": "complete",
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1_score": metrics.f1_score,
            "connectivity": metrics.graph_connectivity,
            "coherence": metrics.semantic_coherence,
            "total_relations": metrics.total_extracted
        }
        
    except Exception as e:
        print(f"❌ Relation evaluation failed: {e}")
        return {"status": "failed", "error": str(e)}


async def _run_community_evaluation() -> Dict[str, Any]:
    """Run community detection evaluation"""
    connection_string = get_config().docintel_dsn
    
    print("🏘️ Evaluating community detection...")
    
    try:
        evaluator = ClinicalEvaluationFramework(connection_string)
        await evaluator.connect()
        
        metrics = await evaluator.evaluate_community_detection()
        
        print(f"\n📊 COMMUNITY EVALUATION RESULTS")
        print("=" * 40)
        print(f"✅ Number of Communities: {metrics.num_communities}")
        print(f"✅ Average Community Size: {metrics.average_community_size:.1f}")
        print(f"✅ Modularity: {metrics.modularity:.3f}")
        print(f"✅ Silhouette Score: {metrics.silhouette_score:.3f}")
        print(f"✅ Clinical Clustering Quality: {metrics.clinical_clustering_quality:.3f}")
        print(f"✅ Coverage: {metrics.coverage:.3f}")
        
        await evaluator.close()
        
        return {
            "status": "complete",
            "num_communities": metrics.num_communities,
            "avg_size": metrics.average_community_size,
            "modularity": metrics.modularity,
            "silhouette": metrics.silhouette_score,
            "clinical_quality": metrics.clinical_clustering_quality,
            "coverage": metrics.coverage
        }
        
    except Exception as e:
        print(f"❌ Community evaluation failed: {e}")
        return {"status": "failed", "error": str(e)}


async def _run_retrieval_evaluation() -> Dict[str, Any]:
    """Run U-Retrieval system evaluation"""
    connection_string = get_config().docintel_dsn
    
    print("🔍 Evaluating U-Retrieval system...")
    
    try:
        evaluator = ClinicalEvaluationFramework(connection_string)
        await evaluator.connect()
        
        metrics = await evaluator.evaluate_retrieval_system()
        
        print(f"\n📊 RETRIEVAL EVALUATION RESULTS")
        print("=" * 40)
        print(f"✅ Mean Reciprocal Rank: {metrics.mean_reciprocal_rank:.3f}")
        print(f"✅ NDCG@5: {metrics.ndcg_at_k.get(5, 0.0):.3f}")
        print(f"✅ Precision@5: {metrics.precision_at_k.get(5, 0.0):.3f}")
        print(f"✅ Recall@5: {metrics.recall_at_k.get(5, 0.0):.3f}")
        print(f"✅ Average Query Time: {metrics.average_query_time:.1f}ms")
        print(f"✅ Clinical Relevance Correlation: {metrics.clinical_relevance_correlation:.3f}")
        
        await evaluator.close()
        
        return {
            "status": "complete",
            "mrr": metrics.mean_reciprocal_rank,
            "ndcg_5": metrics.ndcg_at_k.get(5, 0.0),
            "precision_5": metrics.precision_at_k.get(5, 0.0),
            "recall_5": metrics.recall_at_k.get(5, 0.0),
            "avg_query_time": metrics.average_query_time,
            "clinical_correlation": metrics.clinical_relevance_correlation
        }
        
    except Exception as e:
        print(f"❌ Retrieval evaluation failed: {e}")
        return {"status": "failed", "error": str(e)}


def run_comprehensive_evaluation(output_file: Optional[str] = None) -> Dict[str, Any]:
    """Run comprehensive evaluation"""
    _configure_logging()
    return asyncio.run(_run_comprehensive_evaluation(output_file))


def run_entity_evaluation() -> Dict[str, Any]:
    """Run entity evaluation"""
    _configure_logging()
    return asyncio.run(_run_entity_evaluation())


def run_relation_evaluation() -> Dict[str, Any]:
    """Run relation evaluation"""
    _configure_logging()
    return asyncio.run(_run_relation_evaluation())


def run_community_evaluation() -> Dict[str, Any]:
    """Run community evaluation"""
    _configure_logging()
    return asyncio.run(_run_community_evaluation())


def run_retrieval_evaluation() -> Dict[str, Any]:
    """Run retrieval evaluation"""
    _configure_logging()
    return asyncio.run(_run_retrieval_evaluation())


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Knowledge Graph Operations")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Entity extraction subcommand
    extract_parser = subparsers.add_parser("extract", help="Extract entities and relations")
    
    # Community detection subcommand
    community_parser = subparsers.add_parser("communities", help="Detect communities in knowledge graph")
    
    # AGE sync subcommand
    sync_parser = subparsers.add_parser("sync", help="Sync to Apache AGE property graph")
    
    # Full pipeline subcommand
    pipeline_parser = subparsers.add_parser("pipeline", help="Run full knowledge graph pipeline")
    
    # Evaluation subcommands
    eval_parser = subparsers.add_parser("evaluate", help="Run comprehensive evaluation")
    eval_parser.add_argument("--output", "-o", help="Output file for evaluation report (JSON)")
    
    eval_entities_parser = subparsers.add_parser("eval-entities", help="Evaluate entity extraction")
    eval_relations_parser = subparsers.add_parser("eval-relations", help="Evaluate relation extraction")
    eval_communities_parser = subparsers.add_parser("eval-communities", help="Evaluate community detection")
    eval_retrieval_parser = subparsers.add_parser("eval-retrieval", help="Evaluate U-Retrieval system")
    
    args = parser.parse_args(argv)
    
    if not args.command:
        parser.print_help()
        return
    
    logger = logging.getLogger(__name__)
    
    if args.command == "extract":
        logger.info("🧠 Running entity extraction...")
        result = run_entity_extraction()
        logger.info("Entity extraction completed", extra={"result": result})
        
    elif args.command == "communities":
        logger.info("🏘️ Running community detection...")
        result = run_community_detection()
        logger.info("Community detection completed", extra={"result": result})
        
    elif args.command == "sync":
        logger.info("🔄 Syncing to Apache AGE...")
        result = run_sync_age()
        logger.info("AGE sync completed", extra={"result": result})
        
    elif args.command == "pipeline":
        logger.info("🚀 Running full knowledge graph pipeline...")
        
        # Run full pipeline: extract -> sync -> communities
        try:
            # Step 1: Extract entities and relations
            print("Step 1/3: Entity extraction...")
            extract_result = run_entity_extraction()
            print("✅ Entity extraction complete")
            
            # Step 2: Sync to AGE
            print("Step 2/3: Syncing to Apache AGE...")
            sync_result = run_sync_age()
            print("✅ AGE sync complete")
            
            # Step 3: Community detection
            print("Step 3/3: Community detection...")
            community_result = run_community_detection()
            print("✅ Community detection complete")
            
            result = {
                "status": "complete",
                "extract": extract_result,
                "sync": sync_result,
                "communities": community_result
            }
            
            print("\n🎉 Full knowledge graph pipeline completed!")
            extracted_entities = extract_result.get("entities_extracted") or extract_result.get("total_entities") or 0
            extracted_relations = extract_result.get("relations_extracted") or extract_result.get("total_relations") or 0
            processed_chunks = extract_result.get("chunks_processed") or extract_result.get("total_chunks") or 0
            print(f"  Entities extracted: {extracted_entities}")
            print(f"  Relations extracted: {extracted_relations}")
            print(f"  Chunks processed: {processed_chunks}")
            print(f"  Communities: {community_result.get('communities_created', 0)}")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            result = {"status": "failed", "error": str(e)}
        
        logger.info("Knowledge graph pipeline completed", extra={"result": result})
    
    elif args.command == "evaluate":
        logger.info("📊 Running comprehensive evaluation...")
        result = run_comprehensive_evaluation(getattr(args, 'output', None))
        logger.info("Comprehensive evaluation completed", extra={"result": result})
        
    elif args.command == "eval-entities":
        logger.info("👥 Running entity evaluation...")
        result = run_entity_evaluation()
        logger.info("Entity evaluation completed", extra={"result": result})
        
    elif args.command == "eval-relations":
        logger.info("🔗 Running relation evaluation...")
        result = run_relation_evaluation()
        logger.info("Relation evaluation completed", extra={"result": result})
        
    elif args.command == "eval-communities":
        logger.info("🏘️ Running community evaluation...")
        result = run_community_evaluation()
        logger.info("Community evaluation completed", extra={"result": result})
        
    elif args.command == "eval-retrieval":
        logger.info("🔍 Running retrieval evaluation...")
        result = run_retrieval_evaluation()
        logger.info("Retrieval evaluation completed", extra={"result": result})


if __name__ == "__main__":  # pragma: no cover
    main()