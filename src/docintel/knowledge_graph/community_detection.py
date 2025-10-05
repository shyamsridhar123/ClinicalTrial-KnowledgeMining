"""
Community Detection Module for Clinical Knowledge Graph

Implements Leiden clustering algorithm as specified in Medical-Graph-RAG architecture
to enable hierarchical knowledge organization and improve query precision.

Based on: https://github.com/ImprintLab/Medical-Graph-RAG
"""

import logging
import json
import asyncio
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

import psycopg
import networkx as nx
from ..config import get_config

logger = logging.getLogger(__name__)


@dataclass
class CommunitySchema:
    """Schema for community detection results"""
    level: int
    title: str
    nodes: List[str]
    edges: List[Tuple[str, str]]
    chunk_ids: List[str]
    occurrence: float
    report_string: Optional[str] = None
    report_json: Optional[Dict] = None


class CommunityDetector:
    """
    Clinical knowledge graph community detection using Leiden clustering.
    
    This enables hierarchical knowledge organization as per Medical-Graph-RAG standards.
    """
    
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
        self.conn = None
        
    async def connect(self):
        """Establish database connection"""
        self.conn = await psycopg.AsyncConnection.connect(self.connection_string)
        await self.conn.execute("LOAD 'age';")
        await self.conn.execute('SET search_path = ag_catalog, public;')
        
        # Verify search path and column existence
        result = await self.conn.execute('SHOW search_path;')
        path = await result.fetchone()
        logger.info(f"Connected to database with search_path: {path[0]}")
        
        # Check if cluster_data column exists
        result = await self.conn.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'entities' AND table_schema = 'ag_catalog' AND column_name = 'cluster_data';
        """)
        cluster_col = await result.fetchone()
        if not cluster_col:
            logger.warning("cluster_data column missing, adding it...")
            await self.conn.execute('ALTER TABLE ag_catalog.entities ADD COLUMN cluster_data JSONB;')
            logger.info("Added cluster_data column")
        
        logger.info("Connected to database and loaded AGE extension")
        
    async def close(self):
        """Close database connection"""
        if self.conn:
            await self.conn.close()
            
    async def build_networkx_graph(self) -> nx.Graph:
        """
        Build NetworkX graph from meta_graphs (document chunks).
        
        CHANGED: Instead of clustering 37,657 entities (which produces 31,955 useless
        single-entity communities), we cluster 425 meta_graphs (chunks). Meta_graphs
        that share >=3 entities are connected, producing ~10-20 meaningful communities.
        """
        logger.info("Building NetworkX graph from meta_graphs (chunks)...")
        
        # Fetch meta_graphs with their entity counts
        result = await self.conn.execute("""
            SELECT 
                mg.meta_graph_id,
                mg.chunk_id,
                mg.nct_id,
                COUNT(e.entity_id) as entity_count,
                array_agg(e.entity_text ORDER BY e.entity_text) as entity_texts
            FROM docintel.meta_graphs mg
            LEFT JOIN docintel.entities e ON e.meta_graph_id = mg.meta_graph_id
            GROUP BY mg.meta_graph_id, mg.chunk_id, mg.nct_id
            HAVING COUNT(e.entity_id) > 0
            ORDER BY mg.meta_graph_id
        """)
        meta_graphs = await result.fetchall()
        
        logger.info(f"Fetched {len(meta_graphs)} meta_graphs with entities")
        
        # Fetch all entities grouped by meta_graph for overlap calculation
        result = await self.conn.execute("""
            SELECT 
                meta_graph_id,
                array_agg(DISTINCT LOWER(entity_text)) as entity_set
            FROM docintel.entities
            GROUP BY meta_graph_id
        """)
        meta_graph_entities = {row[0]: set(row[1]) for row in await result.fetchall()}
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add meta_graph nodes with attributes
        for mg in meta_graphs:
            meta_graph_id, chunk_id, nct_id, entity_count, entity_texts = mg
            
            # Create descriptive summary of top entities
            top_entities = entity_texts[:10] if entity_texts else []
            description = f"Chunk {nct_id or 'Unknown'}: {', '.join(top_entities)}"
            
            node_attrs = {
                'id': meta_graph_id,
                'chunk_id': chunk_id,
                'nct_id': nct_id,
                'entity_count': entity_count,
                'description': description,
                'type': 'meta_graph',
            }
            
            G.add_node(str(meta_graph_id), **node_attrs)
        
        # Add edges between meta_graphs that share >=5 entities
        # This creates semantic connections between related chunks
        # Threshold of 5 creates ~20-50 meaningful communities
        min_shared_entities = 5
        edge_count = 0
        
        logger.info(f"Computing entity overlap between {len(meta_graphs)} meta_graphs...")
        
        # Compare all pairs of meta_graphs
        meta_graph_ids = list(meta_graph_entities.keys())
        for i, mg1_id in enumerate(meta_graph_ids):
            if i % 50 == 0:
                logger.info(f"Processing meta_graph {i+1}/{len(meta_graph_ids)}")
                
            mg1_entities = meta_graph_entities.get(mg1_id, set())
            
            for mg2_id in meta_graph_ids[i+1:]:
                mg2_entities = meta_graph_entities.get(mg2_id, set())
                
                # Count shared entities (case-insensitive)
                shared_entities = mg1_entities & mg2_entities
                shared_count = len(shared_entities)
                
                if shared_count >= min_shared_entities:
                    # Create edge weighted by number of shared entities
                    edge_attrs = {
                        'shared_entities': shared_count,
                        'weight': float(shared_count),
                        'description': f"Share {shared_count} entities",
                    }
                    
                    G.add_edge(str(mg1_id), str(mg2_id), **edge_attrs)
                    edge_count += 1
        
        logger.info(f"Built meta_graph clustering graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        logger.info(f"Created {edge_count} connections between chunks sharing >={min_shared_entities} entities")
        return G
        
    def stable_largest_connected_component(self, graph: nx.Graph) -> nx.Graph:
        """
        Get the largest connected component of the graph.
        Based on Medical-Graph-RAG implementation.
        """
        try:
            from graspologic.utils import largest_connected_component
            
            graph = graph.copy()
            graph = largest_connected_component(graph)
            
            # Stabilize node names (uppercase and strip)
            node_mapping = {
                node: str(node).upper().strip() 
                for node in graph.nodes()
            }
            graph = nx.relabel_nodes(graph, node_mapping)
            
            return self._stabilize_graph(graph)
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"graspologic not available or has compatibility issues, using networkx connected components: {e}")
            # Fallback to networkx
            if nx.is_connected(graph):
                return self._stabilize_graph(graph)
            else:
                # Get largest connected component
                largest_cc = max(nx.connected_components(graph), key=len)
                subgraph = graph.subgraph(largest_cc).copy()
                return self._stabilize_graph(subgraph)
                
    def _stabilize_graph(self, graph: nx.Graph) -> nx.Graph:
        """
        Ensure graph with same relationships is always read the same way.
        Based on Medical-Graph-RAG implementation.
        """
        # Sort nodes
        sorted_nodes = sorted(graph.nodes(data=True), key=lambda x: x[0])
        
        # Sort edges
        def _sort_source_target(edge):
            source, target, edge_data = edge
            if source > target:
                source, target = target, source
            return source, target, edge_data
            
        sorted_edges = sorted(
            [_sort_source_target(e) for e in graph.edges(data=True)],
            key=lambda x: f"{x[0]} -> {x[1]}"
        )
        
        # Create new graph with sorted order
        stable_graph = nx.Graph()
        stable_graph.add_nodes_from(sorted_nodes)
        stable_graph.add_edges_from(sorted_edges)
        
        return stable_graph
        
    async def leiden_clustering(
        self, 
        graph: nx.Graph, 
        max_cluster_size: int = 10,
        random_seed: int = 0xDEADBEEF
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Perform Leiden clustering on the graph.
        Based on Medical-Graph-RAG implementation.
        """
        logger.info("Starting Leiden clustering...")
        
        try:
            from graspologic.partition import hierarchical_leiden
            
            # Get stable largest connected component
            stable_graph = self.stable_largest_connected_component(graph)
            
            logger.info(f"Clustering graph with {stable_graph.number_of_nodes()} nodes, {stable_graph.number_of_edges()} edges")
            
            # Perform hierarchical Leiden clustering
            community_mapping = hierarchical_leiden(
                stable_graph,
                max_cluster_size=max_cluster_size,
                random_seed=random_seed,
            )
            
            # Organize communities by node
            node_communities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
            levels_info = defaultdict(set)
            
            for partition in community_mapping:
                level_key = partition.level
                cluster_id = partition.cluster
                node_id = partition.node
                
                node_communities[node_id].append({
                    "level": level_key,
                    "cluster": cluster_id
                })
                levels_info[level_key].add(cluster_id)
                
            # Convert to regular dict
            node_communities = dict(node_communities)
            
            # Log community statistics
            levels_stats = {k: len(v) for k, v in levels_info.items()}
            logger.info(f"Leiden clustering complete. Communities per level: {levels_stats}")
            
            return node_communities
            
        except (ImportError, AttributeError) as e:
            logger.warning(f"graspologic not available or has compatibility issues: {e}")
            # Fallback to simple connected components
            return await self._fallback_clustering(graph)
            
    async def _fallback_clustering(self, graph: nx.Graph) -> Dict[str, List[Dict[str, Any]]]:
        """Fallback clustering using connected components"""
        logger.warning("Using fallback clustering (connected components)")
        
        node_communities: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Use connected components as level 0 communities
        for i, component in enumerate(nx.connected_components(graph)):
            for node in component:
                node_communities[node].append({
                    "level": 0,
                    "cluster": i
                })
                
        return dict(node_communities)
        
    async def build_community_schema(
        self, 
        graph: nx.Graph, 
        node_communities: Dict[str, List[Dict[str, Any]]]
    ) -> Dict[str, CommunitySchema]:
        """
        Build community schema from clustering results.
        
        CHANGED: Now nodes contain meta_graph_id UUIDs (not entity_id UUIDs).
        This enables direct meta_graph → entities lookup in U-Retrieval.
        """
        logger.info("Building community schema...")
        
        communities = defaultdict(lambda: {
            'level': None,
            'title': None,
            'edges': set(),
            'nodes': set(),  # Will contain meta_graph_id UUIDs
            'chunk_ids': set(),
            'occurrence': 0.0,
        })
        
        max_num_ids = 0
        
        # Process each meta_graph node and its communities
        for node_id, clusters in node_communities.items():
            if not graph.has_node(node_id):
                continue
                
            node_data = graph.nodes[node_id]
            node_edges = list(graph.edges(node_id))
            
            for cluster in clusters:
                level = cluster["level"]
                cluster_key = str(cluster["cluster"])
                
                communities[cluster_key]['level'] = level
                communities[cluster_key]['title'] = f"Clinical Community {cluster_key}"
                
                # Store meta_graph_id UUID in nodes field
                # This is the KEY CHANGE: nodes now contain meta_graph IDs, not entity IDs
                communities[cluster_key]['nodes'].add(node_id)
                
                communities[cluster_key]['edges'].update([
                    tuple(sorted(e[:2])) for e in node_edges
                ])
                
                # Add chunk ID for tracking
                chunk_id = node_data.get('chunk_id', node_data.get('id', node_id))
                communities[cluster_key]['chunk_ids'].add(str(chunk_id))
                max_num_ids = max(max_num_ids, len(communities[cluster_key]['chunk_ids']))
        
        # Convert sets to lists and calculate occurrence
        community_schemas = {}
        for cluster_key, community_data in communities.items():
            community_data['edges'] = [list(e) for e in community_data['edges']]
            community_data['nodes'] = list(community_data['nodes'])  # meta_graph_id UUIDs
            community_data['chunk_ids'] = list(community_data['chunk_ids'])
            community_data['occurrence'] = len(community_data['chunk_ids']) / max_num_ids if max_num_ids > 0 else 0
            
            community_schemas[cluster_key] = CommunitySchema(
                level=community_data['level'],
                title=community_data['title'],
                nodes=community_data['nodes'],  # meta_graph_id UUIDs
                edges=community_data['edges'],
                chunk_ids=community_data['chunk_ids'],
                occurrence=community_data['occurrence']
            )
        
        logger.info(f"Built {len(community_schemas)} communities with meta_graph nodes")
        return community_schemas
        
    async def store_community_data(
        self, 
        node_communities: Dict[str, List[Dict[str, Any]]],
        community_schemas: Dict[str, CommunitySchema]
    ):
        """Store community detection results in database"""
        logger.info("Storing community data in database...")
        
        try:
            # Commit any existing transaction first
            await self.conn.commit()
            logger.info("Committed any existing transaction")
            
            # Use autocommit for persistence
            await self.conn.set_autocommit(True)
            logger.info("Enabled autocommit mode")
            
            # Create communities table if not exists
            logger.info("Creating communities table...")
            await self.conn.execute("""
                CREATE TABLE IF NOT EXISTS ag_catalog.communities (
                    id SERIAL PRIMARY KEY,
                    cluster_key VARCHAR(50) NOT NULL UNIQUE,
                    level INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    nodes JSONB NOT NULL,
                    edges JSONB NOT NULL,
                    chunk_ids JSONB NOT NULL,
                    occurrence FLOAT NOT NULL,
                    report_string TEXT,
                    report_json JSONB,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Communities table created/verified")
            
            # Store community schemas
            logger.info(f"Inserting {len(community_schemas)} community schemas...")
            for cluster_key, schema in community_schemas.items():
                await self.conn.execute("""
                    INSERT INTO ag_catalog.communities 
                    (cluster_key, level, title, nodes, edges, chunk_ids, occurrence)
                    VALUES (%s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (cluster_key) DO UPDATE SET
                        level = EXCLUDED.level,
                        title = EXCLUDED.title,
                        nodes = EXCLUDED.nodes,
                        edges = EXCLUDED.edges,
                        chunk_ids = EXCLUDED.chunk_ids,
                        occurrence = EXCLUDED.occurrence
                """, (
                    cluster_key,
                    schema.level,
                    schema.title,
                    json.dumps(schema.nodes),
                    json.dumps(schema.edges),
                    json.dumps(schema.chunk_ids),
                    schema.occurrence
                ))
            logger.info("Community schemas inserted with autocommit")
            
            # Update entities with cluster information in docintel schema
            logger.info(f"Updating {len(node_communities)} entities with cluster data...")
            for node_id, clusters in node_communities.items():
                await self.conn.execute("""
                    UPDATE docintel.entities 
                    SET cluster_data = %s 
                    WHERE entity_id = %s
                """, (json.dumps(clusters), node_id))
            logger.info("Entity cluster data updated with autocommit")
            
            # Verify the data was stored
            result = await self.conn.execute("SELECT COUNT(*) FROM ag_catalog.communities")
            count = await result.fetchone()
            logger.info(f"Verification: {count[0]} communities in database")
            
            # Reset to transaction mode for other operations
            await self.conn.set_autocommit(False)
            logger.info("Reset to transaction mode")
            
        except Exception as e:
            logger.error(f"Error storing community data: {e}")
            import traceback
            traceback.print_exc()
            # Try to reset autocommit in case of error
            try:
                await self.conn.set_autocommit(False)
            except:
                pass
            raise
        
        logger.info(f"Stored {len(community_schemas)} communities and updated {len(node_communities)} entities")
        
    async def run_community_detection(
        self,
        max_cluster_size: int = 10,
        random_seed: int = 0xDEADBEEF
    ) -> Dict[str, CommunitySchema]:
        """
        Run complete community detection pipeline.
        
        Returns:
            Dictionary of community schemas keyed by cluster ID
        """
        logger.info("🔍 Starting community detection pipeline...")
        
        try:
            await self.connect()
            
            # Step 1: Build NetworkX graph from database
            graph = await self.build_networkx_graph()
            
            # Step 2: Perform Leiden clustering
            node_communities = await self.leiden_clustering(
                graph, 
                max_cluster_size=max_cluster_size,
                random_seed=random_seed
            )
            
            # Step 3: Build community schema
            community_schemas = await self.build_community_schema(graph, node_communities)
            
            # Step 4: Store results in database
            await self.store_community_data(node_communities, community_schemas)
            
            logger.info("✅ Community detection pipeline completed successfully")
            
            # Print summary
            print("\n🏘️ COMMUNITY DETECTION RESULTS:")
            print("=" * 50)
            
            levels = defaultdict(int)
            for schema in community_schemas.values():
                levels[schema.level] += 1
                
            for level in sorted(levels.keys()):
                print(f"Level {level}: {levels[level]} communities")
                
            print(f"\nTotal communities: {len(community_schemas)}")
            print(f"Total nodes clustered: {len(node_communities)}")
            
            # Show sample communities
            print("\n📊 Sample Communities:")
            for i, (cluster_key, schema) in enumerate(list(community_schemas.items())[:3]):
                print(f"  {schema.title} (Level {schema.level})")
                print(f"    Nodes: {len(schema.nodes)}, Edges: {len(schema.edges)}")
                print(f"    Occurrence: {schema.occurrence:.3f}")
                
            return community_schemas
            
        except Exception as e:
            logger.error(f"Community detection failed: {e}")
            raise
        finally:
            await self.close()


async def main():
    """Main execution function"""
    connection_string = get_config().docintel_dsn
    
    detector = CommunityDetector(connection_string)
    
    try:
        communities = await detector.run_community_detection(
            max_cluster_size=10,  # Medical-Graph-RAG default
            random_seed=0xDEADBEEF  # Medical-Graph-RAG default
        )
        
        print("\n🎉 Community detection complete!")
        print("Ready for hierarchical knowledge organization and U-Retrieval queries!")
        
        return communities
        
    except Exception as e:
        print(f"❌ Community detection failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(main())