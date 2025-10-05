# Architecture Deep Dive: Community Detection Clusterfuck

## THE ACTUAL STATE

### Database Facts
```
docintel.entities:        37,657 rows
docintel.relations:        5,266 rows  
docintel.meta_graphs:        426 rows (document chunks)
ag_catalog.communities:   33,311 rows

AGE clinical_graph:
  - Entity nodes:         37,657
  - RELATES_TO edges:      5,266
```

### What Communities Actually Are

**Communities = Connected Components of Entity Graph**

```sql
-- Community 1461 (largest, 28 entities):
nodes: ["entity-uuid-1", "entity-uuid-2", ... "entity-uuid-28"]  -- 28 entity UUIDs
edges: [[uuid1, uuid2], ...]  -- 36 relation edges between entities
chunk_ids: same as nodes (entities)  -- DUPLICATE DATA
occurrence: 1.0 (highest)
```

**Result of clustering:** 31,955 communities with only 1 entity (isolated nodes)

### The Data Flow

```
1. DATA EXTRACTION (scripts/build_knowledge_graph.py)
   ├─> Extracts entities from documents
   ├─> Creates meta_graphs (one per document chunk)
   └─> Stores in:
       ├─> docintel.entities (37,657 rows, each has meta_graph_id FK)
       └─> docintel.relations (5,266 rows, each has subject/object entity_id FKs)

2. AGE GRAPH SYNC (scripts/sync_relations_to_age.py)
   ├─> Creates AGE vertices: one per entity (37,657 Entity nodes)
   ├─> Creates AGE edges: one per relation (5,266 RELATES_TO edges)
   └─> Purpose: Enable Cypher graph traversal queries
   
3. COMMUNITY DETECTION (src/docintel/knowledge_graph/community_detection.py)
   ├─> build_networkx_graph():
   │   ├─> Queries docintel.entities → 37,657 NetworkX nodes
   │   ├─> Queries docintel.relations → 5,266 NetworkX edges
   │   └─> Returns: Graph(nodes=37657, edges=5266)
   │
   ├─> leiden_clustering() or fallback to connected_components():
   │   ├─> Input: Sparse entity graph (5266 edges / 37657 nodes = 0.014% density)
   │   ├─> Algorithm: Connected components (graspologic not working)
   │   └─> Output: 33,311 components (most entities isolated)
   │
   └─> store_community_data():
       ├─> Inserts into ag_catalog.communities (33,311 rows)
       │   └─> Each community.nodes = list of entity UUIDs
       └─> Updates docintel.entities.cluster_data (37,657 rows)
           └─> cluster_data = [{"level": 0, "cluster": 12345}]

4. U-RETRIEVAL (src/docintel/knowledge_graph/u_retrieval.py)
   ├─> _find_relevant_communities():
   │   ├─> Queries ag_catalog.communities (LIMIT 100 for performance)
   │   └─> For each community:
   │       └─> _calculate_community_entity_relevance(community.nodes)
   │
   └─> _calculate_community_entity_relevance(entity_ids):
       ├─> entity_ids = ["uuid1", "uuid2", ...] from community.nodes
       ├─> Query: Get meta_graph_ids for these entity UUIDs
       │   └─> SELECT DISTINCT meta_graph_id FROM docintel.entities WHERE entity_id IN (...)
       ├─> Query: Get all entities from those meta_graphs
       │   └─> SELECT entity_text FROM docintel.entities WHERE meta_graph_id IN (...)
       └─> Match entity_text against query terms → relevance score
```

## THE FUNDAMENTAL PROBLEM

### Graph Density is Too Sparse

```
Density = edges / (nodes × (nodes-1) / 2)
        = 5,266 / (37,657 × 37,656 / 2)
        = 5,266 / 708,946,796
        = 0.0000074 (0.00074%)
```

**Connected components algorithm finds disconnected islands:**
- Graph: 37,657 nodes with only 5,266 edges
- Result: 31,955 isolated single-entity "communities"
- Largest component: only 28 entities

**Why so sparse?**
- Relations extracted from clinical text are conservative
- Most entities don't have explicit relations in text
- Example: "niraparib" might appear 50 times but not be explicitly related to other entities

### What Communities SHOULD Be

**Medical-Graph-RAG Design Intent (from paper):**

Communities should cluster **document chunks** (meta_graphs) not individual entities.

```
Nodes = meta_graphs (426 chunks)
Edges = shared entities between chunks

Community 1: "PARP Inhibitor Trials"
├─> meta_graph 316 (NCT03840967, chunk 1): 93 entities
├─> meta_graph 317 (NCT03840967, chunk 2): 64 entities  
├─> meta_graph 318 (NCT03840967, chunk 3): 118 entities
├─> ... 15 more related chunks
└─> Connected by: shared entities (niraparib, ovarian cancer, PARP, etc.)

Community 2: "Safety Monitoring Protocols"
├─> meta_graph 42 (NCT01234567, safety section): 78 entities
├─> meta_graph 189 (NCT09876543, AE monitoring): 65 entities
└─> ... 20 more chunks
```

**Result:** ~10-20 meaningful communities, each containing 10-50 related document chunks

### What Communities ACTUALLY Are

```
Community 1461: 28 entities from 1 meta_graph
├─> entity "163" (measurement)
├─> entity "niraparib" (person)  
├─> entity "adverse event" (clinical_event)
└─> ... 25 more entities
Connected by: 36 relation edges within the same document chunk

Community 12345: 1 entity (isolated)
└─> entity "headache" (symptom)
    No relations to any other entities
```

**Result:** 33,311 garbage communities, mostly size=1, useless for retrieval

## THE ROOT CAUSE

### Wrong Abstraction Level

**Current (BROKEN):**
```python
# community_detection.py line 78-152
async def build_networkx_graph(self):
    # Query entities as nodes
    entities = SELECT * FROM docintel.entities  # 37,657 rows
    relations = SELECT * FROM docintel.relations  # 5,266 rows
    
    G = nx.Graph()
    for entity in entities:
        G.add_node(entity.entity_id, text=entity.entity_text, ...)  # 37K nodes
    
    for relation in relations:
        G.add_edge(relation.subject_entity_id, relation.object_entity_id, ...)  # 5K edges
    
    return G  # Sparse entity graph
```

**Should be (FIXED):**
```python
async def build_networkx_graph(self):
    # Query meta_graphs as nodes
    meta_graphs = SELECT mg.*, array_agg(e.entity_id) as entity_ids
                  FROM docintel.meta_graphs mg
                  JOIN docintel.entities e ON e.meta_graph_id = mg.meta_graph_id
                  GROUP BY mg.meta_graph_id  # 426 rows
    
    G = nx.Graph()
    for mg in meta_graphs:
        G.add_node(mg.meta_graph_id, nct_id=mg.nct_id, entity_count=len(mg.entity_ids))  # 426 nodes
    
    # Create edges between meta_graphs that share entities
    for mg1, mg2 in combinations(meta_graphs, 2):
        shared_entities = set(mg1.entity_ids) & set(mg2.entity_ids)
        
        # Connect if share ≥3 entities OR same NCT study
        if len(shared_entities) >= 3 or mg1.nct_id == mg2.nct_id:
            G.add_edge(mg1.meta_graph_id, mg2.meta_graph_id, 
                      weight=len(shared_entities))  # ~2K-5K edges
    
    return G  # Dense meta_graph graph
```

### Why Meta_Graph Clustering Works

```
426 meta_graphs × shared entities = dense connections

Example:
- meta_graph 316: entities [niraparib, ovarian cancer, PARP, dose, ...]
- meta_graph 317: entities [niraparib, adverse event, monitoring, dose, ...]
- Shared: {niraparib, dose} (2 entities) → no edge (threshold=3)

- meta_graph 316: entities [niraparib, ovarian cancer, PARP, dose, efficacy, ...]
- meta_graph 318: entities [niraparib, PARP, inhibitor, dose, efficacy, ...]  
- Shared: {niraparib, PARP, dose, efficacy} (4 entities) → CREATE EDGE

Result: Dense graph where chunks from same study or similar topics are connected
Louvain clustering: ~10-20 communities of related chunks
```

## THE FIX

### Option A: Keep AGE as Entity Graph, Fix Community Detection Only

**Pros:**
- Smaller change
- AGE graph stays useful for entity-level queries
- Community detection separate concern

**Cons:**
- Two different graph abstractions (confusing)
- AGE graph not used by retrieval anyway

**Changes:**
1. Rewrite `build_networkx_graph()` to create meta_graph nodes
2. Keep AGE sync as-is (entity nodes)
3. Communities now cluster meta_graphs
4. U-Retrieval simplified (no UUID→meta_graph mapping needed)

### Option B: Rebuild Everything as Meta_Graph-Based

**Pros:**
- Consistent abstraction throughout
- AGE graph becomes useful for chunk-level queries
- Cleaner architecture

**Cons:**
- Bigger change
- Lose entity-level graph traversal in AGE
- Need to resync AGE graph

**Changes:**
1. Rewrite `sync_relations_to_age.py` to create meta_graph nodes
2. Rewrite `build_networkx_graph()` to match
3. Communities cluster meta_graphs
4. U-Retrieval works directly with meta_graph IDs

## CURRENT RETRIEVAL FLOW (Inefficient)

```
Query: "What is niraparib?"

1. Find relevant communities (100 of 33,311)
   └─> Loop over 100 communities:
       ├─> community.nodes = [entity-uuid-1, entity-uuid-2, ...]
       ├─> Query: Get meta_graph_ids for these entity UUIDs (DB query #1)
       ├─> Query: Get entities from those meta_graphs (DB query #2)
       └─> Match entities against "niraparib" → relevance score
   
2. Collect entities from relevant communities
   └─> Found 50 entities matching "niraparib"

3. Group by meta_graph_id
   └─> meta_graph 316: 23 entities
   └─> meta_graph 317: 15 entities
   └─> meta_graph 318: 12 entities

4. Retrieve chunks
   └─> Query embeddings table for these meta_graphs
   └─> Return 3 chunks

Total: 200+ database queries (100 communities × 2 queries each)
Time: 4897ms (5 seconds)
```

## PROPOSED RETRIEVAL FLOW (Efficient)

```
Query: "What is niraparib?"

1. Find relevant communities (10 of 20)
   └─> community.nodes = [meta-graph-id-1, meta-graph-id-2, ...]  (already meta_graphs!)
   └─> Load entities from these meta_graphs (1 query)
   └─> Match against "niraparib" → relevance scores

2. Get top meta_graphs
   └─> meta_graph 316: relevance 0.9
   └─> meta_graph 317: relevance 0.7
   └─> meta_graph 318: relevance 0.6

3. Retrieve chunks
   └─> Query embeddings table for these meta_graphs
   └─> Return 3 chunks

Total: 3-5 database queries
Time: <1 second
```

## RECOMMENDATION

**Go with Option A (community detection fix only)** because:

1. **Smaller blast radius:** Only touch community_detection.py (~50 lines changed)
2. **Query already works:** Just need to make it faster
3. **AGE graph unused:** Retrieval doesn't query AGE anyway, so keeping entity nodes is fine
4. **Reversible:** Can always do Option B later if needed

**Implementation:**
1. Rewrite `build_networkx_graph()` to create meta_graph nodes + entity-sharing edges
2. Use Louvain clustering (connected components won't work on 426 nodes)
3. Store meta_graph IDs in communities.nodes (not entity UUIDs)
4. Simplify U-Retrieval (no UUID→meta_graph mapping)
5. Result: ~10-20 communities, <1s queries

**Expected outcome:**
- 33,311 communities → ~15 communities
- 5 second queries → <1 second queries
- Communities actually meaningful (semantically coherent chunks)
