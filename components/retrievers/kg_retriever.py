import networkx as nx
from typing import List, Dict, Any, Tuple
from core.interfaces import BaseRetriever, RetrievalResult, QueryContext, Document, RetrievalType
from core.logging import RAGLogger

class KnowledgeGraphRetriever(BaseRetriever):
    """Knowledge graph-based retrieval using NetworkX."""
    
    def __init__(self, kg_data: List[Tuple[str, str, str]]):  # (subject, predicate, object) triples
        self.logger = RAGLogger("kg_retriever")
        self.graph = nx.DiGraph()
        
        with self.logger.log_operation("kg_building", num_triples=len(kg_data)):
            # Build the knowledge graph
            for subject, predicate, obj in kg_data:
                self.graph.add_edge(subject, obj, relation=predicate)
    
    async def retrieve(self, query_context: QueryContext, top_k: int) -> List[RetrievalResult]:
        """Retrieve relevant knowledge graph paths."""
        with self.logger.log_operation("kg_retrieval", entities=query_context.entities, top_k=top_k):
            results = []
            
            # Find relevant subgraphs for each entity
            for entity in query_context.entities:
                entity_results = await self._find_entity_subgraph(entity, query_context, top_k)
                results.extend(entity_results)
            
            # Sort by relevance and return top-k
            results.sort(key=lambda x: x.score, reverse=True)
            return results[:top_k]
    
    async def _find_entity_subgraph(self, entity: str, query_context: QueryContext, max_results: int) -> List[RetrievalResult]:
        """Find relevant subgraph for an entity."""
        results = []
        
        # Find nodes that match the entity (fuzzy matching)
        matching_nodes = [node for node in self.graph.nodes() if entity.lower() in node.lower()]
        
        for node in matching_nodes[:max_results]:
            # Get neighbors and their relationships
            neighbors = list(self.graph.neighbors(node))
            predecessors = list(self.graph.predecessors(node))
            
            # Build context from relationships
            context_parts = []
            
            # Outgoing relationships
            for neighbor in neighbors:
                relation = self.graph[node][neighbor]['relation']
                context_parts.append(f"{node} {relation} {neighbor}")
            
            # Incoming relationships
            for pred in predecessors:
                relation = self.graph[pred][node]['relation']
                context_parts.append(f"{pred} {relation} {node}")
            
            if context_parts:
                content = ". ".join(context_parts)
                
                # Score based on query relevance
                score = self._calculate_kg_relevance(content, query_context)
                
                doc = Document(
                    id=f"kg_{node}",
                    content=content,
                    metadata={"entity": node, "type": "knowledge_graph"},
                    source="knowledge_graph"
                )
                
                results.append(RetrievalResult(
                    document=doc,
                    score=score,
                    retrieval_type=RetrievalType.KNOWLEDGE_GRAPH,
                    explanation=f"Knowledge graph path for entity: {entity}"
                ))
        
        return results
    
    def _calculate_kg_relevance(self, content: str, query_context: QueryContext) -> float:
        """Calculate relevance score for KG content."""
        score = 0.0
        content_lower = content.lower()
        query_lower = query_context.transformed_query.lower()
        
        # Entity match bonus
        for entity in query_context.entities:
            if entity.lower() in content_lower:
                score += 0.3
        
        # Query term overlap
        query_terms = set(query_lower.split())
        content_terms = set(content_lower.split())
        overlap = len(query_terms.intersection(content_terms))
        score += overlap * 0.1
        
        # Intent-based scoring
        if query_context.intent == "definition" and any(rel in content_lower for rel in ["is", "type", "category"]):
            score += 0.2
        elif query_context.intent == "location_lookup" and any(rel in content_lower for rel in ["located", "in", "at"]):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0