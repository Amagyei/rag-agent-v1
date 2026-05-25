from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from typing import Any
from data_loader import EMBED_DIM


class QDrantStorage:
    def __init__(self, url="http://localhost:6333", collection_name="docs", dim=EMBED_DIM):
        self.client = QdrantClient(url=url, timeout=30)
        self.collection = collection_name
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                    )


    def upsert(self, ids, vectors, payloads):
        points = [PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i]) for i in range(len(ids))]
        self.client.upsert(self.collection, points=points)
        

    def search(self, query_vector: list[float], top_k: int = 5) -> dict[str, Any]:
        """
        Executes a hierarchical vector search. Queries precise child nodes 
        and extracts/returns their parent node texts for enriched LLM context.
        
        Parameters:
        -----------
        query_vector : list[float]
            The dense vector embedding of the user's search query.
        top_k : int
            The number of matching child documents to retrieve initially.
            
        Returns:
        --------
        dict[str, list]
            A dictionary containing structural context strings and source track IDs.
            Example: {"contexts": ["...", "..."], "sources": ["doc_123"]}
        """
        # Step 1: Perform similarity search to locate nearby child chunks
        child_results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            with_payload=True,
            limit=top_k
        ).points
        
        contexts = []
        sources = set()
        parent_ids_to_fetch = []
        
        # Step 2: Loop through hits to isolate source metadata and parent connections
        for hit in child_results:
            payload = hit.payload or {}
            
            # Track the original file identifier
            source = payload.get("source")
            if source:
                sources.add(source)
                
            # Queue parent IDs for batch fetching
            parent_id = payload.get("parent_id")
            if parent_id:
                parent_ids_to_fetch.append(parent_id)
        
        # Step 3: Batch retrieve parent texts using primary keys (Deduplicated)
        parent_text_map = {}
        unique_parent_ids = list(set(parent_ids_to_fetch))
        
        if unique_parent_ids:
            # client.retrieve uses fast KV lookup on Point IDs
            parent_records = self.client.retrieve(
                collection_name=self.collection,
                ids=unique_parent_ids,
                with_payload=True
            )
            # Map the point ID to its full text payload
            for record in parent_records:
                p_payload = record.payload or {}
                if "text" in p_payload:
                    parent_text_map[str(record.id)] = p_payload["text"]

        # Step 4: Construct the final enriched context array
        for hit in child_results:
            payload = hit.payload or {}
            parent_id = payload.get("parent_id")
            
            # If parent exists in our map, use parent text; otherwise fallback to child text
            if parent_id and str(parent_id) in parent_text_map:
                contexts.append(parent_text_map[str(parent_id)])
            else:
                child_text = payload.get("text", "")
                if child_text:
                    contexts.append(child_text)

        return {
            "contexts": contexts, 
            "sources": list(sources)
        }

