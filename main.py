import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime 
from inngest.experimental import ai
from data_loader import load_and_create_nodes_from_pdf, embed_model
from vector_db import QDrantStorage
from llama_index.core.schema import Node, TextNode  # Added TextNode import
from llama_index.core.schema import NodeRelationship
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGNodesAndSrc 

load_dotenv()

inngest_client = inngest.Inngest(
        app_id='rag_prod_app',
        logger=logging.getLogger("uvicorn"),
        is_production=False,
        serializer= inngest.PydanticSerializer()
        )

@inngest_client.create_function(
        fn_id="RAG: Inngest PDF",
        trigger=inngest.TriggerEvent(event="rag/inngest_pdf")
        )
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGNodesAndSrc:
        pdf_path = ctx.event.data['pdf_path']
        source_id = ctx.event.data.get('source_id', pdf_path)
        
        nodes = load_and_create_nodes_from_pdf(pdf_path)
        
        node_dicts = [node.to_dict() for node in nodes]
        
        return RAGNodesAndSrc(nodes=node_dicts, source_id=source_id)

    def _upsert(nodes_and_src : RAGNodesAndSrc) -> RAGUpsertResult:
        node_dicts = nodes_and_src.nodes
        source_id = nodes_and_src.source_id

        nodes = [TextNode.from_dict(n) for n in node_dicts]

        vecs = embed_model(nodes)
        ids = [node.id_ for node in nodes]
        payloads = []
        for node in nodes:
            parent_relation = node.relationships.get(NodeRelationship.PARENT)
            
            parent_id = parent_relation.node_id if parent_relation else None
            
            payloads.append({
                "source": source_id,
                "text": node.text,
                "parent_id": parent_id
            })

        QDrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested= len(nodes))

    nodes_and_src = await ctx.step.run('load-and-create-nodes', lambda: _load(ctx), output_type=RAGNodesAndSrc)
    ingested = await ctx.step.run('embed-nodes-and-upsert', lambda: _upsert(nodes_and_src), output_type= RAGUpsertResult)

    return ingested.model_dump()

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf])