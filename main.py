import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from dotenv import load_dotenv
import uuid
import os
import datetime 
from inngest.experimental import ai
from data_loader import embed_nodes, embed_query, load_and_create_nodes_from_pdf, embed_model
from vector_db import QDrantStorage
from llama_index.core.schema import Node, TextNode  # Added TextNode import
from llama_index.core.schema import NodeRelationship
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGNodesAndSrc 

load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL")
OLLAMA_LLM_MODEL = os.getenv("OLLAMA_LLM_MODEL")
OLLAMA_AUTH_KEY = os.getenv("OLLAMA_AUTH_KEY")

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

@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf")
)
async def rag_query_pdf(ctx: inngest.Context) -> RAGSearchResult:
    def _search(question: str, top_k: int= 3):
        query_vector = embed_query(question)
        store =QDrantStorage()
        found = store.search(query_vector, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data['question']
    
    top_k = ctx.event.data['top_k']
    found = await ctx.step.run('embed-and-search', lambda: _search(question, top_k), output_type=RAGSearchResult)
    context_block = "n\n".join(f"- {c}" for c in found.contexts)
    user_content= (
        "Use the following context to answer the question. \n\n"
        f"Context: \n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above only. If the answer is not in the provided context, say so." 
    )
    adapter=  ai.openai.Adapter(
        auth_key=os.getenv("OLLAMA_API_KEY", "ollama"),
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_LLM_MODEL,
    )
    
    response = await ctx.step.ai.infer(
        "llm-answer",
        adapter=adapter,
        body={
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.2,
        },
    )

    answer = response["choices"][0]["message"]["content"].strip()

    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts),
    ).model_dump()

    

app = FastAPI()

inngest.fast_api.serve(app, inngest_client, functions=[rag_ingest_pdf, rag_query_pdf])