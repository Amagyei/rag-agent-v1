from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter, HierarchicalNodeParser
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.core.schema import Node
from dotenv import load_dotenv


EMBED_MODEL =''

EMBED_DIM = 768
parent_splitter = SentenceSplitter(chunk_size=512, chunk_overlap=128, include_prev_next_rel=True)
child_splitter = SentenceSplitter(chunk_size=128, chunk_overlap=32, include_prev_next_rel=True)
node_parser = HierarchicalNodeParser(
    chunk_sizes=[512, 128],
    node_parser_ids=["parent_chunk", "child_chunk"],
    node_parser_map={"parent_chunk": parent_splitter, "child_chunk": child_splitter}
    )

# def load_and_create_nodes_from_pdf(path: str):
#     docs = PDFReader().load_data(file=path)
#     nodes = node_parser.get_nodes_from_documents(docs)
#     return nodes

# node_parser = SentenceWindowNodeParser(
#     sentence_splitter=SentenceSplitter(chunk_size=128, chunk_overlap=32),
#     include_prev_next_rel=True,
#     include_metadata=True,
#     window_size=3,
# )


### use simple directory reader to load all the pdfs###


def load_and_create_nodes_from_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    nodes = node_parser.get_nodes_from_documents(docs)
    return nodes


### track node meta data and track what gets embedded ###
embed_model = OllamaEmbedding(
    model_name="nomic-embed-text",
    base_url="http://localhost:11434",
)

def embed_nodes(nodes: list[Node]):
    embeddings = embed_model.get_text_embedding_batch([n.text for n in nodes])
    for node, emb in zip(nodes, embeddings):
        node.embedding = emb
    return nodes

def embed_query(query: str):
    return embed_model.get_text_embedding(query)