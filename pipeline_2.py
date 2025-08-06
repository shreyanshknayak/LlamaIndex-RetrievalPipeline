
import time
from pathlib import Path
from typing import List, Any
import asyncio # Import asyncio for concurrent operations

from llama_index.core import Document, StorageContext, VectorStoreIndex, Settings
from llama_index.core.node_parser import HierarchicalNodeParser, get_leaf_nodes, get_root_nodes
from llama_index.core.retrievers import AutoMergingRetriever, BaseRetriever
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.readers.file import PyMuPDFReader
from llama_index.llms.groq import Groq
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


class Pipeline:
    """
    A pipeline to process a PDF, create nodes, and generate embeddings.
    It exposes a retriever to fetch nodes for a given query,
    but does not handle the answer generation itself. The embedding
    model is now passed in, not initialized internally.
    """

    def __init__(self, groq_api_key: str, pdf_path: str, embed_model: HuggingFaceEmbedding):
        """
        Initializes the pipeline with API keys, file path, and a pre-initialized embedding model.

        Args:
            groq_api_key (str): Your API key for Groq.
            pdf_path (str): The path to the PDF file to be processed.
            embed_model (HuggingFaceEmbedding): The pre-initialized embedding model.
        """
        self.groq_api_key = groq_api_key
        self.pdf_path = Path(pdf_path)
        self.embed_model = embed_model

        # Configure Llama-Index LLM setting only
        Settings.llm = Groq(model="llama3-70b-8192", api_key=self.groq_api_key)
        
        # Initialize components
        self.documents: List[Document] = []
        self.nodes: List[Any] = []
        self.storage_context: StorageContext | None = None
        self.index: VectorStoreIndex | None = None
        self.retriever: BaseRetriever | None = None
        self.leaf_nodes: List[Any] = []
        self.root_nodes: List[Any] = []


    def _parse_pdf(self) -> None:
        """Parses the PDF file into Llama-Index Document objects."""
        print(f"Parsing PDF at: {self.pdf_path}")
        start_time = time.perf_counter()
        loader = PyMuPDFReader()
        docs = loader.load(file_path=self.pdf_path)
        # Concatenate all document parts into a single document for simpler processing
        # Adjust this if you need to maintain per-page document context
        doc_text = "\n\n".join([d.get_content() for d in docs])
        self.documents = [Document(text=doc_text)]
        end_time = time.perf_counter()
        print(f"PDF parsing completed in {end_time - start_time:.2f} seconds.")

    def _create_nodes(self) -> None:
        """Creates hierarchical nodes from the parsed documents."""
        print("Creating nodes from documents...")
        start_time = time.perf_counter()
        node_parser = HierarchicalNodeParser.from_defaults()
        self.nodes = node_parser.get_nodes_from_documents(self.documents)
        self.leaf_nodes = get_leaf_nodes(self.nodes)
        self.root_nodes = get_root_nodes(self.nodes)
        end_time = time.perf_counter()
        print(f"Node creation completed in {end_time - start_time:.2f} seconds.")

    async def _generate_embeddings_concurrently(self) -> None:
        """
        Generates embeddings for leaf nodes concurrently using asyncio.to_thread
        and then builds the VectorStoreIndex.
        """
        print("Generating embeddings for leaf nodes concurrently...")
        start_time_embeddings = time.perf_counter()

        # Define a batch size for sending texts to the embedding model
        # Adjust this based on your system's memory and CPU/GPU capabilities
        BATCH_SIZE = 300 

        embedding_tasks = []
        # Extract text content from leaf nodes
        node_texts = [node.get_content() for node in self.leaf_nodes]

        # Create batches of texts and schedule embedding generation in separate threads
        for i in range(0, len(node_texts), BATCH_SIZE):
            batch_texts = node_texts[i : i + BATCH_SIZE]
            # Use asyncio.to_thread to run the synchronous embedding model call in a separate thread
            # This prevents blocking the main event loop
            embedding_tasks.append(asyncio.to_thread(self.embed_model.get_text_embedding_batch, texts=batch_texts, show_progress=False))

        # Wait for all concurrent embedding tasks to complete
        all_embeddings_batches = await asyncio.gather(*embedding_tasks)

        # Flatten the list of lists of embeddings into a single list
        flat_embeddings = [emb for sublist in all_embeddings_batches for emb in sublist]

        # Assign the generated embeddings back to their respective leaf nodes
        for i, node in enumerate(self.leaf_nodes):
            node.embedding = flat_embeddings[i]

        end_time_embeddings = time.perf_counter()
        print(f"Embeddings generated for {len(self.leaf_nodes)} nodes in {end_time_embeddings - start_time_embeddings:.2f} seconds.")

        # Now, build the VectorStoreIndex using the nodes that now have pre-computed embeddings
        print("Building VectorStoreIndex...")
        start_time_index_build = time.perf_counter()
        
        # Add all nodes (root and leaf) to the document store
        docstore = SimpleDocumentStore()
        docstore.add_documents(self.nodes)
        
        self.storage_context = StorageContext.from_defaults(docstore=docstore)
        
        # When nodes already have embeddings, VectorStoreIndex will use them
        self.index = VectorStoreIndex(
            self.leaf_nodes, # Pass leaf nodes which now contain their embeddings
            storage_context=self.storage_context,
            embed_model=self.embed_model # Still pass the embed_model, though it won't re-embed if nodes have embeddings
        )
        end_time_index_build = time.perf_counter()
        print(f"VectorStoreIndex built in {end_time_index_build - start_time_index_build:.2f} seconds.")
        print(f"Total index generation and embedding process completed in {end_time_index_build - start_time_embeddings:.2f} seconds.")


    def _setup_retriever(self) -> None:
        """Sets up the retriever."""
        print("Setting up retriever...")
        base_retriever = self.index.as_retriever(similarity_top_k=6)
        self.retriever = AutoMergingRetriever(
            base_retriever, storage_context=self.storage_context, verbose=True
        )

    async def run(self) -> None:
        """Runs the entire pipeline from parsing to retriever setup."""
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found at: {self.pdf_path}")

        self._parse_pdf()
        self._create_nodes()
        await self._generate_embeddings_concurrently() # Await the async embedding generation
        self._setup_retriever()
        print("Pipeline is ready for retrieval.")

    def retrieve_nodes(self, query_str: str) -> List[dict]:
        """
        Retrieves relevant nodes for a given query and converts them to a
        list of dictionaries for external use.

        Args:
            query_str (str): The query string.

        Returns:
            List[dict]: A list of dictionaries with node content and metadata.
        """
        if not self.retriever:
            raise RuntimeError("Retriever is not initialized. Run the pipeline first.")

        print(f"\nRetrieving nodes for query: '{query_str}'")
        start_time = time.perf_counter()
        
        # This is a synchronous call
        nodes = self.retriever.retrieve(query_str)
        
        end_time = time.perf_counter()
        print(f"Retrieval completed in {end_time - start_time:.2f} seconds. Found {len(nodes)} nodes.")

        # Convert the Llama-Index nodes to a dictionary format
        retrieved_results = [
            {
                "content": n.text,
                "document_metadata": n.metadata
            }
            for n in nodes
        ]
        return retrieved_results
