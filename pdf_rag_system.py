import os
import json
import pickle
from typing import List, Dict, Tuple, Optional, Any, Union
import numpy as np
from dataclasses import dataclass, field
import fitz  # PyMuPDF
from rank_bm25 import BM25Okapi
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import re
from collections import defaultdict
import logging
import requests  # Add this import at the top with other imports
import ollama
# pymupdf4llm 
# python cli-interface.py --data-dir ./documents query --query "What did Trevor do from May 2023 to June 2024"


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Download NLTK resources if not already present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt_tab')

# Data structures
@dataclass
class Chunk:
    """Represents a chunk of text from a document with metadata."""
    text: str
    page_num: int
    section: Optional[str] = None
    embedding: Optional[np.ndarray] = None
    doc_id: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def __post_init__(self):
        # Clean the text to remove extra whitespace
        self.text = re.sub(r'\s+', ' ', self.text).strip()
    
    def __hash__(self):
        # Create a unique hash based on the chunk's content and metadata
        return hash((self.text, self.page_num, self.doc_id))
    
    def __eq__(self, other):
        if not isinstance(other, Chunk):
            return False
        return (self.text == other.text and 
                self.page_num == other.page_num and 
                self.doc_id == other.doc_id)

@dataclass
class Document:
    """Represents a document with metadata and chunks."""
    doc_id: str
    title: str
    path: str
    num_pages: int = 0
    summary: str = ""
    chunks: List[Chunk] = field(default_factory=list)
    embedding: Optional[np.ndarray] = None
    
    def __post_init__(self):
        # Clean the title
        self.title = re.sub(r'\.pdf$', '', os.path.basename(self.path))

class PDFProcessor:
    """Handles PDF processing, extraction, and chunking."""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        
    def get_pdf_files(self) -> List[str]:
        """Get all PDF files in the data directory."""
        pdf_files = []
        for file in os.listdir(self.data_dir):
            if file.lower().endswith('.pdf'):
                pdf_files.append(os.path.join(self.data_dir, file))
        return pdf_files
    
    def extract_text_and_sections(self, pdf_path: str) -> Tuple[List[str], Dict[int, str]]:
        """Extract text from PDF pages and identify sections."""
        doc = fitz.open(pdf_path)
        page_texts = []
        section_headers = {}
        
        for page_num, page in enumerate(doc):
            # Extract text from the page
            text = page.get_text()
            page_texts.append(text)
            
            # Try to identify section headers using font size and style
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            # Heuristic: larger font or bold might be a section header
                            if span["size"] > 12 or span["flags"] & 16:  # 16 is bold
                                potential_header = span["text"].strip()
                                if potential_header and len(potential_header) < 100:  # Reasonable section title length
                                    section_headers[page_num] = potential_header
                                    break
        
        doc.close()
        return page_texts, section_headers
    
    def chunk_by_size(self, text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
        """Chunk text by character count with overlap."""
        chunks = []
        start = 0
        
        while start < len(text):
            end = min(start + max_chars, len(text))
            
            # Try to end at a sentence boundary if possible
            if end < len(text):
                # Find the last sentence break within our chunk
                last_period = text.rfind('. ', start, end)
                if last_period != -1 and last_period > start + max_chars // 2:
                    end = last_period + 1  # Include the period
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
            
        return chunks
    
    def chunk_by_semantic(self, text: str) -> List[str]:
        """Chunk text by semantic units (paragraphs, sentences)."""
        # Split by paragraph first
        paragraphs = text.split('\n\n')
        chunks = []
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # If paragraph is short enough, keep it as a chunk
            if len(para) < 1500:
                chunks.append(para)
            else:
                # Otherwise, split into sentences
                sentences = sent_tokenize(para)
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    # If adding this sentence exceeds threshold, store current chunk and start a new one
                    if current_length + len(sentence) > 1500:
                        if current_chunk:
                            chunks.append(' '.join(current_chunk))
                        current_chunk = [sentence]
                        current_length = len(sentence)
                    else:
                        current_chunk.append(sentence)
                        current_length += len(sentence)
                
                # Don't forget the last chunk
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def adaptive_chunking(self, text: str, doc_length: int) -> List[str]:
        """Choose chunking strategy based on document length."""
        if doc_length <= 10:  # Short document (under 10 pages)
            return self.chunk_by_semantic(text)
        else:  # Longer document
            return self.chunk_by_size(text, max_chars=1500, overlap=300)
    
    def process_document(self, pdf_path: str) -> Document:
        """Process a PDF document, extract text, and create chunks."""
        doc_id = os.path.basename(pdf_path).replace('.pdf', '')
        
        try:
            # Open PDF
            pdf_doc = fitz.open(pdf_path)
            num_pages = len(pdf_doc)
            
            # Create document object
            document = Document(
                doc_id=doc_id,
                title=doc_id,  # Will be updated if title is found
                path=pdf_path,
                num_pages=num_pages
            )
            
            # Extract text and sections
            page_texts, section_headers = self.extract_text_and_sections(pdf_path)
            
            # Extract potential title from first page
            if page_texts:
                first_page = page_texts[0]
                title_match = re.search(r'^(.+?)(?:\n|$)', first_page.strip())
                if title_match:
                    potential_title = title_match.group(1).strip()
                    if len(potential_title) < 100:  # Reasonable title length
                        document.title = potential_title
            
            # Process each page
            all_text = ""
            for page_num, page_text in enumerate(page_texts):
                all_text += page_text
                
                # Get current section header if available
                current_section = None
                for p in range(page_num, -1, -1):
                    if p in section_headers:
                        current_section = section_headers[p]
                        break
                
                # Create chunks for this page
                chunks = self.chunk_by_semantic(page_text) if len(page_text) < 5000 else self.chunk_by_size(page_text)
                
                for i, chunk_text in enumerate(chunks):
                    if not chunk_text.strip():
                        continue
                        
                    chunk = Chunk(
                        text=chunk_text,
                        page_num=page_num + 1,  # 1-indexed for user-friendly display
                        section=current_section,
                        doc_id=doc_id
                    )
                    document.chunks.append(chunk)
                    
            pdf_doc.close()
            return document
            
        except Exception as e:
            logger.error(f"Error processing document {pdf_path}: {str(e)}")
            raise
    
    def process_all_documents(self) -> List[Document]:
        """Process all PDF documents in the data directory."""
        pdf_files = self.get_pdf_files()
        documents = []
        
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path}")
            try:
                document = self.process_document(pdf_path)
                logger.info(f"Successfully processed {pdf_path}: {len(document.chunks)} chunks created")
                documents.append(document)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path}: {str(e)}")
                
        logger.info(f"Total documents processed: {len(documents)} with {sum(len(doc.chunks) for doc in documents)} chunks")
        return documents

class EmbeddingService:
    """Interface for embedding service."""
    
    def __init__(self, api_endpoint: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.api_endpoint = api_endpoint
        self.model = model
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for a text using Ollama API."""
        try:
            response = ollama.embeddings(
                model=self.model,
                prompt=text
            )
            
            # The embeddings are returned directly in the response
            embedding = np.array(response.embedding)
            return embedding
            
        except Exception as e:
            logger.error(f"Error getting embedding: {str(e)}")
            # Fallback to random embedding in case of error
            return np.random.randn(768)
    
    def embed_documents(self, documents: List[Document]) -> None:
        """Embed all documents and their chunks."""
        for doc in documents:
            # Embed document summary if available
            if doc.summary:
                doc.embedding = self.get_embedding(doc.summary)
            
            # Embed all chunks
            for chunk in doc.chunks:
                chunk.embedding = self.get_embedding(chunk.text)
                
        logger.info(f"Embedded {sum(len(doc.chunks) for doc in documents)} chunks across {len(documents)} documents")

class LLMService:
    """Interface for LLM service."""
    
    def __init__(self, api_endpoint: str = "http://localhost:11434", model: str = "llama3.1"):
        self.api_endpoint = api_endpoint
        self.model = model
    
    def generate_text(self, prompt: str, context: str = None, max_tokens: int = 90000) -> str:
        """Generate text using Ollama API."""
        try:
            # Combine prompt and context if context is provided
            full_prompt = f"{prompt}\n\nContext: {context}" if context else prompt
            
            # Use the ollama library instead of raw requests
            response = ollama.chat(
                model=self.model,
                messages=[{
                    "role": "user",
                    "content": full_prompt
                }]
            )
            
            # Extract the response text
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating text: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def generate_document_summary(self, document: Document, max_tokens: int = 90000) -> str:
        """Generate a summary for a document."""
        # Create a prompt for summarization
        chunk_texts = [chunk.text for chunk in document.chunks[:50]]  # Limit to first 50 chunks to avoid token limits
        context = "\n\n".join(chunk_texts)
        
        prompt = f"""Please provide a comprehensive summary of the following document titled "{document.title}".
                    Focus on the main topics, key findings, and overall structure of the document.
                    Make the summary brief enough to understand what the document contains and what questions it might answer."""
        
        summary = self.generate_text(prompt, context, max_tokens)
        return summary

class RAGRetriever:
    """Handles retrieval of relevant document chunks."""
    
    def __init__(self, documents: List[Document], embedding_service: EmbeddingService):
        self.documents = documents
        self.embedding_service = embedding_service
        self.bm25_corpus = []
        self.bm25_indexes = {}
        self.chunk_map = {}  # Maps corpus index to document chunk
        self.initialize_indexes()
    
    def initialize_indexes(self):
        """Initialize BM25 indexes for all documents."""
        corpus_idx = 0
        
        # Process each document
        for doc_idx, doc in enumerate(self.documents):
            doc_corpus = []
            doc_chunk_map = {}
            
            # Process each chunk
            for chunk_idx, chunk in enumerate(doc.chunks):
                # Create BM25 entry
                tokenized_text = word_tokenize(chunk.text.lower())
                doc_corpus.append(tokenized_text)
                
                # Track the mapping from corpus index to chunk
                doc_chunk_map[len(doc_corpus) - 1] = (doc_idx, chunk_idx)
                self.chunk_map[corpus_idx] = (doc_idx, chunk_idx)
                corpus_idx += 1
            
            # Create BM25 index for this document
            if doc_corpus:
                self.bm25_indexes[doc.doc_id] = BM25Okapi(doc_corpus)
            
            # Add to global corpus
            self.bm25_corpus.extend(doc_corpus)
        
        # Create global BM25 index
        if self.bm25_corpus:
            self.global_bm25 = BM25Okapi(self.bm25_corpus)
            
        logger.info(f"Initialized BM25 indexes for {len(self.documents)} documents with {len(self.bm25_corpus)} chunks total")
    
    def vector_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for relevant chunks using vector similarity."""
        results = []
        
        for doc_idx, doc in enumerate(self.documents):
            for chunk_idx, chunk in enumerate(doc.chunks):
                if chunk.embedding is not None:
                    # Calculate cosine similarity
                    similarity = np.dot(query_embedding, chunk.embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk.embedding)
                    )
                    results.append((chunk, float(similarity)))
        
        # Sort by similarity (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    
    def bm25_search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """Search for relevant chunks using BM25."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.global_bm25.get_scores(tokenized_query)
        
        # Get top_k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            doc_idx, chunk_idx = self.chunk_map[idx]
            chunk = self.documents[doc_idx].chunks[chunk_idx]
            results.append((chunk, scores[idx]))
        
        return results
    
    def hybrid_search(self, query: str, query_embedding: np.ndarray, top_k: int = 5) -> List[Chunk]:
        """Combine vector and BM25 search results."""
        # Get results from both methods
        vector_results = self.vector_search(query_embedding, top_k=top_k)
        bm25_results = self.bm25_search(query, top_k=top_k)
        
        # Combine and deduplicate results
        # Convert scores to ranks for rank fusion
        vector_ranks = {chunk: 1/(rank+1) for rank, (chunk, _) in enumerate(vector_results)}
        bm25_ranks = {chunk: 1/(rank+1) for rank, (chunk, _) in enumerate(bm25_results)}
        
        # Combine using reciprocal rank fusion
        combined_scores = defaultdict(float)
        for chunk, _ in vector_results + bm25_results:
            if chunk in vector_ranks:
                combined_scores[chunk] += vector_ranks[chunk]
            if chunk in bm25_ranks:
                combined_scores[chunk] += bm25_ranks[chunk]
        
        # Sort by combined score
        ranked_chunks = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, _ in ranked_chunks[:top_k]]

class QueryPlanner:
    """Handles query decomposition and planning."""
    
    def __init__(self, llm_service: LLMService):
        self.llm = llm_service
    
    def decompose_query(self, query: str, documents: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """Break down a complex query into sub-queries using document context."""
        # Create context from document summaries
        doc_context = "\n\n".join([
            f"Document: {doc['title']}\nSummary: {doc['summary']}"
            for doc in documents
        ])
        
        prompt = f"""Break down the following query into specific sub-queries that directly relate to answering the main question.
        Each sub-query should be a specific question that helps gather information needed for the main query.

        Main Query: "{query}"

        Available Documents:
        {doc_context}

        Guidelines for breaking down the query:
        1. Each sub-query should directly relate to the main query
        2. Make sub-queries specific and focused, not generic
        3. Consider different aspects or components of the main query
        4. Target specific information mentioned in the query
        5. Consider relevant context from the available documents

        Output your response in this format:
        1. Sub-query: [specific question directly related to main query]
           Target Documents: [document titles]
           Focus: [specific information we're looking for]
           Relevance: [how this helps answer the main query]

        2. Sub-query: ...
        """
        
        response = self.llm.generate_text(prompt)
        
        # Parse the response into structured sub-queries
        sub_queries = []
        current_query = {}
        
        for line in response.split('\n'):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
                
            if line.startswith(('1.', '2.', '3.', '4.', '5.')):
                if current_query:
                    # Ensure query key exists before adding
                    if 'query' not in current_query and 'focus' in current_query:
                        current_query['query'] = current_query['focus']
                    sub_queries.append(current_query)
                current_query = {}
            elif line.startswith('Sub-query:'):
                current_query['query'] = line.split(':', 1)[1].strip()
            elif line.startswith('Target Documents:'):
                current_query['target_docs'] = [
                    doc.strip() for doc in line.split(':', 1)[1].strip().split(',')
                ]
            elif line.startswith('Focus:'):
                current_query['focus'] = line.split(':', 1)[1].strip()
                # Use focus as query if no explicit query was provided
                if 'query' not in current_query:
                    current_query['query'] = line.split(':', 1)[1].strip()
            elif line.startswith('Relevance:'):
                current_query['relevance'] = line.split(':', 1)[1].strip()
        
        # Don't forget to add the last query
        if current_query:
            # Ensure query key exists before adding
            if 'query' not in current_query and 'focus' in current_query:
                current_query['query'] = current_query['focus']
            sub_queries.append(current_query)
        
        # Final validation to ensure all sub-queries have the required keys
        validated_queries = []
        for sq in sub_queries:
            if 'query' in sq and sq['query']:  # Only include if it has a valid query
                # Ensure all required keys exist
                sq.setdefault('target_docs', [])
                sq.setdefault('focus', sq['query'])
                sq.setdefault('relevance', 'Not specified')
                validated_queries.append(sq)
        
        logger.info(validated_queries)
        return validated_queries

class ContextBuilder:
    """Manages iterative context gathering and refinement."""
    
    def __init__(self, retriever: RAGRetriever, llm_service: LLMService):
        self.retriever = retriever
        self.llm = llm_service
    
    def evaluate_context_relevance(self, context: str, query: str) -> Tuple[float, str]:
        """Evaluate how well the context answers the query."""
        prompt = f"""On a scale of 0-1, how well does the following context answer this query?
        Also explain what information is missing or could be improved.
        
        Query: {query}
        
        Context: {context}
        
        Format your response as:
        Score: [0-1]
        Explanation: [your explanation]
        """
        
        response = self.llm.generate_text(prompt)
        score = float(response.split('Score:')[1].split('\n')[0].strip())
        explanation = response.split('Explanation:')[1].strip()
        
        return score, explanation
    
    def gather_context(self, sub_query: Dict[str, Any], doc_ids: List[str] = None) -> List[Chunk]:
        """Gather context for a sub-query, potentially focusing on specific documents."""
        query_text = sub_query['query']
        query_embedding = self.retriever.embedding_service.get_embedding(query_text)
        
        # If specific documents are specified, filter the search
        if doc_ids:
            original_docs = self.retriever.documents
            self.retriever.documents = [doc for doc in original_docs if doc.doc_id in doc_ids]
        
        relevant_chunks = self.retriever.hybrid_search(query_text, query_embedding, top_k=5)
        
        # Restore original documents if they were filtered
        if doc_ids:
            self.retriever.documents = original_docs
        
        return relevant_chunks

class PDFRagAgent:
    """Enhanced agent that orchestrates the entire RAG pipeline with query planning."""
    
    def __init__(self, data_dir: str, embedding_service: EmbeddingService, llm_service: LLMService):
        self.data_dir = data_dir
        self.embedding_service = embedding_service
        self.llm_service = llm_service
        self.documents = []
        self.retriever = None
        self.is_initialized = False
        self.query_planner = QueryPlanner(llm_service)
        self.context_builder = None  # Will be initialized after retriever
    
    def initialize(self, force_reload: bool = False) -> None:
        """Initialize the agent by processing documents and building indexes."""
        cache_file = os.path.join(self.data_dir, "rag_cache.pkl")
        
        # Add logging to track document processing
        logger.info(f"Starting initialization with data_dir: {self.data_dir}")
        
        # Process documents
        logger.info("Processing PDF documents...")
        processor = PDFProcessor(self.data_dir)
        self.documents = processor.process_all_documents()
        logger.info(f"Processed {len(self.documents)} documents with {sum(len(doc.chunks) for doc in self.documents)} total chunks")
        
        # Generate summaries for documents
        logger.info("Generating document summaries...")
        for doc in self.documents:
            doc.summary = self.llm_service.generate_document_summary(doc)
            logger.info(f"Generated summary for document: {doc.doc_id}")
        
        # Embed documents and chunks
        logger.info("Embedding documents and chunks...")
        self.embedding_service.embed_documents(self.documents)
        
        # Initialize retriever with detailed logging
        logger.info("Initializing retriever...")
        self.retriever = RAGRetriever(self.documents, self.embedding_service)
        
        self.context_builder = ContextBuilder(self.retriever, self.llm_service)
        
        self.is_initialized = True
        logger.info("Initialization complete")
    
    def get_document_list(self) -> List[Dict]:
        """Get a list of available documents with metadata."""
        if not self.is_initialized:
            self.initialize()
            
        return [
            {
                "doc_id": doc.doc_id,
                "title": doc.title,
                "num_pages": doc.num_pages,
                "summary": doc.summary
            }
            for doc in self.documents
        ]
    
    def answer_query(self, query: str, **kwargs) -> Dict:
        """Enhanced query answering with query decomposition and iterative context gathering."""
        if not self.is_initialized:
            self.initialize()
        
        # Get document information for context
        doc_info = [
            {
                'title': doc.title,
                'summary': doc.summary
            }
            for doc in self.documents
        ]
        
        all_citations = []
        citation_counter = 1
        sub_query_answers = []
        
        # Step 1: Decompose the query with document context
        sub_queries = self.query_planner.decompose_query(query, doc_info)
        logger.info(f"Decomposed query into {len(sub_queries)} sub-queries")
        
        # If no sub-queries were generated, use the original query directly
        if not sub_queries:
            logger.info("No sub-queries generated. Using original query for hybrid search.")
            query_embedding = self.embedding_service.get_embedding(query)
            relevant_chunks = self.retriever.hybrid_search(query, query_embedding, top_k=5)
            
            contexts = []
            
            for chunk in relevant_chunks:
                doc = next((d for d in self.documents if d.doc_id == chunk.doc_id), None)
                contexts.append(f"[{citation_counter}] {chunk.text}")
                
                all_citations.append({
                    "id": citation_counter,
                    "document": doc.title if doc else chunk.doc_id,
                    "page": chunk.page_num,
                    "section": chunk.section or "N/A",
                    "sub_query": query
                })
                
                citation_counter += 1
            
            # Generate answer using gathered context
            context = "\n\n".join(contexts)
            answer = self.llm_service.generate_text(
                f"""Please answer this question using the provided context. Cite your sources using [n].
                Question: {query}
                Context: {context}""")
            
        else:
            # Process each sub-query independently
            for sub_query in sub_queries:
                logger.info(f"Processing sub-query: {sub_query['query']}")
                
                # Gather context for this sub-query
                relevant_chunks = self.context_builder.gather_context(sub_query)
                
                # Evaluate context quality
                context_text = "\n\n".join([chunk.text for chunk in relevant_chunks])
                relevance_score, feedback = self.context_builder.evaluate_context_relevance(
                    context_text, sub_query['query']
                )
                
                # If context quality is low, try to gather more context
                if relevance_score < 0.7:
                    logger.info(f"Initial context quality low ({relevance_score}). Gathering more context...")
                    additional_chunks = self.context_builder.gather_context(
                        sub_query,
                        doc_ids=[chunk.doc_id for chunk in relevant_chunks]
                    )
                    relevant_chunks.extend(additional_chunks)
                
                # Build context with citations for this sub-query
                contexts = []
                for chunk in relevant_chunks:
                    doc = next((d for d in self.documents if d.doc_id == chunk.doc_id), None)
                    contexts.append(f"[{citation_counter}] {chunk.text}")
                    
                    all_citations.append({
                        "id": citation_counter,
                        "document": doc.title if doc else chunk.doc_id,
                        "page": chunk.page_num,
                        "section": chunk.section or "N/A",
                        "sub_query": sub_query['query']
                    })
                    
                    citation_counter += 1
                
                # Generate answer for this sub-query
                sub_query_context = "\n\n".join(contexts)
                sub_query_answer = self.llm_service.generate_text(
                    f"""Please answer this specific question using the provided context. Cite your sources using [n].
                    Question: {sub_query['query']}
                    Context: {sub_query_context}""")
                
                sub_query_answers.append({
                    "question": sub_query['query'],
                    "answer": sub_query_answer,
                    "focus": sub_query['focus'],
                    "relevance": sub_query['relevance']
                })
            
            # Step 3: Generate final answer using sub-query answers as context
            sub_query_context = "\n\n".join([
                f"Question: {sqa['question']}\nAnswer: {sqa['answer']}\nRelevance: {sqa['relevance']}"
                for sqa in sub_query_answers
            ])
            
            prompt = f"""Please provide a comprehensive answer to the following main question.
            Use the answers from the sub-questions below to formulate your response.
            Maintain any source citations [n] from the sub-answers in your final response.
            
            Main Question: {query}
            
            Sub-question Answers:
            {sub_query_context}
            
            Guidelines:
            1. Synthesize information from all relevant sub-answers
            2. Maintain citation numbers [n] from the original answers
            3. Ensure your answer directly addresses the main question
            4. Acknowledge any gaps or uncertainties in the information
            5. Connect information logically across sub-answers
            """
            
            answer = self.llm_service.generate_text(prompt)
        
        return {
            "query": query,
            "answer": answer,
            "citations": all_citations,
            "sub_queries": sub_queries,
            "sub_query_answers": sub_query_answers if sub_queries else None
        }
    
    def summarize_documents(self, doc_ids: List[str] = None) -> str:
        """Summarize one or more documents."""
        if not self.is_initialized:
            self.initialize()
            
        if not doc_ids:
            # Summarize all documents
            docs_to_summarize = self.documents
        else:
            # Summarize specific documents
            docs_to_summarize = [doc for doc in self.documents if doc.doc_id in doc_ids]
        
        if not docs_to_summarize:
            return "No documents found to summarize."
        
        # Create context from document summaries
        context = "\n\n".join([
            f"Document: {doc.title}\nSummary: {doc.summary}"
            for doc in docs_to_summarize
        ])
        
        # Create prompt
        if len(docs_to_summarize) == 1:
            prompt = f"Please provide a detailed summary of the document titled '{docs_to_summarize[0].title}'."
        else:
            prompt = f"Please provide a comprehensive summary of the {len(docs_to_summarize)} documents described below."
        
        # Generate summary
        summary = self.llm_service.generate_text(prompt, context)
        return summary
    
    def compare_documents(self, doc_ids: List[str]) -> str:
        """Compare multiple documents, highlighting similarities and differences."""
        if not self.is_initialized:
            self.initialize()
            
        if len(doc_ids) < 2:
            return "Need at least two documents to compare."
            
        # Get the documents
        docs_to_compare = [doc for doc in self.documents if doc.doc_id in doc_ids]
        
        if len(docs_to_compare) < 2:
            return "Could not find at least two of the specified documents."
        
        # Create context from document summaries
        context = "\n\n".join([
            f"Document {i+1}: {doc.title}\nSummary: {doc.summary}"
            for i, doc in enumerate(docs_to_compare)
        ])
        
        # Create prompt
        doc_titles = ", ".join([f"'{doc.title}'" for doc in docs_to_compare])
        prompt = f"""Please compare the following documents: {doc_titles}.
                   Highlight the key similarities and differences in terms of:
                   1. Main topics and themes
                   2. Approaches or methodologies
                   3. Conclusions or findings
                   4. Any other significant aspects
                   
                   Organize your comparison in a clear, structured way."""
        
        # Generate comparison
        comparison = self.llm_service.generate_text(prompt, context)
        return comparison

# Example usage
def main():
    # Create services
    embedding_service = EmbeddingService(api_endpoint="http://localhost:11434", model="nomic-embed-text")
    llm_service = LLMService(api_endpoint="http://localhost:11434", model="llama3.1")
    
    # Initialize agent
    agent = PDFRagAgent(
        data_dir="./pdfs",
        embedding_service=embedding_service,
        llm_service=llm_service
    )
    
    # Initialize (process documents)
    agent.initialize()
    
    # Get document list
    documents = agent.get_document_list()
    print(f"Found {len(documents)} documents:")
    for doc in documents:
        print(f"- {doc['title']} ({doc['num_pages']} pages)")
    
    # Example query
    query = "What are the main findings of the study?"
    result = agent.answer_query(query)
    
    print(f"\nQuery: {result['query']}")
    print(f"Answer: {result['answer']}")
    print("\nCitations:")
    for citation in result['citations']:
        print(f"[{citation['id']}] {citation['document']}, Page {citation['page']}, Section: {citation['section']}")

if __name__ == "__main__":
    main()
