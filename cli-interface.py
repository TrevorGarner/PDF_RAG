#!/usr/bin/env python3
import argparse
import os
import json
import logging
from typing import List, Dict

from pdf_rag_system import EmbeddingService, LLMService, PDFRagAgent

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class PDFRagCLI:
    """Command-line interface for the PDF RAG system."""
    
    def __init__(self):
        self.agent = None
    
    def initialize(self, data_dir: str, force_reload: bool = False) -> None:
        """Initialize the RAG agent."""
        # Validate data directory
        if not os.path.exists(data_dir):
            raise ValueError(f"Data directory does not exist: {data_dir}")
        
        if not os.path.isdir(data_dir):
            raise ValueError(f"Specified path is not a directory: {data_dir}")
            
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
        if not pdf_files:
            raise ValueError(f"No PDF files found in directory: {data_dir}")
            
        logger.info(f"Found {len(pdf_files)} PDF files in {data_dir}")
        
        # Initialize services and agent
        embedding_service = EmbeddingService()
        llm_service = LLMService()
        self.agent = PDFRagAgent(
            data_dir=data_dir,
            embedding_service=embedding_service,
            llm_service=llm_service
        )
        self.agent.initialize(force_reload=force_reload)
        logger.info(f"Agent initialized with documents from {data_dir}")
        
        return self.agent.get_document_list()
    
    def list_documents(self) -> dict:
        """List all available documents."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return {"documents": self.agent.get_document_list()}
    
    def answer_query(self, query: str, top_k: int = 5) -> dict:
        """Answer a query using the RAG system."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        return self.agent.answer_query(query)
    
    def summarize(self, doc_ids: list = None) -> dict:
        """Summarize specified documents."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        summary = self.agent.summarize_documents(doc_ids)
        return {"summary": summary}
    
    def compare(self, doc_ids: list) -> dict:
        """Compare specified documents."""
        if not self.agent:
            raise RuntimeError("Agent not initialized. Call initialize() first.")
        comparison = self.agent.compare_documents(doc_ids)
        return {"comparison": comparison}
    
    def save_result(self, result: dict, output_file: str) -> None:
        """Save results to a file."""
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)

def main():
    parser = argparse.ArgumentParser(description="PDF RAG System CLI")
    parser.add_argument("--data-dir", required=True, help="Directory containing PDF files")
    parser.add_argument("--force-reload", action="store_true", help="Force reload documents and rebuild index")
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List documents command
    list_parser = subparsers.add_parser("list", help="List available documents")
    list_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Query command
    query_parser = subparsers.add_parser("query", help="Answer a query based on documents")
    query_parser.add_argument("--query", required=True, help="The query to answer")
    query_parser.add_argument("--top-k", type=int, default=5, help="Number of chunks to retrieve")
    query_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Summarize documents")
    summarize_parser.add_argument("--doc-ids", nargs="*", help="Document IDs to summarize (defaults to all)")
    summarize_parser.add_argument("--output", help="Output file for results (JSON)")
    
    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare multiple documents")
    compare_parser.add_argument("--doc-ids", nargs="+", required=True, help="Document IDs to compare")
    compare_parser.add_argument("--output", help="Output file for results (JSON)")
    
    args = parser.parse_args()
    
    try:
        # Initialize CLI and agent
        cli = PDFRagCLI()
        documents = cli.initialize(args.data_dir, args.force_reload)
        
        # Execute requested command
        result = None
        if args.command == "list":
            result = cli.list_documents()
        elif args.command == "query":
            result = cli.answer_query(args.query, args.top_k)
        elif args.command == "summarize":
            result = cli.summarize(args.doc_ids)
        elif args.command == "compare":
            result = cli.compare(args.doc_ids)
        else:
            parser.print_help()
            return
        
        # Handle output
        if hasattr(args, 'output') and args.output:
            cli.save_result(result, args.output)
        else:
            print(json.dumps(result, indent=2))
            
    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        exit(1)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()