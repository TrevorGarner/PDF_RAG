#!/usr/bin/env python3
"""
DF_RAG - A tool for interacting with CSV and Excel data using natural language through an LLM
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from dotenv import load_dotenv
from fuzzywuzzy import process, fuzz

# Load environment variables
load_dotenv()


class LLMBackend:
    """Base class for LLM backends"""
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """Generate a response from the LLM"""
        raise NotImplementedError("Subclasses must implement this method")


class OllamaBackend(LLMBackend):
    """Ollama backend for LLM interactions"""
    
    def __init__(self, model: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the Ollama backend
        
        Args:
            model: The model to use (defaults to env var OLLAMA_MODEL or 'llama3')
            base_url: The base URL for the Ollama API (defaults to env var OLLAMA_BASE_URL or 'http://localhost:11434')
        """
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.api_url = f"{self.base_url}/api/generate"
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None, **kwargs) -> str:
        """
        Generate a response from the Ollama LLM
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt to guide the model
            **kwargs: Additional parameters to pass to the Ollama API
            
        Returns:
            The generated text response
        """
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            **kwargs
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = requests.post(self.api_url, json=payload)
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            error_msg = f"Error communicating with Ollama: {str(e)}"
            print(error_msg)
            return error_msg


class LLMFactory:
    """Factory for creating LLM backends"""
    
    @staticmethod
    def create(backend_type: str = "ollama", **kwargs) -> LLMBackend:
        """
        Create an LLM backend
        
        Args:
            backend_type: The type of backend to create ('ollama' is currently the only supported type)
            **kwargs: Additional parameters to pass to the backend constructor
            
        Returns:
            An LLM backend instance
        """
        if backend_type.lower() == "ollama":
            return OllamaBackend(**kwargs)
        else:
            raise ValueError(f"Unsupported LLM backend type: {backend_type}")


class DataHandler:
    """Handler for loading and processing data from CSV and Excel files"""
    
    def __init__(self, path: Optional[str] = None):
        """
        Initialize the data handler
        
        Args:
            path: Optional path to a data file or directory to load
        """
        self.dataframes = {}  # Dictionary to store multiple dataframes
        self.metadata = {}    # Dictionary to store metadata for each file
        
        if path:
            self.load_path(path)
    
    def load_path(self, path: str) -> bool:
        """
        Load data from a file or directory
        
        Args:
            path: Path to the data file or directory
            
        Returns:
            True if loading was successful, False otherwise
        """
        if not os.path.exists(path):
            print(f"Path not found: {path}")
            return False
        
        if os.path.isdir(path):
            return self._load_directory(path)
        else:
            return self._load_file(path)
    
    def _load_directory(self, dir_path: str) -> bool:
        """Load all supported files from a directory"""
        success = False
        for file in os.listdir(dir_path):
            if file.lower().endswith(('.csv', '.xlsx', '.xls')):
                file_path = os.path.join(dir_path, file)
                if self._load_file(file_path):
                    success = True
        return success
    
    def _load_file(self, file_path: str) -> bool:
        """Load a single data file"""
        file_ext = os.path.splitext(file_path)[1].lower()
        file_name = os.path.basename(file_path)
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(file_path)
                file_type = 'csv'
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
                file_type = 'excel'
            else:
                print(f"Unsupported file type: {file_ext}")
                return False
            
            self.dataframes[file_name] = df
            self._generate_metadata(file_name, file_path, file_type)
            return True
        except Exception as e:
            print(f"Error loading {file_path}: {str(e)}")
            return False
    
    def _generate_metadata(self, file_name: str, file_path: str, file_type: str) -> None:
        """Generate metadata about a loaded dataframe"""
        df = self.dataframes.get(file_name)
        if df is None:
            return
        
        # Basic metadata
        self.metadata[file_name] = {
            'file_path': file_path,
            'file_type': file_type,
            'rows': len(df),
            'columns': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isna().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(df.select_dtypes(include=['datetime']).columns),
        }
        
        # Add summary statistics for numeric columns (using sampling for large datasets)
        self.metadata[file_name]['numeric_stats'] = {}
        sample_size = min(len(df), 10000)  # Limit sample size for large datasets
        df_sample = df.sample(n=sample_size) if len(df) > sample_size else df
        
        for col in self.metadata[file_name]['numeric_columns']:
            self.metadata[file_name]['numeric_stats'][col] = {
                'min': float(df[col].min()) if not pd.isna(df[col].min()) else None,
                'max': float(df[col].max()) if not pd.isna(df[col].max()) else None,
                'mean': float(df_sample[col].mean()) if not pd.isna(df_sample[col].mean()) else None,
                'median': float(df_sample[col].median()) if not pd.isna(df_sample[col].median()) else None,
                'std': float(df_sample[col].std()) if not pd.isna(df_sample[col].std()) else None,
            }
        
        # Add value counts for categorical columns (limited to top 10, using sampling)
        self.metadata[file_name]['categorical_stats'] = {}
        for col in self.metadata[file_name]['categorical_columns']:
            value_counts = df_sample[col].value_counts().head(10).to_dict()
            self.metadata[file_name]['categorical_stats'][col] = {str(k): int(v) for k, v in value_counts.items()}
    
    def get_dataframe_info(self, file_name: Optional[str] = None) -> str:
        """
        Get a string representation of the dataframe info
        
        Args:
            file_name: Optional specific file to get info for
            
        Returns:
            A string with information about the dataframe(s)
        """
        if not self.dataframes:
            return "No data loaded"
        
        info = []
        files_to_process = [file_name] if file_name else self.dataframes.keys()
        
        for fname in files_to_process:
            if fname not in self.dataframes:
                continue
                
            df = self.dataframes[fname]
            metadata = self.metadata[fname]
            
            info.append(f"\nFile: {metadata['file_path']} ({metadata['file_type']})")
            info.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            info.append("Columns:")
            
            for col in df.columns:
                dtype = df[col].dtype
                missing = df[col].isna().sum()
                missing_pct = (missing / len(df)) * 100
                info.append(f"  - {col} ({dtype}): {missing} missing values ({missing_pct:.1f}%)")
        
        return "\n".join(info)
    
    def find_columns(self, query: str, file_name: Optional[str] = None, threshold: int = 70) -> Dict[str, List[str]]:
        """
        Find columns that match the query using fuzzy matching
        
        Args:
            query: The search query
            file_name: Optional specific file to search in
            threshold: The minimum similarity score (0-100)
            
        Returns:
            A dictionary mapping file names to lists of matching column names
        """
        results = {}
        files_to_process = [file_name] if file_name else self.dataframes.keys()
        
        for fname in files_to_process:
            if fname not in self.dataframes:
                continue
                
            df = self.dataframes[fname]
            if not df.columns.any():
                continue
            
            matches = process.extractBests(
                query, 
                df.columns, 
                scorer=fuzz.token_sort_ratio, 
                score_cutoff=threshold
            )
            
            if matches:
                results[fname] = [match[0] for match in matches]
        
        return results

    def get_context_for_llm(self, max_rows: int = 100, file_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get context information about the dataframe(s) for the LLM
        
        Args:
            max_rows: Maximum number of rows to include in the sample per file
            file_name: Optional specific file to get context for
            
        Returns:
            A dictionary with context information
        """
        if not self.dataframes:
            return {"error": "No data loaded"}
        
        context = {"files": {}}
        files_to_process = [file_name] if file_name else self.dataframes.keys()
        
        for fname in files_to_process:
            if fname not in self.dataframes:
                continue
                
            df = self.dataframes[fname]
            metadata = self.metadata[fname]
            
            # Use smart sampling based on data size
            sample_size = min(max_rows, len(df))
            if len(df) > max_rows * 10:  # For very large datasets
                # Take samples from start, middle, and end
                start_sample = df.head(sample_size // 3)
                middle_start = len(df) // 2 - (sample_size // 3) // 2
                middle_sample = df.iloc[middle_start:middle_start + sample_size // 3]
                end_sample = df.tail(sample_size // 3)
                sample_df = pd.concat([start_sample, middle_sample, end_sample])
            else:
                sample_df = df.sample(n=sample_size)
            
            context["files"][fname] = {
                "file_info": {
                    "path": metadata['file_path'],
                    "type": metadata['file_type'],
                },
                "dataframe_info": {
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "dtypes": metadata['dtypes'],
                },
                "sample_data": sample_df.to_dict(orient='records'),
                "summary_stats": {
                    "numeric": metadata.get('numeric_stats', {}),
                    "categorical": metadata.get('categorical_stats', {})
                }
            }
        
        return context


class DFRAG:
    """
    A tool for interacting with CSV and Excel data using natural language through an LLM
    """
    
    def __init__(
        self, 
        path: Optional[str] = None,
        llm_backend: str = "ollama",
        llm_model: Optional[str] = None,
        max_context_rows: int = 100
    ):
        """
        Initialize the DFRAG tool
        
        Args:
            path: Optional path to a data file or directory to load
            llm_backend: The LLM backend to use (currently only 'ollama' is supported)
            llm_model: The model to use with the LLM backend
            max_context_rows: Maximum number of rows to include in the context per file
        """
        self.data_handler = DataHandler(path)
        self.llm = LLMFactory.create(llm_backend, model=llm_model)
        self.max_context_rows = max_context_rows
        self.conversation_history = []
        self.current_context = None  # Store the last used context
    
    def load_path(self, path: str) -> bool:
        """
        Load data from a file or directory
        
        Args:
            path: Path to the data file or directory
            
        Returns:
            True if loading was successful, False otherwise
        """
        return self.data_handler.load_path(path)
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM
        
        Returns:
            The system prompt string
        """
        if not self.data_handler.dataframes:
            return "You are an AI assistant that helps users analyze data. No data is currently loaded."
        
        df_info = self.data_handler.get_dataframe_info()
        
        system_prompt = f"""You are an AI assistant that helps users analyze data from multiple files.
        
The available data has the following structure:
{df_info}

Your task is to:
1. Understand the user's question about the data
2. Provide accurate and helpful answers based on the data provided
3. When appropriate, suggest additional insights or visualizations that might be helpful
4. If you're unsure about something or need more information, ask clarifying questions
5. When multiple files are available, specify which file(s) you're using for the analysis

Important guidelines:
- Base your answers ONLY on the data provided, not on external knowledge
- If the user asks for something that cannot be determined from the data, explain why
- Be precise and accurate in your descriptions of the data
- When referring to columns, use the exact column names from the data and specify which file they're from
- If the user's question is ambiguous, ask for clarification
- Format numerical results appropriately (e.g., use appropriate decimal places, units)
- When describing trends or patterns, provide specific examples from the data
"""
        
        return system_prompt
    
    def _analyze_query(self, query: str) -> Dict[str, Any]:
        """
        Analyze the query to determine relevant files and columns
        
        Args:
            query: The user's query
            
        Returns:
            Dictionary with analysis results
        """
        # First, get all column names across all files
        all_columns = set()
        for df in self.data_handler.dataframes.values():
            all_columns.update(df.columns)
        
        # Find potential column references in the query
        relevant_columns = []
        for col in all_columns:
            if col.lower() in query.lower():
                relevant_columns.append(col)
        
        # Find potential file references
        relevant_files = []
        for fname in self.data_handler.dataframes.keys():
            if fname.lower() in query.lower():
                relevant_files.append(fname)
        
        return {
            "relevant_columns": relevant_columns,
            "relevant_files": relevant_files,
            "all_files": list(self.data_handler.dataframes.keys())
        }
    
    def _build_user_prompt(self, query: str) -> str:
        """
        Build the user prompt for the LLM, including relevant data context
        
        Args:
            query: The user's query
            
        Returns:
            The complete user prompt string
        """
        # Analyze the query to determine relevant context
        analysis = self._analyze_query(query)
        
        # If specific files are mentioned, use only those
        target_files = analysis["relevant_files"] if analysis["relevant_files"] else None
        
        # Get context with smart sampling
        context = self.data_handler.get_context_for_llm(
            self.max_context_rows,
            file_name=target_files[0] if len(target_files) == 1 else None
        )
        
        # Store the current context for potential follow-up queries
        self.current_context = context
        
        # Convert context to a formatted string
        context_str = json.dumps(context, indent=2)
        
        user_prompt = f"""
User Query: {query}

Data Context:
{context_str}

Query Analysis:
- Relevant columns detected: {', '.join(analysis['relevant_columns']) if analysis['relevant_columns'] else 'None'}
- Files mentioned in query: {', '.join(analysis['relevant_files']) if analysis['relevant_files'] else 'None'}
- Available files: {', '.join(analysis['all_files'])}

Please analyze the data and answer the query based on the provided context.
If you need information from files not included in the current context, please indicate this in your response.
"""
        
        return user_prompt
    
    def query(self, query: str) -> str:
        """
        Query the data using natural language
        
        Args:
            query: The user's natural language query
            
        Returns:
            The LLM's response
        """
        if not self.data_handler.dataframes:
            return "No data is currently loaded. Please load a CSV or Excel file first."
        
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(query)
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Generate response from LLM
        response = self.llm.generate(user_prompt, system_prompt=system_prompt)
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response
    
    def get_dataframes(self) -> Dict[str, Any]:
        """
        Get all current dataframes
        
        Returns:
            Dictionary mapping file names to DataFrame objects
        """
        return self.data_handler.dataframes
    
    def get_dataframe_info(self, file_name: Optional[str] = None) -> str:
        """
        Get information about the current dataframe(s)
        
        Args:
            file_name: Optional specific file to get info for
            
        Returns:
            A string with information about the dataframe(s)
        """
        return self.data_handler.get_dataframe_info(file_name)
    
    def filter_data(self, filter_query: Dict[str, Any]) -> None:
        """
        Filter the dataframe based on a query dictionary
        
        Args:
            filter_query: A dictionary with column names as keys and filter values
        """
        if not self.data_handler.dataframes:
            return
        
        for fname, df in self.data_handler.dataframes.items():
            self.data_handler.dataframes[fname] = self.data_handler.filter_dataframe(filter_query)
            self.data_handler._generate_metadata(fname, self.data_handler.metadata[fname]['file_path'], self.data_handler.metadata[fname]['file_type'])
    
    def reset_filters(self) -> None:
        """Reset any filters applied to the dataframe"""
        if self.data_handler.dataframes:
            for fname in self.data_handler.dataframes.keys():
                self.data_handler.load_path(self.data_handler.metadata[fname]['file_path'])
    
    def clear_conversation(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []
        self.current_context = None


# Simple usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        path = sys.argv[1]
        rag = DFRAG(path)
        
        print(f"Loaded data from {path}")
        print(rag.get_dataframe_info())
        
        while True:
            query = input("\nAsk a question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            print("\nThinking...")
            response = rag.query(query)
            print(f"\nResponse: {response}")
    else:
        print("Usage: python df_rag.py <path_to_file_or_directory>")
        print("Example: python df_rag.py data.csv")
        print("Example: python df_rag.py ./data_directory") 