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

    def filter_by_value(self, query: str, threshold: float = 80.0) -> Dict[str, pd.DataFrame]:
        """
        Filter dataframes by matching cell values using fuzzy matching
        
        Args:
            query: The search value to match against cell values
            threshold: Minimum similarity score (0-100) for fuzzy matching
            
        Returns:
            Dictionary of filtered dataframes
        """
        filtered_dfs = {}
        
        for fname, df in self.dataframes.items():
            # Create a mask for matching rows
            mask = pd.Series(False, index=df.index)
            
            # Check each column for matches
            for col in df.columns:
                # Convert column to string for matching
                str_series = df[col].astype(str)
                
                # Apply fuzzy matching to each cell
                matches = str_series.apply(lambda x: fuzz.ratio(query.lower(), x.lower()) > threshold)
                mask = mask | matches
            
            # If we found any matches, add the filtered dataframe
            if mask.any():
                filtered_dfs[fname] = df[mask].copy()
        
        return filtered_dfs

    def filter_by_columns(self, columns: List[str], threshold: float = 70.0) -> Dict[str, pd.DataFrame]:
        """
        Filter dataframes to include only columns that match the given names
        
        Args:
            columns: List of column names to match
            threshold: Minimum similarity score (0-100) for fuzzy matching
            
        Returns:
            Dictionary of filtered dataframes with matching columns
        """
        filtered_dfs = {}
        
        for fname, df in self.dataframes.items():
            # Find matching columns using fuzzy matching
            matched_cols = []
            for col_query in columns:
                matches = process.extractBests(
                    col_query,
                    df.columns,
                    scorer=fuzz.ratio,
                    score_cutoff=threshold
                )
                matched_cols.extend([match[0] for match in matches])
            
            # If we found any matching columns, create filtered dataframe
            if matched_cols:
                filtered_dfs[fname] = df[list(set(matched_cols))].copy()
        
        return filtered_dfs


class DFRAG:
    """
    A tool for interacting with CSV and Excel data using natural language through an LLM
    """
    
    def __init__(
        self, 
        path: Optional[str] = None,
        llm_backend: str = "ollama",
        llm_model: Optional[str] = None,
        max_context_rows: int = 100,
        max_context_size: int = 4000  # Approximate max characters for context
    ):
        """
        Initialize the DFRAG tool
        
        Args:
            path: Optional path to a data file or directory to load
            llm_backend: The LLM backend to use (currently only 'ollama' is supported)
            llm_model: The model to use with the LLM backend
            max_context_rows: Maximum number of rows to include in the context per file
            max_context_size: Approximate max characters for context
        """
        self.data_handler = DataHandler(path)
        self.llm = LLMFactory.create(llm_backend, model=llm_model)
        self.max_context_rows = max_context_rows
        self.max_context_size = max_context_size
        self.messages = []
        self.current_context = None
    
    def _build_system_prompt(self) -> Dict[str, str]:
        """Build the system message"""
        if not self.data_handler.dataframes:
            content = "You are an AI assistant that helps users analyze data. No data is currently loaded."
        else:
            content = """You are an AI assistant that helps users analyze data from CSV and Excel files.
            
Your task is to:
1. Understand the user's question about the data
2. Provide accurate and helpful answers based on the data provided
3. When appropriate, suggest additional insights or visualizations
4. If you're unsure about something or need more information, ask clarifying questions

Important guidelines:
- Base your answers ONLY on the data provided
- Be precise and accurate in your descriptions
- Format numerical results appropriately
- When describing trends or patterns, provide specific examples"""

        return {"role": "system", "content": content}
    
    def _analyze_query_for_filtering(self, query: str) -> Dict[str, Any]:
        """
        Use LLM to analyze query and determine filtering strategy
        
        Returns dict with filtering recommendations
        """
        metadata = {}
        for fname, df in self.data_handler.dataframes.items():
            metadata[fname] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
            }
        
        filter_prompt = f"""Given the following query and available data, suggest how to filter the data to best answer the query.
Only respond with a JSON object containing:
1. relevant_columns: List of column names that are most relevant
2. filter_values: List of specific values to filter by
3. reasoning: Brief explanation of your filtering strategy

Query: {query}

Available Data:
{json.dumps(metadata, indent=2)}

Response should be valid JSON like:
{{
    "relevant_columns": ["col1", "col2"],
    "filter_values": ["value1", "value2"],
    "reasoning": "brief explanation"
}}"""

        response = self.llm.generate(filter_prompt)
        try:
            return json.loads(response)
        except:
            return {
                "relevant_columns": [],
                "filter_values": [],
                "reasoning": "Could not determine filtering strategy"
            }
    
    def _filter_data_intelligently(self, query: str) -> Dict[str, pd.DataFrame]:
        """Filter data based on query analysis and content size"""
        analysis = self._analyze_query_for_filtering(query)
        filtered_dfs = {}
        
        # Try column-based filtering first if columns were identified
        if analysis["relevant_columns"]:
            filtered_dfs = self.data_handler.filter_by_columns(analysis["relevant_columns"])
        
        # If we have specific values to filter by, apply value filtering
        if analysis["filter_values"] and (not filtered_dfs or self._context_too_large(filtered_dfs)):
            for value in analysis["filter_values"]:
                value_filtered = self.data_handler.filter_by_value(str(value))
                # Merge with column filtering results if they exist
                if filtered_dfs:
                    for fname in filtered_dfs:
                        if fname in value_filtered:
                            filtered_dfs[fname] = filtered_dfs[fname][
                                filtered_dfs[fname].index.isin(value_filtered[fname].index)
                            ]
                else:
                    filtered_dfs = value_filtered
        
        return filtered_dfs
    
    def _context_too_large(self, dfs: Dict[str, pd.DataFrame]) -> bool:
        """Check if the context from the dataframes would be too large"""
        context_size = 0
        for df in dfs.values():
            # Estimate size of sample data
            sample = df.head(self.max_context_rows)
            context_size += len(str(sample))
        
        return context_size > self.max_context_size
    
    def _get_data_context(self, query: str) -> str:
        """Get the current data context as a string, filtering if needed"""
        dfs = self.data_handler.dataframes
        context = []
        
        # If context would be too large, apply filtering
        if self._context_too_large(dfs):
            filtered_dfs = self._filter_data_intelligently(query)
            if filtered_dfs:
                dfs = filtered_dfs
        
        # Build context string
        for fname, df in dfs.items():
            context.append(f"\nFile: {fname}")
            context.append(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
            context.append("Sample data (first 5 rows):")
            context.append(df.head().to_string())
            context.append("\n")
        
        return "\n".join(context)
    
    def _build_user_prompt(self, query: str) -> Dict[str, str]:
        """Build the user message with context"""
        context = self._get_data_context(query)
        content = f"""Question: {query}

Available Data:
{context}

Please analyze the data and provide a clear, concise answer."""
        
        return {"role": "user", "content": content}
    
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
        
        # Initialize messages if empty
        if not self.messages:
            self.messages = [self._build_system_prompt()]
        
        # Add user message with filtered context
        user_message = self._build_user_prompt(query)
        self.messages.append(user_message)
        
        # Generate response from LLM
        response = self.llm.generate(
            prompt=user_message["content"],
            system_prompt=self.messages[0]["content"]
        )
        
        # Add assistant message
        assistant_message = {"role": "assistant", "content": response}
        self.messages.append(assistant_message)
        
        return response
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history in standard format"""
        return self.messages
    
    def clear_conversation(self) -> None:
        """Clear the conversation history"""
        self.messages = []


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