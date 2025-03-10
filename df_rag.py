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
    
    def __init__(self, file_path: Optional[str] = None):
        """
        Initialize the data handler
        
        Args:
            file_path: Optional path to a data file to load
        """
        self.df = None
        self.file_path = None
        self.file_type = None
        self.metadata = {}
        
        if file_path:
            self.load_data(file_path)
    
    def load_data(self, file_path: str) -> bool:
        """
        Load data from a CSV or Excel file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if loading was successful, False otherwise
        """
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return False
        
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                self.df = pd.read_csv(file_path)
                self.file_type = 'csv'
            elif file_ext in ['.xlsx', '.xls']:
                self.df = pd.read_excel(file_path)
                self.file_type = 'excel'
            else:
                print(f"Unsupported file type: {file_ext}")
                return False
            
            self.file_path = file_path
            self._generate_metadata()
            return True
        except Exception as e:
            print(f"Error loading data: {str(e)}")
            return False
    
    def _generate_metadata(self) -> None:
        """Generate metadata about the loaded dataframe"""
        if self.df is None:
            return
        
        # Basic metadata
        self.metadata = {
            'file_path': self.file_path,
            'file_type': self.file_type,
            'rows': len(self.df),
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'sample': self.df.head(5).to_dict(orient='records'),
            'missing_values': self.df.isna().sum().to_dict(),
            'numeric_columns': list(self.df.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(self.df.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(self.df.select_dtypes(include=['datetime']).columns),
        }
        
        # Add summary statistics for numeric columns
        self.metadata['numeric_stats'] = {}
        for col in self.metadata['numeric_columns']:
            self.metadata['numeric_stats'][col] = {
                'min': float(self.df[col].min()) if not pd.isna(self.df[col].min()) else None,
                'max': float(self.df[col].max()) if not pd.isna(self.df[col].max()) else None,
                'mean': float(self.df[col].mean()) if not pd.isna(self.df[col].mean()) else None,
                'median': float(self.df[col].median()) if not pd.isna(self.df[col].median()) else None,
                'std': float(self.df[col].std()) if not pd.isna(self.df[col].std()) else None,
            }
        
        # Add value counts for categorical columns (limited to top 10)
        self.metadata['categorical_stats'] = {}
        for col in self.metadata['categorical_columns']:
            value_counts = self.df[col].value_counts().head(10).to_dict()
            self.metadata['categorical_stats'][col] = {str(k): int(v) for k, v in value_counts.items()}
    
    def get_dataframe_info(self) -> str:
        """
        Get a string representation of the dataframe info
        
        Returns:
            A string with information about the dataframe
        """
        if self.df is None:
            return "No data loaded"
        
        info = []
        info.append(f"File: {self.file_path} ({self.file_type})")
        info.append(f"Shape: {self.df.shape[0]} rows × {self.df.shape[1]} columns")
        info.append("Columns:")
        
        for col in self.df.columns:
            dtype = self.df[col].dtype
            missing = self.df[col].isna().sum()
            missing_pct = (missing / len(self.df)) * 100
            info.append(f"  - {col} ({dtype}): {missing} missing values ({missing_pct:.1f}%)")
        
        return "\n".join(info)
    
    def find_columns(self, query: str, threshold: int = 70) -> List[str]:
        """
        Find columns that match the query using fuzzy matching
        
        Args:
            query: The search query
            threshold: The minimum similarity score (0-100)
            
        Returns:
            A list of matching column names
        """
        if self.df is None or not self.df.columns.any():
            return []
        
        matches = process.extractBests(
            query, 
            self.df.columns, 
            scorer=fuzz.token_sort_ratio, 
            score_cutoff=threshold
        )
        
        return [match[0] for match in matches]
    
    def filter_dataframe(self, query: Dict[str, Any]) -> pd.DataFrame:
        """
        Filter the dataframe based on a query dictionary
        
        Args:
            query: A dictionary with column names as keys and filter values
            
        Returns:
            A filtered dataframe
        """
        if self.df is None:
            return pd.DataFrame()
        
        filtered_df = self.df.copy()
        
        for col, value in query.items():
            if col not in filtered_df.columns:
                # Try fuzzy matching
                matches = self.find_columns(col)
                if matches:
                    col = matches[0]
                else:
                    continue
            
            if isinstance(value, (list, tuple)):
                filtered_df = filtered_df[filtered_df[col].isin(value)]
            elif isinstance(value, dict):
                if 'min' in value and 'max' in value:
                    filtered_df = filtered_df[(filtered_df[col] >= value['min']) & 
                                             (filtered_df[col] <= value['max'])]
                elif 'min' in value:
                    filtered_df = filtered_df[filtered_df[col] >= value['min']]
                elif 'max' in value:
                    filtered_df = filtered_df[filtered_df[col] <= value['max']]
                elif 'contains' in value:
                    filtered_df = filtered_df[filtered_df[col].astype(str).str.contains(
                        value['contains'], case=False, na=False)]
            else:
                filtered_df = filtered_df[filtered_df[col] == value]
        
        return filtered_df
    
    def execute_query(self, query: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Execute a pandas query on the dataframe
        
        Args:
            query: A pandas query string
            
        Returns:
            A tuple of (result_dataframe, query_info)
        """
        if self.df is None:
            return pd.DataFrame(), {"error": "No data loaded"}
        
        try:
            result = self.df.query(query)
            info = {
                "query": query,
                "rows_before": len(self.df),
                "rows_after": len(result),
                "columns": list(result.columns)
            }
            return result, info
        except Exception as e:
            return pd.DataFrame(), {"error": str(e), "query": query}
    
    def get_context_for_llm(self, max_rows: int = 100) -> Dict[str, Any]:
        """
        Get context information about the dataframe for the LLM
        
        Args:
            max_rows: Maximum number of rows to include in the sample
            
        Returns:
            A dictionary with context information
        """
        if self.df is None:
            return {"error": "No data loaded"}
        
        context = {
            "file_info": {
                "path": self.file_path,
                "type": self.file_type,
            },
            "dataframe_info": {
                "shape": self.df.shape,
                "columns": list(self.df.columns),
                "dtypes": {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            },
            "sample_data": self.df.head(min(max_rows, len(self.df))).to_dict(orient='records'),
            "summary_stats": {}
        }
        
        # Add summary statistics for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            context["summary_stats"]["numeric"] = self.df[numeric_cols].describe().to_dict()
        
        # Add value counts for categorical columns (limited to top 5 values)
        cat_cols = self.df.select_dtypes(include=['object', 'category']).columns
        if len(cat_cols) > 0:
            context["summary_stats"]["categorical"] = {}
            for col in cat_cols:
                context["summary_stats"]["categorical"][col] = self.df[col].value_counts().head(5).to_dict()
        
        return context


class DFRAG:
    """
    A tool for interacting with CSV and Excel data using natural language through an LLM
    """
    
    def __init__(
        self, 
        file_path: Optional[str] = None,
        llm_backend: str = "ollama",
        llm_model: Optional[str] = None,
        max_context_rows: int = 100
    ):
        """
        Initialize the DFRAG tool
        
        Args:
            file_path: Optional path to a data file to load
            llm_backend: The LLM backend to use (currently only 'ollama' is supported)
            llm_model: The model to use with the LLM backend
            max_context_rows: Maximum number of rows to include in the context
        """
        self.data_handler = DataHandler(file_path)
        self.llm = LLMFactory.create(llm_backend, model=llm_model)
        self.max_context_rows = max_context_rows
        self.conversation_history = []
    
    def load_data(self, file_path: str) -> bool:
        """
        Load data from a CSV or Excel file
        
        Args:
            file_path: Path to the data file
            
        Returns:
            True if loading was successful, False otherwise
        """
        return self.data_handler.load_data(file_path)
    
    def _build_system_prompt(self) -> str:
        """
        Build the system prompt for the LLM
        
        Returns:
            The system prompt string
        """
        if self.data_handler.df is None:
            return "You are an AI assistant that helps users analyze data. No data is currently loaded."
        
        df_info = self.data_handler.get_dataframe_info()
        
        system_prompt = f"""You are an AI assistant that helps users analyze data from {self.data_handler.file_path}.
        
The data has the following structure:
{df_info}

Your task is to:
1. Understand the user's question about the data
2. Provide accurate and helpful answers based on the data provided
3. When appropriate, suggest additional insights or visualizations that might be helpful
4. If you're unsure about something or need more information, ask clarifying questions

Important guidelines:
- Base your answers ONLY on the data provided, not on external knowledge
- If the user asks for something that cannot be determined from the data, explain why
- Be precise and accurate in your descriptions of the data
- When referring to columns, use the exact column names from the data
- If the user's question is ambiguous, ask for clarification
- Format numerical results appropriately (e.g., use appropriate decimal places, units)
- When describing trends or patterns, provide specific examples from the data
"""
        
        return system_prompt
    
    def _build_user_prompt(self, query: str) -> str:
        """
        Build the user prompt for the LLM, including data context
        
        Args:
            query: The user's query
            
        Returns:
            The complete user prompt string
        """
        context = self.data_handler.get_context_for_llm(self.max_context_rows)
        
        # Convert context to a formatted string
        context_str = json.dumps(context, indent=2)
        
        user_prompt = f"""
User Query: {query}

Data Context:
{context_str}

Please analyze the data and answer the query based on the provided context.
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
        if self.data_handler.df is None:
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
    
    def get_dataframe(self) -> Any:
        """
        Get the current dataframe
        
        Returns:
            The pandas DataFrame object or None if no data is loaded
        """
        return self.data_handler.df
    
    def get_dataframe_info(self) -> str:
        """
        Get information about the current dataframe
        
        Returns:
            A string with information about the dataframe
        """
        return self.data_handler.get_dataframe_info()
    
    def filter_data(self, filter_query: Dict[str, Any]) -> None:
        """
        Filter the dataframe based on a query dictionary
        
        Args:
            filter_query: A dictionary with column names as keys and filter values
        """
        if self.data_handler.df is None:
            return
        
        self.data_handler.df = self.data_handler.filter_dataframe(filter_query)
        self.data_handler._generate_metadata()
    
    def reset_filters(self) -> None:
        """Reset any filters applied to the dataframe"""
        if self.data_handler.file_path:
            self.data_handler.load_data(self.data_handler.file_path)
    
    def clear_conversation(self) -> None:
        """Clear the conversation history"""
        self.conversation_history = []


# Simple usage example
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        rag = DFRAG(file_path)
        
        print(f"Loaded data from {file_path}")
        print(rag.get_dataframe_info())
        
        while True:
            query = input("\nAsk a question (or 'exit' to quit): ")
            if query.lower() == 'exit':
                break
            
            print("\nThinking...")
            response = rag.query(query)
            print(f"\nResponse: {response}")
    else:
        print("Usage: python df_rag.py <path_to_csv_or_excel_file>")
        print("Example: python df_rag.py data.csv") 