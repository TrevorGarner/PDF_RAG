import pandas as pd
import os
from typing import List, Dict, Union, Optional
from rapidfuzz import fuzz, process
import argparse
import logging
from pathlib import Path
import warnings
import json
from dataclasses import dataclass
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Represents a loaded data source with its metadata"""
    filename: str
    data: pd.DataFrame
    description: str = ""

class DataManager:
    """Manages loading and accessing data from CSV and Excel files"""
    
    def __init__(self):
        self.data_sources: Dict[str, DataSource] = {}
        
    def load_file(self, file_path: str) -> bool:
        """Load a single file into memory"""
        try:
            file_path = Path(file_path)
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                return False
                
            if file_path.suffix.lower() == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)
            else:
                logger.error(f"Unsupported file type: {file_path.suffix}")
                return False
                
            self.data_sources[file_path.name] = DataSource(
                filename=file_path.name,
                data=df,
                description=f"Data from {file_path.name} with {len(df)} rows and {len(df.columns)} columns"
            )
            logger.info(f"Successfully loaded {file_path.name}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {str(e)}")
            return False
            
    def load_directory(self, directory_path: str) -> int:
        """Load all supported files from a directory"""
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            logger.error(f"Invalid directory: {directory_path}")
            return 0
            
        loaded_files = 0
        for file_path in directory.glob("*"):
            if file_path.suffix.lower() in ['.csv', '.xlsx', '.xls']:
                if self.load_file(str(file_path)):
                    loaded_files += 1
                    
        return loaded_files

class QueryProcessor:
    """Processes natural language queries and retrieves relevant data"""
    
    def __init__(self, data_manager: DataManager):
        self.data_manager = data_manager
        self.conversation_history: List[Dict[str, str]] = []  # Store full conversation
        self.SIMILARITY_THRESHOLD = 80  # Minimum fuzzy match score (0-100)
        self.MAX_HISTORY = 5  # Maximum number of previous exchanges to include
        
    def _fuzzy_search_columns(self, query: str, df: pd.DataFrame) -> List[str]:
        """Find columns that match the query using fuzzy matching"""
        try:
            matches = process.extract(
                query,
                df.columns.tolist(),  # Convert to list explicitly
                scorer=fuzz.partial_ratio,
                limit=None
            )

            # matches will be a list of tuples: (choice, score, index)
            return [col for col, score, *_ in matches if score >= self.SIMILARITY_THRESHOLD]
        except Exception as e:
            logger.error(f"Error in fuzzy column search: {str(e)}")
            return []
        
    def _fuzzy_search_values(self, query: str, df: pd.DataFrame) -> pd.DataFrame:
        """Search for matching values in the dataframe"""
        mask = pd.DataFrame(False, index=df.index, columns=df.columns)
        
        for col in df.columns:
            if df[col].dtype == object:  # Only search string columns
                mask[col] = df[col].astype(str).apply(
                    lambda x: fuzz.WRatio(query.lower(), str(x).lower()) >= self.SIMILARITY_THRESHOLD
                )
                
        return df[mask.any(axis=1)]
        
    def _prepare_context(self, relevant_data: pd.DataFrame, max_rows: int = 50) -> str:
        """Prepare data context for the LLM"""
        if len(relevant_data) > max_rows:
            return f"Data sample ({max_rows} of {len(relevant_data)} rows):\n{relevant_data.head(max_rows).to_string()}"
        return f"Data ({len(relevant_data)} rows):\n{relevant_data.to_string()}"
        
    def _get_conversation_context(self) -> str:
        """Create a formatted string of recent conversation history"""
        if not self.conversation_history:
            return ""
            
        context = "\nPrevious conversation:\n"
        # Get last MAX_HISTORY exchanges
        recent_history = self.conversation_history[-self.MAX_HISTORY:]
        for exchange in recent_history:
            context += f"User: {exchange['query']}\n"
            if 'response' in exchange:
                context += f"Assistant: {exchange['response']}\n"
        return context
        
    def _call_llama(self, prompt: str) -> str:
        """Make an API call to Ollama running Llama3.1"""
        try:
            # Add conversation history to the prompt
            context = self._get_conversation_context()
            
            # Add citation instructions
            citation_instructions = """
When responding, please cite your sources using [filename:column/row] format.
For example: 
- When referring to data from a specific column: [sales.csv:column=revenue]
- When referring to specific rows: [employees.csv:rows=10-15]
- When using column statistics: [products.csv:stats=category]
Include these citations inline with your response.
"""
            full_prompt = f"{context}\n{citation_instructions}\nCurrent query: {prompt}"
            
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    'model': 'llama3.1',
                    'prompt': full_prompt,
                    'stream': False
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            logger.error(f"Error calling Llama: {str(e)}")
            return "I apologize, but I encountered an error processing your request."
            
    def _get_column_stats(self, df: pd.DataFrame, column: str) -> str:
        """Get high-level statistics for a column"""
        try:
            stats = []
            # Get basic info
            unique_count = df[column].nunique()
            total_count = len(df[column])
            null_count = df[column].isnull().sum()
            
            stats.append(f"Total values: {total_count}")
            stats.append(f"Unique values: {unique_count}")
            
            if null_count > 0:
                stats.append(f"Null values: {null_count}")
            
            # For numeric columns
            if pd.api.types.is_numeric_dtype(df[column]):
                stats.append(f"Range: {df[column].min()} to {df[column].max()}")
            # For categorical/text columns
            elif pd.api.types.is_string_dtype(df[column]):
                top_values = df[column].value_counts().head(3)
                stats.append("Top 3 values:")
                for val, count in top_values.items():
                    stats.append(f"  - {val}: {count}")
                    
            return f"Column '{column}' stats:\n" + "\n".join(stats)
        except Exception as e:
            logger.error(f"Error getting column stats: {str(e)}")
            return ""

    def process_query(self, query: str) -> str:
        """Process a natural language query and return a response"""
        if not self.data_manager.data_sources:
            return "No data sources have been loaded. Please load some CSV or Excel files first."
            
        # Add query to conversation history
        current_exchange = {"query": query}
        
        # Handle general queries about the loaded data
        if any(keyword in query.lower() for keyword in ['summarize', 'describe', 'what data', 'show me']):
            summaries = []
            for source in self.data_manager.data_sources.values():
                summary = f"\n[{source.filename}:summary]\n"
                summary += f"- {len(source.data)} rows, {len(source.data.columns)} columns\n"
                summary += f"- Columns: {', '.join(source.data.columns)}\n"
                summaries.append(summary)
            response = "Here's a summary of the loaded data:" + ''.join(summaries)
            current_exchange['response'] = response
            self.conversation_history.append(current_exchange)
            return response
            
        # Search for relevant data across all sources
        relevant_results = []
        column_stats = []  # Store column statistics
        
        for source in self.data_manager.data_sources.values():
            # Search in column names
            matching_cols = self._fuzzy_search_columns(query, source.data)
            if matching_cols:
                relevant_subset = source.data[matching_cols]
                relevant_results.append({
                    'source': source.filename,
                    'data': relevant_subset,
                    'columns': matching_cols,
                    'context': f"Matching columns from {source.filename}"
                })
                # Add column statistics for matching columns
                for col in matching_cols:
                    stats = self._get_column_stats(source.data, col)
                    if stats:
                        column_stats.append(f"\n[{source.filename}:stats={col}]\n{stats}")
                
            # Search in values
            matching_rows = self._fuzzy_search_values(query, source.data)
            if not matching_rows.empty:
                row_indices = matching_rows.index.tolist()
                row_range = f"{min(row_indices)}-{max(row_indices)}"
                relevant_results.append({
                    'source': source.filename,
                    'data': matching_rows,
                    'row_range': row_range,
                    'context': f"Matching rows {row_range} from {source.filename}"
                })
                
        # Prepare prompt for Llama
        prompt = f"""Based on the following data and conversation history, please answer this query: "{query}"\n\n"""
        
        # Add column statistics to the prompt if available
        if column_stats:
            prompt += "\nColumn Analysis:\n" + "\n".join(column_stats) + "\n"
        
        for result in relevant_results:
            context = self._prepare_context(result['data'])
            citation = f"[{result['source']}:"
            if 'columns' in result:
                citation += f"columns={','.join(result['columns'])}"
            if 'row_range' in result:
                citation += f"rows={result['row_range']}"
            citation += "]"
            prompt += f"\n{citation}\n{context}\n"
            
        # Call Llama and get response
        response = self._call_llama(prompt)
        current_exchange['response'] = response
        self.conversation_history.append(current_exchange)
        return response

class ChatBot:
    """Main chatbot class that handles user interaction"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.query_processor = QueryProcessor(self.data_manager)
        
    def load_data(self, path: str) -> None:
        """Load data from a file or directory"""
        path = Path(path)
        if path.is_dir():
            loaded = self.data_manager.load_directory(str(path))
            print(f"Loaded {loaded} files from directory")
        else:
            if self.data_manager.load_file(str(path)):
                print(f"Successfully loaded {path.name}")
            else:
                print(f"Failed to load {path.name}")
                
    def run(self):
        """Run the interactive chat session"""
        print("Welcome to the CSV & Excel Query Chatbot!")
        print("Type 'exit' to quit, 'load <path>' to load data, or enter your query.")
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'exit':
                    print("Goodbye!")
                    break
                    
                if user_input.lower().startswith('load '):
                    path = user_input[5:].strip()
                    self.load_data(path)
                    continue
                    
                if not user_input:
                    continue
                    
                response = self.query_processor.process_query(user_input)
                print("\nChatbot:", response)
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
                print("An error occurred. Please try again.")

def main():
    parser = argparse.ArgumentParser(description="CSV & Excel Query Chatbot")
    parser.add_argument('--path', type=str, help='Initial file or directory to load')
    args = parser.parse_args()
    
    chatbot = ChatBot()
    if args.path:
        chatbot.load_data(args.path)
        
    chatbot.run()

if __name__ == "__main__":
    main()
