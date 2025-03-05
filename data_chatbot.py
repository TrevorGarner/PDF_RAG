import os
import pandas as pd
import ollama
from fuzzywuzzy import fuzz

datasets = {}

def load_files(directory):
    """Loads all CSV and Excel files in the given directory into memory."""
    global datasets
    for file in os.listdir(directory):
        if file.endswith(".csv"):
            datasets[file] = pd.read_csv(os.path.join(directory, file))
        elif file.endswith(".xlsx") or file.endswith(".xls"):
            datasets[file] = pd.read_excel(os.path.join(directory, file))

def query(user_query):
    """Handles user queries and returns relevant data using Ollama."""
    user_query = user_query.lower()
    
    if not datasets:
        return "No files loaded."
    
    matched_data = []
    references = []
    
    for file, df in datasets.items():
        # First check column names as before
        for column in df.columns:
            if any(keyword in column.lower() for keyword in user_query.split()):
                relevant_rows = df[[column]].dropna()
                matched_data.append(relevant_rows)
                references.append(f"{file} -> {column}")
        
        # Now check actual data with fuzzy matching
        for column in df.select_dtypes(include=['object']).columns:
            # Apply fuzzy matching to each cell
            mask = df[column].astype(str).apply(lambda x: fuzz.partial_ratio(x.lower(), user_query.lower()) > 80)
            matching_rows = df[mask]
            if not matching_rows.empty:
                matched_data.append(matching_rows)
                references.append(f"{file} -> full row match from {column}")
    
    if not matched_data:
        return "No relevant data found."
    
    # Format the data in a more readable way
    formatted_data = []
    used_references = []  # Keep track of which references are actually used
    
    for i, data_frame in enumerate(matched_data):
        source = references[i]
        # Convert to a list of dictionaries for better readability
        rows = data_frame.to_dict(orient='records')
        if rows:  # Only include non-empty data
            formatted_data.append({
                "source": source,
                "data": rows
            })
            used_references.append(source)  # Add this reference to used references
    
    # Create a better prompt
    system_prompt = """You are a helpful assistant that provides clear, concise answers based on the user's data.
    Directly answer the user's question using only the data provided.
    Focus on the specific information they're asking about and present it in a friendly, organized way.
    If the data contains coordinates, convert them to a more readable format if possible.
    If multiple data sources are present, prioritize the most relevant ones to the query."""
    
    user_content = f"""
    User query: {user_query}
    
    Relevant data:
    {formatted_data}
    
    Please provide a direct answer to the user's question based only on this data.
    """
    
    response = ollama.chat(
        model="llama3.1", 
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
    )
    
    return {"response": response['message']['content'], "references": used_references}

if __name__ == "__main__":
    load_files("data")  # Ensure you have a "data" directory with CSV/Excel files
    while True:
        user_query = input("Enter your query (or type 'exit' to quit): ")
        if user_query.lower() == "exit":
            break
        result = query(user_query)
        print("\nResponse:", result["response"], "\nReferences:", result["references"], "\n")
