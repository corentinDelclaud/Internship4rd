import os
import json
from typing import List, Dict

def load_documents(directory: str) -> List[Dict]:
    """
    Load JSON documents from a directory.
    
    Returns a list of processed documents with flat structure for embedding.
    """
    # Fix path resolution
    if not os.path.isabs(directory):
        # If running from project root
        if os.path.exists(os.path.join('2', directory)):
            directory = os.path.join('2', directory)
        # If running from within the '2' directory
        elif os.path.exists(directory):
            pass
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            directory = os.path.join(base_dir, directory)
    
    print(f"Looking for documents in: {directory}")
    
    if not os.path.exists(directory):
        print(f"Warning: Directory {directory} does not exist!")
        return []
    
    documents = []
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    json_data = json.load(file)
                    
                    # Handle case where JSON file contains an array of objects
                    if isinstance(json_data, list):
                        for i, json_obj in enumerate(json_data):
                            # Process each object in the array
                            process_json_object(json_obj, filename, i, documents)
                    else:
                        # Handle single JSON object
                        process_json_object(json_data, filename, 0, documents)
                        
            except json.JSONDecodeError as e:
                print(f"Error: {filename} is not a valid JSON file: {e}")
                
    print(f"Processed {len(documents)} documents from {directory}")
    return documents

def process_json_object(json_obj, filename, index, documents):
    """Process a single JSON object and add it to documents list."""
    # Extract content from sections for embedding
    full_content = ""
    for section in json_obj.get('sections', []):
        if 'content' in section:
            full_content += section['content'] + " "
    
    # Create a document that preserves the JSON structure but adds
    # a flat content field for embedding purposes
    document = {
        'id': f"{filename}_{index}" if index > 0 else filename,
        'title': json_obj.get('title', f'Document {index+1}'),
        'author': json_obj.get('author', 'Unknown'),
        'date': json_obj.get('date', ''),
        'tags': json_obj.get('tags', []),
        'sections': json_obj.get('sections', []),
        'content': full_content.strip()  # For embedding
    }
    
    documents.append(document)