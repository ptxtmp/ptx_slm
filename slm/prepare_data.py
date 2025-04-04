import os
import re
import json
from pathlib import Path
import markdown
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET

# Define paths to your data sources
data_dir = Path("c:/Users/HP/Documents/dev/slm/data")
output_file = Path("c:/Users/HP/Documents/dev/slm/processed_data.txt")

# Function to extract text from markdown
def extract_from_markdown(md_file):
    with open(md_file, 'r', encoding='utf-8') as f:
        md_text = f.read()
    html = markdown.markdown(md_text)
    soup = BeautifulSoup(html, 'html.parser')
    return soup.get_text()

# Function to extract text from XML
def extract_from_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # Adjust based on your XML structure
    texts = []
    for elem in root.findall('.//content'):
        if elem.text:
            texts.append(elem.text.strip())
    return "\n".join(texts)

# Process all files and write to output
with open(output_file, 'w', encoding='utf-8') as out_file:
    # Process markdown files
    for md_file in data_dir.glob('**/*.md'):
        text = extract_from_markdown(md_file)
        out_file.write(text + "\n\n")
    
    # Process XML files
    for xml_file in data_dir.glob('**/*.xml'):
        text = extract_from_xml(xml_file)
        out_file.write(text + "\n\n")
    
    # Add processing for other file types as needed

print(f"Data processing complete. Output saved to {output_file}")