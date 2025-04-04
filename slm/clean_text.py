import re

def clean_text_file(input_file, output_file):
    """
    Clean up a text file by:
    1. Removing hyphens at the end of lines when they split words
    2. Joining lines that are part of the same paragraph
    3. Preserving intentional line breaks (like paragraph breaks)
    """
    print(f"Cleaning text file: {input_file}")
    
    # Read the entire file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Replace Unicode characters that might cause issues
    content = content.replace('\ufeff', '')  # BOM
    content = content.replace('\u201c', '"')  # Opening double quote
    content = content.replace('\u201d', '"')  # Closing double quote
    content = content.replace('\u2018', "'")  # Opening single quote
    content = content.replace('\u2019', "'")  # Closing single quote
    content = content.replace('\u2013', '-')  # En dash
    content = content.replace('\u2014', '--')  # Em dash
    
    # Split into lines
    lines = content.split('\n')
    cleaned_lines = []
    i = 0
    
    while i < len(lines):
        current_line = lines[i].rstrip()
        
        # Skip page numbers and headers
        if re.match(r'^\d+\s*$', current_line) or re.match(r'^2005 Students Guide\.book', current_line):
            i += 1
            continue
            
        # Check if this line ends with a hyphen and the next line starts with a letter
        if (i < len(lines) - 1 and 
            current_line.endswith('-') and 
            len(lines[i+1].strip()) > 0 and 
            not re.match(r'^[0-9\[\(\.]', lines[i+1].strip())):
            
            # Remove hyphen and join with next line without space
            current_line = current_line[:-1] + lines[i+1].strip()
            i += 2  # Skip the next line since we've incorporated it
        
        # If the current line is not empty and doesn't end with a period, question mark, etc.
        # and the next line is not empty and doesn't start with a special character
        elif (i < len(lines) - 1 and 
              len(current_line) > 0 and 
              not re.search(r'[.?!:;"\']$', current_line) and
              not current_line.endswith(' ') and
              len(lines[i+1].strip()) > 0 and
              not re.match(r'^[0-9\[\(\.]', lines[i+1].strip()) and
              not re.match(r'^(PART|Figure|Table|\d+\.\d+)', lines[i+1].strip())):
            
            # Join with the next line with a space
            current_line = current_line + ' ' + lines[i+1].strip()
            i += 2  # Skip the next line
        else:
            # Keep the line as is
            i += 1
        
        # Add the processed line
        cleaned_lines.append(current_line)
    
    # Join the cleaned lines
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Remove multiple consecutive blank lines
    cleaned_content = re.sub(r'\n{3,}', '\n\n', cleaned_content)
    
    # Write the cleaned content to the output file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)
    
    print(f"Cleaned text saved to: {output_file}")

if __name__ == "__main__":
    input_file = "c:/Users/HP/Documents/dev/slm/tc_short.txt"
    output_file = "c:/Users/HP/Documents/dev/slm/tc_short_cleaned.txt"
    clean_text_file(input_file, output_file)