# Small Language Model Fine-tuning

This repository contains tools and scripts for fine-tuning small language models, specifically DistilGPT2, on custom datasets.

## Project Overview

This project provides a complete pipeline for:
1. Processing and preparing text data from various sources (Markdown, XML, etc.)
2. Creating datasets suitable for language model fine-tuning
3. Fine-tuning DistilGPT2 on custom data
4. Testing and using the fine-tuned model

## Repository Structure

- `prepare_data.py` - Script to extract and process text from various file formats
- `distilgpt2_finetuning.py` - Script to fine-tune DistilGPT2 on processed data
- `data/` - Directory for source data files
- `processed_data.txt` - Processed text output
- `distilgpt2-finetuned/` - Directory for the fine-tuned model

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- Datasets
- BeautifulSoup4
- Markdown

## Setup

1. Install the required packages:
```bash
pip install torch transformers datasets markdown beautifulsoup4
```
python c:\Users\HP\Documents\dev\slm\prepare_data.py

This will process files from the `data/` directory and create a processed text file.

### 2. Fine-tune the Model

Run the fine-tuning script:
```bash
python c:\Users\HP\Documents\dev\slm\distilgpt2_finetuning.py
```

This will:
- Load the DistilGPT2 model
- Prepare your dataset
- Fine-tune the model
- Save the fine-tuned model to `distilgpt2-finetuned/`
- Test the model with a sample prompt

### 3. Use Your Fine-tuned Model

After fine-tuning, you can use your model for text generation:

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load fine-tuned model
model_path = "c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

# Generate text
prompt = "Your prompt here"
input_ids = tokenizer.encode(prompt, return_tensors="pt")
output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(generated_text)
```
## Hardware Requirements
This project has been tested on:

- HP Z640 Workstation
- Intel Xeon CPU E5-2640 v4 @ 2.40GHz
- 32GB DDR4 RAM
- NVIDIA Quadro P4000 8GB
- Windows 10 Pro
## Extending the Project
### Adding Support for New File Types
To add support for additional file types, modify the prepare_data.py script to include new extraction functions.

### Fine-Tuning Hyperparameters
Experiment with different hyperparameters to find the best model for the dataset. Modify the distilgpt2_finetuning.py script to include these parameters.

### Optimizing the Fine-Tuning Process
Consider using distributed training or mixed-precision training for faster fine-tuning.

### Using Different Models
While this project focuses on DistilGPT2, the same approach can be used for other models like:

- Phi-2 (2.7B)
- TinyLlama (1.1B)
- Gemma 2B
## License
[Your license information here]

## Acknowledgments
- Hugging Face for the Transformers library
- OpenAI for the GPT-2 architecture