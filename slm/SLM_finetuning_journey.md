# Fine-tuning DistilGPT2 on a Quadro P4000: Our Journey

In this post, I'll walk through our experience fine-tuning a small language model (DistilGPT2) on a workstation with an NVIDIA Quadro P4000. This journey illustrates how you can leverage modest hardware for AI development and the common challenges you might encounter along the way.

## Our Hardware Setup

We started with a decent workstation:

- HP Z640 Workstation
- Intel Xeon CPU E5-2640 v4 @ 2.40GHz
- 32GB DDR4 RAM
- 256GB SSD + 1TB HDD
- NVIDIA Quadro P4000 8GB
- Windows 10 Pro

While not a powerhouse by modern AI standards, this setup is perfectly capable of fine-tuning smaller models like DistilGPT2.

## Step 1: Verifying GPU Availability

Our first task was to confirm that our GPU was properly recognized by the system. We ran the `nvidia-smi` command, which showed:

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 572.83                 Driver Version: 572.83         CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Quadro P4000                 WDDM  |   00000000:02:00.0  On |                  N/A |
| 48%   43C    P8              8W /  105W |     587MiB /   8192MiB |      8%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```


This confirmed that our GPU was working correctly with driver version 572.83 and CUDA 12.8.

## Step 2: Setting Up Our Environment

We created a Python script to fine-tune DistilGPT2 on our custom dataset. Initially, we encountered an issue where PyTorch wasn't detecting our GPU. We added diagnostic code to investigate:

```python
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
```


The output showed that CUDA wasn't available to PyTorch, despite being installed on our system.

## Step 3: Fixing CUDA Support in PyTorch

We needed to reinstall PyTorch with CUDA support. The command we used was:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```


After reinstalling, we ran our diagnostics again and confirmed that PyTorch could now see our GPU.

## Step 4: Model Selection and Loading

Initially, we tried to load a GGUF model file directly:

```python
model_name = r"C:\\Users\\HP\\.lmstudio\\models\\RichardErkhov\\distilbert_-_distilgpt2-gguf\\distilgpt2.Q4_K_S.gguf"
```

This caused an error because Hugging Face's transformers library doesn't support loading GGUF files directly. We corrected this by using the model ID instead:

```python
model_name = "distilgpt2"
```


## Step 5: Data Preparation

We created a data preparation script to process our text files:

```python
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
```
### Training Data
For our fine-tuning experiment, we used a specialized dataset focused on textual criticism. The dataset ( textual_criticism.txt ) contained approximately 800KB of text material covering:

- Principles of textual criticism in biblical studies
- Historical methods for manuscript analysis
- Techniques for identifying and resolving textual variants
- Case studies of significant manuscript discoveries
- Scholarly articles on the transmission of ancient texts

This domain-specific dataset was chosen to test how well a small language model could adapt to specialized academic content. The relatively small size of the dataset (compared to the gigabytes used for training larger models) made it suitable for our hardware constraints while still providing enough material for meaningful fine-tuning.

## Step 6: Fine-tuning Configuration

We configured our training parameters with consideration for our hardware limitations:

```python
training_args = TrainingArguments(
    output_dir="c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced from 4 to 2 for GPU memory constraints
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # Enable mixed precision training
)
```

We reduced the batch size to 2 to accommodate our 8GB VRAM and enabled mixed precision training (fp16) to further optimize memory usage.

## Step 7: Training the Model

With everything set up, we started the training process:

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

trainer.train()
```


The training process utilized our GPU effectively, and we could monitor its usage with `nvidia-smi` during training.

## Step 8: Testing the Fine-tuned Model

After training, we tested our model with a simple prompt:

```python
prompt = "What is the role of textual criticism"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
model = model.to(device)

output = model.generate(
    input_ids,
    max_length=100,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(f"Generated text: {generated_text}")
```


## Lessons Learned

1. **Hardware Compatibility**: Even modest GPUs like the Quadro P4000 can be used for fine-tuning smaller models.

2. **PyTorch CUDA Setup**: Ensuring PyTorch is installed with the correct CUDA support is crucial for GPU utilization.

3. **Model Selection**: Starting with smaller models like DistilGPT2 allows for faster experimentation and learning.

4. **Memory Management**: Techniques like reducing batch size and using mixed precision training help overcome VRAM limitations.

5. **Data Preparation**: Clean, well-structured data is essential for effective fine-tuning.

## Next Steps

Having successfully fine-tuned DistilGPT2, we're now ready to explore slightly larger models like Phi-2 (2.7B) or TinyLlama (1.1B). These models offer better performance while still being manageable on our hardware with the right optimization techniques.

We're also considering:
- Experimenting with different learning rates and training parameters
- Implementing more sophisticated data preprocessing
- Exploring quantization techniques to run larger models
- Evaluating our model more systematically

## Conclusion

Our journey demonstrates that AI development doesn't always require high-end hardware. With the right approach, even a modest workstation can be used to fine-tune smaller language models effectively. This makes AI experimentation more accessible to developers, researchers, and enthusiasts with limited resources.


