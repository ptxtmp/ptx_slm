import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Add diagnostic information
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda if torch.cuda.is_available() else 'Not available'}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
# After loading the model
model_name = "distilgpt2" # r"C:\\Users\\HP\\.lmstudio\\models\\RichardErkhov\\distilbert_-_distilgpt2-gguf\\distilgpt2.Q4_K_S.gguf" 
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Explicitly move model to GPU if available
if torch.cuda.is_available():
    model = model.cuda()
    print("Model successfully moved to GPU")
else:
    print("GPU not available, using CPU")

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Prepare your dataset
# For this example, we'll create a simple dataset from text files
def load_and_prepare_dataset(file_path, tokenizer):
    dataset = load_dataset('text', data_files=file_path)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=['text'])
    return tokenized_dataset['train']

# Path to your processed data file
data_file = "c:/Users/HP/Documents/dev/slm/textual_criticism.txt"

# Load and tokenize dataset
tokenized_dataset = load_and_prepare_dataset(data_file, tokenizer)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,  # Reduced from 4 to 2
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
    fp16=True,  # Enable mixed precision training
)

# Data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, 
    mlm=False  # We're not using masked language modeling
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned")
tokenizer.save_pretrained("c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned")

# Test the model
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