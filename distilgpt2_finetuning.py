import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# Diagnostic setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"PyTorch version: {torch.__version__}")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")

# Model setup
model_name = "distilgpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Add padding token if not present
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

# Streamlined dataset preparation
def load_and_prepare_dataset(file_path, tokenizer):
    dataset = load_dataset('text', data_files=file_path)
    return dataset['train'].map(
        lambda x: tokenizer(
            x['text'], 
            truncation=True, 
            padding='max_length', 
            max_length=128,
        ),
        batched=True, 
        remove_columns=['text']
    )

# Preset paths
data_file = "c:/Users/HP/Documents/dev/slm/ch01.txt"
logging_dir = "c:/Users/HP/Documents/dev/slm/logs"
output_dir = "c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned"
tokenized_dataset = load_and_prepare_dataset(data_file, tokenizer)

# Optimized training arguments for small notebook
training_args = TrainingArguments(
    seed=42,                            # Seed for reproducibility
    num_proc=4,                         # Number of processes
    output_dir=output_dir,              # Output directory
    overwrite_output_dir=True,          # Overwrite the output directory
    num_train_epochs=10,                # Increased from 5 to 10 for better convergence
    per_device_train_batch_size=2,      # Reduced from 4 to 2
    save_strategy="epoch",              # Save only at epoch end
    save_total_limit=3,                 # Keep the latest 3 checkpoints
    prediction_loss_only=True,          # Only log loss
    fp16=torch.cuda.is_available(),     # Use mixed precision only if CUDA available
    logging_steps=100,                  # More frequent logging for better progress tracking
    logging_dir=logging_dir,            # Directory for storing logs
    gradient_accumulation_steps=4,      # Added to handle smaller batch sizes
    warmup_steps=100,                   # Added warmup steps
    learning_rate=5e-5,                 # Slightly reduced learning rate
    weight_decay=0.01,                  # Added weight decay
    max_grad_norm=1.0,                  # Added gradient clipping
    max_steps=-1,                       # Run until convergence
    lr_scheduler_type="cosine",         # Added learning rate scheduler
    optim="adamw_torch",                # Added optimizer
    dataloader_num_workers=4,           # Added number of workers for dataloader    
)

# Initialize trainer and train
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    train_dataset=tokenized_dataset,
)

# Train the model
trainer.train()

# Save the final model
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

# Test generation with optimized parameters
def generate_text(prompt, model, tokenizer, device):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output = model.generate(
        input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        repetition_penalty=1.2,  # Add repetition penalty to avoid loops
        no_repeat_ngram_size=3,  # Prevent repetition of 3-grams
        pad_token_id=tokenizer.pad_token_id,
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

test_prompt = "What is the role of textual criticism"
print(f"Generated text: {generate_text(test_prompt, model, tokenizer, device)}")
