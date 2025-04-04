import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Ensure CUDA errors are reported immediately
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Load the fine-tuned model
model_path = "c:/Users/HP/Documents/dev/slm/qwen-finetuned"
print(f"Loading model from {model_path}")

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

try:
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print("Tokenizer loaded successfully")
    
    # Load model in full precision for stability
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,  # Use full precision
        low_cpu_mem_usage=True,
        device_map="cpu"  # First load on CPU
    )
    
    print("Model loaded successfully")
    
    # Test prompts
    prompts = [
        "What is the role of textual criticism",
        "Textual criticism is important because",
        "The process of determining the original text involves",
        "Textual criticism helps scholars to"
    ]
    
    # Function to safely generate text
    def safe_generate(prompt, use_sampling=False):
        # Always start with model on CPU for safety
        model.to("cpu")
        
        # Tokenize input (on CPU)
        inputs = tokenizer(prompt, return_tensors="pt")
        
        # Generate on CPU first
        model.eval()
        with torch.no_grad():
            try:
                gen_params = {
                    "max_new_tokens": 30,
                    "num_return_sequences": 1,
                    "pad_token_id": tokenizer.eos_token_id,
                    "eos_token_id": tokenizer.eos_token_id
                }
                
                if use_sampling:
                    gen_params.update({
                        "do_sample": True,
                        "temperature": 0.3,  # Very low temperature
                        "top_k": 5,          # Very restrictive
                        "top_p": 0.85,
                        "repetition_penalty": 1.5  # Stronger repetition penalty
                    })
                else:
                    gen_params.update({"do_sample": False})
                
                output = model.generate(**inputs, **gen_params)
                return tokenizer.decode(output[0], skip_special_tokens=True)
            except Exception as e:
                return f"Error: {str(e)}"
    
    # Test each prompt with both greedy and sampling approaches
    for i, prompt in enumerate(prompts):
        print(f"\n\n[{i+1}] Testing prompt: {prompt}")
        
        # Try greedy decoding first
        print("Attempting greedy decoding...")
        greedy_result = safe_generate(prompt, use_sampling=False)
        print(f"Generated text (greedy):\n{greedy_result}")
        
        # Then try sampling with conservative parameters
        print("\nAttempting sampling with conservative parameters...")
        sampling_result = safe_generate(prompt, use_sampling=True)
        print(f"Generated text (sampling):\n{sampling_result}")
            
except Exception as e:
    print(f"Error loading model: {e}")
    
print("\nTesting complete")