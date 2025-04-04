import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import argparse
import os
import sys

def load_model(model_path):
    """Load the fine-tuned model and tokenizer"""
    print(f"Loading model from {model_path}...")
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        sys.exit(1)
        
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load tokenizer and model
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path).to(device)
        
        # Set pad token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            model.config.pad_token_id = model.config.eos_token_id
            
        print("Model loaded successfully!")
        return model, tokenizer, device
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def generate_text(prompt, model, tokenizer, device, 
                  max_length=100, 
                  temperature=0.7,
                  top_p=0.9,
                  top_k=50,
                  repetition_penalty=1.2,
                  no_repeat_ngram_size=3):
    """Generate text based on the given prompt"""
    try:
        # Encode the prompt
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        # Generate text
        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        # Decode the generated text
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return generated_text
    except Exception as e:
        return f"Error generating text: {e}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate text using fine-tuned DistilGPT-2 model")
    parser.add_argument("--model_path", type=str, default="c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned",
                        help="Path to the fine-tuned model")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    
    args = parser.parse_args()
    
    # Load model
    model, tokenizer, device = load_model(args.model_path)
    
    # Interactive CLI
    print("\n" + "="*50)
    print("DistilGPT-2 Text Generation CLI")
    print("Type 'exit' or 'quit' to end the session")
    print("="*50 + "\n")
    
    while True:
        # Get user input
        prompt = input("\nEnter your prompt: ")
        
        # Check if user wants to exit
        if prompt.lower() in ["exit", "quit"]:
            print("Exiting...")
            break
        
        # Generate and display text
        print("\nGenerating text...\n")
        generated_text = generate_text(
            prompt, model, tokenizer, device,
            max_length=args.max_length + len(tokenizer.encode(prompt)),
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        print("-"*50)
        print(generated_text)
        print("-"*50)

if __name__ == "__main__":
    main()