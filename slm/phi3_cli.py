import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
import sys
from rich.console import Console
from rich.markdown import Markdown

console = Console()

def load_model(model_name="microsoft/phi-3-mini-128k-instruct", cache_dir=None):
    """Load the Phi-3.1 model and tokenizer"""
    console.print(f"[bold blue]Loading model {model_name}...[/bold blue]")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    console.print(f"Using device: [bold green]{device}[/bold green]")
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        
        # Load model with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            cache_dir=cache_dir,
            trust_remote_code=True
        )
            
        console.print("[bold green]Model loaded successfully![/bold green]")
        return model, tokenizer, device
    except Exception as e:
        console.print(f"[bold red]Error loading model: {e}[/bold red]")
        sys.exit(1)

def generate_response(prompt, model, tokenizer, device, 
                     max_new_tokens=512, 
                     temperature=0.7,
                     top_p=0.9,
                     top_k=50,
                     repetition_penalty=1.1):
    """Generate a response based on the given prompt"""
    try:
        # Format the prompt for Phi-3.1
        formatted_prompt = f"<|user|>\n{prompt}\n<|assistant|>\n"
        
        # Encode the prompt
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
        
        # Generate text with modified parameters to avoid cache issues
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
                # Add these parameters to fix the cache issue
                min_length=1,
                no_repeat_ngram_size=2
            )
        
        # Decode the generated text
        full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        assistant_response = full_response.split("<|assistant|>")[-1].strip()
        return assistant_response
    except Exception as e:
        return f"Error generating response: {e}"

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Chat with Phi-3 Mini locally")
    parser.add_argument("--model_name", type=str, default="microsoft/phi-3-mini-128k-instruct",
                        help="Model name or path")
    parser.add_argument("--cache_dir", type=str, default="c:/Users/HP/Documents/dev/slm/model_cache",
                        help="Directory to cache the downloaded model")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Maximum number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Top-p sampling parameter")
    parser.add_argument("--top_k", type=int, default=50,
                        help="Top-k sampling parameter")
    
    args = parser.parse_args()
    
    # Create cache directory if it doesn't exist
    if args.cache_dir:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Load model
    model, tokenizer, device = load_model(args.model_name, args.cache_dir)
    
    # Interactive CLI
    console.print("\n" + "="*50)
    console.print("[bold cyan]Phi-3.1 Mini Chat CLI[/bold cyan]")
    console.print("[italic]Type 'exit' or 'quit' to end the session[/italic]")
    console.print("="*50 + "\n")
    
    while True:
        # Get user input
        prompt = input("\n[You]: ")
        
        # Check if user wants to exit
        if prompt.lower() in ["exit", "quit"]:
            console.print("[bold]Exiting...[/bold]")
            break
        
        # Generate and display response
        console.print("\n[bold]Generating response...[/bold]")
        response = generate_response(
            prompt, model, tokenizer, device,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        
        console.print("\n[Phi-3.1]:")
        console.print(Markdown(response))
        console.print("-"*50)

if __name__ == "__main__":
    main()