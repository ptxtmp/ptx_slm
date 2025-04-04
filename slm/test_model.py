import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import time
import argparse

def test_model(model_path, prompt, max_length=100, temperature=0.7, top_p=0.9, 
               num_return_sequences=1, do_sample=True):
    """
    Test a fine-tuned GPT-2 model with a given prompt.
    
    Args:
        model_path (str): Path to the fine-tuned model
        prompt (str): Text prompt to generate from
        max_length (int): Maximum length of generated text
        temperature (float): Controls randomness (higher = more random)
        top_p (float): Nucleus sampling parameter
        num_return_sequences (int): Number of sequences to generate
        do_sample (bool): Whether to use sampling or greedy decoding
        
    Returns:
        list: Generated text sequences
    """
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"Loading model from {model_path}...")
    start_time = time.time()
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    model = model.to(device)
    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.2f} seconds")
    
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text
    print(f"Generating text from prompt: '{prompt}'")
    start_time = time.time()
    output = model.generate(
        input_ids,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        pad_token_id=tokenizer.eos_token_id
    )
    generation_time = time.time() - start_time
    
    # Decode the generated text
    generated_texts = []
    for i, sequence in enumerate(output):
        generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
        generated_texts.append(generated_text)
        print(f"\nGenerated text {i+1}:")
        print(f"{generated_text}")
        
    print(f"\nGeneration completed in {generation_time:.2f} seconds")
    print(f"Generation speed: {max_length / generation_time:.2f} tokens/second")
    
    return generated_texts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test a fine-tuned GPT-2 model")
    parser.add_argument("--model_path", type=str, default="c:/Users/HP/Documents/dev/slm/distilgpt2-finetuned",
                        help="Path to the fine-tuned model")
    parser.add_argument("--prompt", type=str, default="What is the role of textual criticism",
                        help="Text prompt to generate from")
    parser.add_argument("--max_length", type=int, default=100,
                        help="Maximum length of generated text")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Controls randomness (higher = more random)")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="Nucleus sampling parameter")
    parser.add_argument("--num_sequences", type=int, default=1,
                        help="Number of sequences to generate")
    
    args = parser.parse_args()
    
    test_model(
        args.model_path,
        args.prompt,
        args.max_length,
        args.temperature,
        args.top_p,
        args.num_sequences
    )