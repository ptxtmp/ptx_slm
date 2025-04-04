import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

# Path to your model
model_path = "C:\\Users\\HP\\.lmstudio\\models\\lmstudio-community\\Phi-3.1-mini-128k-instruct-GGUF"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

def generate_response(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    parser = argparse.ArgumentParser(description="Generate answers using the Phi-3 model.")
    parser.add_argument("prompt", type=str, help="Prompt for the model")
    args = parser.parse_args()
    
    response = generate_response(args.prompt)
    print("Response:", response)

if __name__ == "__main__":
    # while True:
    #     user_input = input("Enter your prompt: ")
    #     if user_input.lower() == "exit":
    #         break
    #     print("Response:", generate_response(user_input))
    main()