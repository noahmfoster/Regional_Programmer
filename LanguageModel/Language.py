from transformers import GPT2LMHeadModel

from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import GPTJForCausalLM

import os

model_path = os.path.dirname(os.path.abspath(__file__)) 

def get_language_model(model_path = model_path):
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return fine_tuned_model, tokenizer

def get_GPTJ():
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B")
    tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-j-6B")
    return model, tokenizer



def generate_text(prompt, model, tokenizer, max_length=200, device = 'cuda:0'):
    input_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
    output = model.generate(input_tokens,
        max_length=max_length,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        temperature=0.8
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)