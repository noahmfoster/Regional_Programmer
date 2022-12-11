from transformers import GPT2LMHeadModel

from transformers import GPT2TokenizerFast
from transformers import GPT2LMHeadModel
from transformers import GPTJForCausalLM
from datasets import load_dataset


import torch

import os
import numpy as np
import pandas as pd

import nltk

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

data_path = os.path.dirname(os.path.abspath(__file__)) + '/../Data/'


def get_tuned(model_path = ""):
    if model_path == "":
        model_path = model_path = os.path.dirname(os.path.abspath(__file__))
    fine_tuned_model = GPT2LMHeadModel.from_pretrained(model_path)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return fine_tuned_model, tokenizer

def get_GPTJ():
    model = GPTJForCausalLM.from_pretrained("EleutherAI/gpt-j-6B").half()
    tokenizer = GPT2TokenizerFast.from_pretrained("EleutherAI/gpt-j-6B")
    return model, tokenizer

def get_untuned():
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    return model, tokenizer

def get_language_model(model_name = 'trained', model_path = ""):
    if model_name == 'tuned':
        return get_tuned(model_path=model_path)
    elif model_name == 'untuned':
        return get_untuned()
    elif model_name == 'GPTJ' or model_name =='gptj':
        return get_GPTJ()
    else:
        return get_tuned(model_path=model_path)

def generate_text(prompt, model, tokenizer, device = 'cuda:0'):
    try:
        input_tokens = tokenizer.encode(prompt, return_tensors='pt').to(device)
        output = model.generate(input_tokens,
            max_length=500 + len(input_tokens),
            num_beams=5,
            no_repeat_ngram_size=2,
            early_stopping=True,
        )
    except RuntimeError:
        print("Model couldn't handle lenght of prompt. Please try again with fewer context scenes (or just try again and hope that it randomly chooses short scenes).")
    return tokenizer.decode(output[0], skip_special_tokens=True)

class ScenePrompt():
    def __init__(self, characters = [], nouns = [], lines = [], gen_nouns = False): # lines is a list of tuples of (speaker, line)
        
        if lines:
            self.characters = list(set([character for character, _ in lines]))
        else:
            self.characters = characters


        self.nouns = nouns
        self.lines = lines

        text = " ".join([line for _, line in lines])

        if gen_nouns:
            self.nouns += [word for (word, pos) in nltk.pos_tag(nltk.word_tokenize(text)) if pos[0] == 'N' and (pos != 'NNP')]

        self.lines = lines if lines else [(np.random.choice(self.characters, 1)[0] if self.characters else "Dwight", '')]
        
    def to_text(self, gen_nouns = False):
        if gen_nouns:
            self.nouns += [
                word for (word, pos) in nltk.pos_tag(
                    nltk.word_tokenize(

                        " ".join([line for _, line in self.lines])
                    )) if pos[0] == 'N' and (pos != 'NNP')]


        output = f"Characters: " + ", ".join(set(self.characters)) + "\n\n"

        if self.nouns:
            output += "Nouns: " + ", ".join(self.nouns) + "\n\n"

        output += "----TEXT----" + ("\n\n" if self.lines else "")
        self.len_prompt = len(output)

        output += "\n\n".join(
            [
                f"{character}: {line}"
                for character, line in self.lines
            ]
        )
        while output[-1] in {"\n", " "}:
            output = output[:-1]

        return output

    def __str__(self) -> str:
        return self.to_text()

    def __repr__(self) -> str:
        return self.to_text()

def get_prompts(data_file = data_path + 'The-Office-Lines-V4.csv'):
    data = pd.read_csv(data_file)
    data = data.drop("Unnamed: 6", axis=1)

    breaks = [0] + [i + 1 for i, scene_num in enumerate(data["scene"][1:]) if scene_num != data["scene"][i]] + [len(data["scene"])]
    n_scenes = len(breaks) - 1 # I added an extra "break" for the end of all the lines

    scenes = [
        ScenePrompt(
            characters = [],
            nouns = [],
            lines = list(zip(data["speaker"][breaks[i]:breaks[i+1]], data["line"][breaks[i]:breaks[i+1]])),
            gen_nouns = True
        )
        for i in range(n_scenes)
    ]
    return scenes

def tokenize(element, tokenizer, max_length=128,):  # TODO: Best value for max length ?????
    outputs = tokenizer(
        element["Prompts"],
        truncation=True,
        max_length=max_length,
        return_overflowing_tokens=False,
        return_length=True,
    )
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
        if length == max_length:
            input_batch.append(input_ids)
    return {"input_ids": input_batch}




def evaluate(model, tokenizer, test_size = 600, device = 'cuda:0', ):
    try:
        prompts_dataset = load_dataset('csv', data_files=data_path + 'test_prompts.csv', split='train') #The only split, I separate test later
    except FileNotFoundError:
        print("Unable to find test data.")
        return 0
    
    prompts_dataset = prompts_dataset.remove_columns('Unnamed: 0')
    test_dataset = prompts_dataset.shuffle(seed=42).select(range(6000))
    test_dataset = test_dataset.train_test_split(train_size=0.9, seed=42)  # Reselecting the train and test sets

    test_dataset = test_dataset["test"]

    tokenizer.pad_token = tokenizer.eos_token
    test_dataset_tokenized = test_dataset.map(
        lambda x: tokenize(x, max_length=128, tokenizer=tokenizer),
        batched=True, remove_columns=test_dataset.column_names
    )

    test_dataset_tokenized.set_format(type='torch', columns=['input_ids'])

    losses = []

    for i in range(min(len(test_dataset_tokenized), test_size)):
        input_ids = test_dataset_tokenized[i]["input_ids"].to(device)

        target_ids = input_ids.clone()
        target_ids[target_ids == tokenizer.pad_token_id] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            loss = outputs[0]
            log_likelihood = loss # * length #Length is always 128 so not needed
            losses.append(log_likelihood)
            
    
    return torch.exp(torch.stack(losses )).mean().numpy()
        




    
    
if __name__ == '__main__':
    model, tokenizer = get_language_model('trained')
    model.eval()

    ppl = evaluate(model = model, tokenizer = tokenizer, test_size = 10, device = 'cpu')
    print(ppl)