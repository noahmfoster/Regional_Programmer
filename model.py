from VisionModel.vision import get_models, get_nouns, run_vision_model
from LanguageModel.Language import get_language_model, generate_text, ScenePrompt, get_prompts
from torch.cuda import is_available

import numpy as np


class AtRM():
    def __init__(self, verbose = False, lm = "tuned", in_context_learing = False):
        '''Full Assistant to the Regional Manager model.
        lm: "tuned", "untuned", "GPTJ"
        verbose: print out model loading progress
        in_context_learing: Collect Training Data to provide context to the model'''
        if verbose: print("Loading Vision models...")

        self.clip_model, self.processor, self.face_model, self.facecascade = get_models()


        self.nouns, self.words = get_nouns()

        if verbose:
            print("Vision models loaded.")
            print("Loading Language models...")

        self.lm, self.tokenizer = get_language_model(model_name = lm)

        if verbose: print("Language models loaded.")

        self.device = "cuda" if is_available() else "cpu"
        self.lm.to(self.device)

        if in_context_learing:
            self.in_context_learing = True
            self.context = get_prompts()
        else:
            self.in_context_learing = False
    
    def init_overide(self, clip_model = None, processor = None, face_model = None, facecascade = None, nouns = None, words = None, lm = None, tokenizer = None, device = None):
        '''For debugging purposes, allows you to overide the models with your own.'''
        if clip_model: self.clip_model = clip_model
        if processor: self.processor = processor
        if face_model: self.face_model = face_model
        if facecascade: self.facecascade = facecascade
        if nouns: self.nouns = nouns
        if words: self.words = words
        if lm: self.lm = lm
        if tokenizer: self.tokenizer = tokenizer
        if device: self.device = device

    def __str__(self) -> str:
        return f"Your Assistant (to) the Regional Manager is here!"

    def __call__(self, img_file, first_character = "", first_line = "", include_nouns = True, include_prompt = True, n_context_scene = 0):
        
        character_vector, nouns = run_vision_model(
            img_file,
            self.clip_model,
            self.processor,
            self.face_model,
            self.facecascade,
            self.nouns,
            self.words
        )

        
        prompt = ScenePrompt(
            characters = character_vector,
            nouns = nouns if include_nouns else [],
            lines = [(first_character, first_line)] if first_character != "" else []
        )
        
        text = ''
        if self.in_context_learing:
            for _ in range(n_context_scene):
                text += self.context[np.random.randint(len(self.context))].to_text()
                text += "\n\n New Scene \n\n"

        garbage = len(text)

        text += prompt.to_text()

        output_w_prompt = generate_text(
            text,
            self.lm,
            self.tokenizer,
            device = self.device
        )

        if include_prompt:
            return output_w_prompt
        else:
            return output_w_prompt[prompt.len_prompt + garbage:]





    