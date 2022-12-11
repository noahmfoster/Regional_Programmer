from VisionModel.vision import get_models, get_nouns, run_vision_model
from LanguageModel.Language import get_language_model, generate_text, ScenePrompt, get_prompts, evaluate
from torch.cuda import is_available

import numpy as np


class AtRM():
    def __init__(self, verbose = False, lm = "tuned", in_context_learning = False):
        '''Full Assistant to the Regional Manager model.
        lm: "tuned", "untuned", "GPTJ"
        verbose: print out model loading progress
        in_context_learning: Collect Training Data to provide context to the model'''
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

        if in_context_learning:
            self.in_context_learning = True
            self.context = get_prompts()
        else:
            self.in_context_learning = False

        self.last_prompt = None

    
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

    def get_context(self):
        self.context = get_prompts()

    def __call__(self, img_file, first_character = "", first_line = "", include_nouns = True, include_prompt = True, n_context_scene = 0):
        character_vector, nouns = self.promptify_img(img_file)
        self.last_prompt = ScenePrompt(
            characters = character_vector,
            nouns = nouns if include_nouns else [],
            lines = [(first_character, first_line)] if first_character != "" else []
        )

        return self.generate_text(self.last_prompt, include_prompt = include_prompt, n_context_scene = n_context_scene)



    
    def promptify_img(self, img_file):
        try:
            character_vector, nouns = run_vision_model(
                img_file,
                self.clip_model,
                self.processor,
                self.face_model,
                self.facecascade,
                self.nouns,
                self.words
            )
        except TypeError: # For some images, the ouput of run_vision_model is None, which raises a TypeError. Not sure why this happens
            print('It seems the models have produced erroneous results. Please try again with a different image. Continuing with empty prompt.')
            return [], []


        return character_vector, nouns

    def generate_text(self, prompt, include_prompt = True, n_context_scene = 0):
        assert prompt is not None, "You must provide a prompt to generate text."
        
        text = ''
        if self.in_context_learning:
            for _ in range(n_context_scene):
                text += self.context[np.random.randint(len(self.context))].to_text()
                text += "\n\n New Scene \n\n"
        elif n_context_scene > 0:
            print("Warning: In context learning is not enabled for this model. n_context_scene will be ignored.")

        garbage = len(text)

        text += prompt.to_text()

        output_w_prompt = generate_text(
            text,
            self.lm,
            self.tokenizer,
            device = self.device,
        )

        if include_prompt:
            return output_w_prompt
        else:
            return output_w_prompt[prompt.len_prompt + garbage:]

    def evalutate_lm(self, n = 100):
        if self.in_context_learning:
            return evaluate(self.lm, self.tokenizer, n = n, device = self.device)
        else:
            print("Requires Data from Context in order to evaluate. Please run <model>.get_context() to collect data.")





    