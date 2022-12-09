from VisionModel.vision import get_models, get_nouns, run_vision_model
from torch.cuda import is_available
import tensorflow as tf

class ScenePrompt():
    def __init__(self, characters = [], nouns = [], first_line = "", first_character = ""): # lines is a list of tuples of (speaker, line)
        self.characters = characters if type(characters) == list else [characters]

        self.first_line = f"{first_character}: {first_line}"

        self.nouns = nouns 

        if first_character:
            self.first = first_character
        elif self.characters: self.first = tf.random.shuffle(self.characters)[0]
        else: self.first = "Dwight:"

    def to_text(self):
        output = f"Characters: " + ", ".join(set(self.characters)) + "\n\n"

        if self.nouns:
            output += "Nouns: " + ", ".join(self.nouns) + "\n\n" # Need to implement random sampling later

        output += "----TEXT----"

        output += "\n\n".join(
            [
                f"{character}: {line}"
                for character, line in zip(
                    self.characters[:self.n_lines],
                    self.lines[:self.n_lines]
                )
            ]
        )

        return output

class AtRM():
    def __init__(self, verbose = False):
        if verbose: print("Loading Vision models...")
        self.clip_model, self.processor, self.face_model, self.facecascade = get_models()
        self.nouns, self.words = get_nouns()

        if verbose:
            print("Vision models loaded.")
            print("Loading Language models...")

        self.device = "cuda" if is_available() else "cpu"

    def __str__(self) -> str:
        return f"Your Assistant (to) the Regional Manager is here"

    def __call__(self, img_file, first_line = "", first_character = ""):
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
            nouns = nouns,
            first_line = first_line,
            first_character = first_character
        )



    