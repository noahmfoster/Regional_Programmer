#pip install gdown
import os
import gdown
lm_url = 'https://drive.google.com/drive/folders/1m9IoP7tIIALx5H7KsbQ1ZIPg4LfOe1gZ'
curr_path = os.path.dirname(os.path.abspath(__file__))

urls = [
    'https://drive.google.com/file/d/1-OBcLD6D5a0dDhroXczdtbOlg--51bYG/view?usp=share_link',
    'https://drive.google.com/file/d/1RTq_2ERsP9FJYS3NEHTlJaiBccXczRos/view?usp=share_link',
    'https://drive.google.com/file/d/1-L9sxJWAeTUyOoOPOFKf-s_HJKKqsYSh/view?usp=share_link'
]
paths = [
    os.path.join(curr_path, 'LanguageModel/pytorch_model.bin'),
    os.path.join(curr_path, 'VisionModel/transfer_learning_trained_the_office_cnn_model.h5'),
    os.path.join(curr_path, 'LanguageModel/config.json')]


for url, path in zip(urls, paths):
    gdown.download(url, path, quiet=False)


