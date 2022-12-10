#pip install gdown
import os
import gdown
curr_path = os.path.dirname(os.path.abspath(__file__))

purposes = ['LanguageModel Model', 'VisionModel Model', 'LanguageModel Config']

urls = [
    'https://drive.google.com/uc?id=1-OBcLD6D5a0dDhroXczdtbOlg--51bYG',
    'https://drive.google.com/uc?id=1RTq_2ERsP9FJYS3NEHTlJaiBccXczRos',
    'https://drive.google.com/uc?id=1-L9sxJWAeTUyOoOPOFKf-s_HJKKqsYSh'
]
paths = [
    os.path.join(curr_path, 'LanguageModel/pytorch_model.bin'),
    os.path.join(curr_path, 'VisionModel/transfer_learning_trained_the_office_cnn_model.h5'),
    os.path.join(curr_path, 'LanguageModel/config.json')]


for url, path, purpose in zip(urls, paths, purposes):
    print(f'Downloading {purpose} from {url} to {path}')
    gdown.download(url, path, quiet=False)


