#pip install gdown
import os
import gdown
url = 'https://drive.google.com/drive/folders/1m9IoP7tIIALx5H7KsbQ1ZIPg4LfOe1gZ'
model_path = os.path.dirname(os.path.abspath(__file__)) + '/Model'
output = model_path
gdown.download(url, output, quiet=False)


