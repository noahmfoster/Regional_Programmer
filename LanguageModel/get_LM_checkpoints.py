#pip install gdown

import gdown
url = https://drive.google.com/drive/folders/1m9IoP7tIIALx5H7KsbQ1ZIPg4LfOe1gZ
output = 'checkpoints'
gdown.download(url, output, quiet=False)


