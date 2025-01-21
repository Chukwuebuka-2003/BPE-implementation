# training the BPE Tokenizer
import os 
import urllib.request

# URL for the plain text version of Romeo and Juliet on Project Gutenberg
url = 'https://www.gutenberg.org/files/1777/1777.txt'

# Specify the filename to save the content
filename = 'romeo_and_juliet.txt'

# Downloading the file
urllib.request.urlretrieve(url, filename)

print(f"Downloaded 'Romeo and Juliet' as {filename}")
