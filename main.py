# training the BPE Tokenizer
import os 
from bpe import EbukaBPETokenizer

with open("romeo_and_juliet.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = EbukaBPETokenizer()

tokenizer.train(text, vocab_size=1000, allowed_special = {"<|endoftext"})

print(len(tokenizer.bpe_merges)) #this vocabulary is created by merging 743 times, which means the first 256 entries are single-character tokens


# now testing the created merges to encode sample texts

sample_text = "Ebuka embraced the simplicity of life"
token_ids = tokenizer.encode(sample_text)
print(token_ids)

print("No of Characters:", len(sample_text))
print("No of token ids:", len(token_ids))

# Here, 37 characters wer encoded into 25 token id

# decoding now
print(tokenizer.decode(token_ids))

for token_id in token_ids:
    print(f"{token_id} -> {tokenizer.decode([token_id])}")
