# BPE Tokenizer Implementation

This repository contains a custom implementation of a Byte Pair Encoding (BPE) tokenizer, along with scripts for training, encoding, and decoding. The implementation is based on the BPE algorithm used in popular models like GPT and is designed for educational and experimental purposes.

## Files

### `file.py`

This script is responsible for downloading the plain text of Romeo and Juliet from Project Gutenberg, which serves as a sample text for training the BPE tokenizer.

### `bpe.py`

This file contains the core implementation of the `EbukaBPETokenizer` class. It includes methods for:

-   **Training:**  `train(text, vocab_size, allowed_special)`
    -   Learns BPE merges from a given text.
    -   Handles the creation of vocabulary and merge rules.
    -   Initializes the vocabulary with ASCII characters and allows for adding special tokens.
-   **Encoding:** `encode(text)`
    -   Converts input text into token IDs.
    -   Handles tokenization using the learned BPE merges and vocab.
-   **Decoding:** `decode(token_ids)`
    -   Converts a sequence of token IDs back into text.
-   **Loading and Saving:**  `load_vocab_and_merges(vocab_path, bpe_merges_path)` and `save_vocab_and_merges(vocab_path, bpe_merges_path)`
    -   Provides functionality to persist the trained vocabulary and merge rules to disk and load them again.
-   **Loading from OpenAI format:** `load_vocab_and_merges_from_openai(vocab_path, bpe_merges_path)`
    - Supports loading vocab and merges in the format used by OpenAI.
- **Internal Methods** `find_freq_pair(token_ids, mode)` and `replace_pair(token_ids, pair_id, new_id)`
    - These functions are used internally for finding the most frequent token pairs and for replacing tokens during training.

### `main.py`

This script provides an example of how to use the `EbukaBPETokenizer`:

-   Loads the "Romeo and Juliet" text.
-   Trains a BPE tokenizer with a vocabulary size of 1000.
-   Encodes a sample text using the trained tokenizer.
-   Prints the encoded token IDs and their length
-   Decodes the token IDs back to text.
-   Decodes each individual token ID.

## Usage

### 1. Download the Dataset
Run `file.py` to download the `romeo_and_juliet.txt` text file.
```bash
python file.py
```
### 2. Train and Test the Tokenizer

Run main.py to train the tokenizer, encode a sample text, and then decode it:

```bash
python main.py
```

This will:

    Train a BPE tokenizer on romeo_and_juliet.txt.

    Print the number of BPE merges made during training.

    Encode the sample text "Ebuka embraced the simplicity of life".

    Print the resulting token IDs and the number of token ids as well as the original number of characters.

    Decode the token IDs and print the original string.

    Decode each token ID individually and print the decoded text.

Training Parameters

The tokenizer is trained using the following parameters:

    text: The text used to train the tokenizer.

    vocab_size: The maximum size of the vocabulary.

    allowed_special: A set of special tokens that are added to the vocabulary and are never split or merged (for example, <|endoftext|>).

Customization
Vocabulary Size

The vocabulary size can be modified by changing the vocab_size parameter in main.py when calling the tokenizer.train() method.
Special Tokens

Additional special tokens can be added by modifying the allowed_special parameter when training.
Saving and Loading

The vocabulary and merge rules can be saved to and loaded from files using the save_vocab_and_merges and load_vocab_and_merges methods. This allows you to reuse trained tokenizers without needing to train them again.
Example Output of main.py

      
Downloaded 'Romeo and Juliet' as romeo_and_juliet.txt
743
[352, 302, 373, 340, 363, 493, 348, 288, 355, 358, 384, 346, 363, 493, 329, 364, 322, 384, 358, 365, 358, 294, 373, 287]
No of Characters: 37
No of token ids: 25
Ebuka embraced the simplicity of life
352 -> E
302 -> b
373 -> u
340 -> k
363 -> a
493 -> G e
348 -> m
288 -> b
355 -> r
358 -> a
384 -> c
346 -> e
363 -> d
493 -> G t
329 -> h
364 -> e
322 -> G s
384 -> i
358 -> m
365 -> p
358 -> l
294 -> i
373 -> c
287 -> y

Notes

    This is a basic implementation of a BPE tokenizer designed for learning purposes. It may not include all optimizations found in production-level implementations.

    The G is used to indicate the separation between words in the encoding.

    The special token "<|endoftext|>" is added to the vocabulary to provide an example of how to include special tokens, but it is not actually used by the tokenizer.

    The load_vocab_and_merges_from_openai method provides a compatibility feature for loading vocab and merges in the format found in OpenAI's tokenizers. However, it is important to be aware of potential differences in preprocessing and special token handling that might make it not directly compatible.

    The tokenizer handles characters not found in the initial vocab by first creating tokens for the basic ascii characters, followed by including all unique characters from the training data. If a new character not present during training is encountered during encoding, a ValueError will be thrown
