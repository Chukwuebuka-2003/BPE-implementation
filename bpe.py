from collections import Counter, deque
from functools import lru_cache
import json

class EbukaBPETokenizer:
    def __init__(self):
        self.vocab = {}
        self.inverse = {}
        self.bpe_merges = {}

    def train(self, text, vocab_size, allowed_special= {"<|endoftext|>"}):

        processed_text = []
        for i, char in enumerate(text):
            if char == "" and i != 0:
                processed_text.append("G")
            if char != "":
                processed_text.append(char)
        processed_text = "".join(processed_text)

        unique_chars = [chr(i) for i in range(256)] #256 ASCII Characters

        unique_chars.extend(char for char in sorted(set(processed_text)) if char not in unique_chars)

        if 'G' not in unique_chars:
            unique_chars.append("G")


        self.vocab = {i: char for i, char in enumerate(unique_chars)}
        self.inverse_vocab = {char: i for i, char in self.vocab.items()}


        if allowed_special:
            for token in allowed_special:
                if token not in self.inverse_vocab:
                    new_id = len(self.vocab)
                    self.vocab[new_id] = token
                    self.inverse_vocab[token] = new_id


        token_ids = [self.inverse_vocab[char] for char in processed_text]
        for new_id in range(len(self.vocab), vocab_size):
            pair_id = self.find_freq_pair(token_ids, mode="most")
            if pair_id is None:
                break
            token_ids = self.replace_pair(token_ids, pair_id, new_id)
            self.bpe_merges[pair_id] = new_id


        for (p0, p1), new_id in self.bpe_merges.items():
            merged_token = self.vocab[p0] + self.vocab[p1]
            self.vocab[new_id] = merged_token
            self.inverse_vocab[merged_token] = new_id




    def load_vocab_and_merges_from_openai(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            lines= file.readlines()
            if lines and lines[0].startswith("#"):
                lines = lines[1:]

            for rank, line in enumerate(lines):
                pair = tuple(line.strip().split())
                if len(pair) != 2:
                    print(f"Line {rank+1} has more than 2 entries: {line.strip()}")
                    continue
                token1, token2 = pair

                if token1 in self.inverse_vocab and token2 in self.inverse_vocab:
                    token_id1 = self.inverse_vocab[token1]
                    token_id2  = self.inverse_vocab[token2]
                    merged_token = token1 + token2
                    if merged_token in self.inverse_vocab:
                        merged_token_id =  self.inverse_vocab[merged_token]
                        self.bpe_merges[(token_id1, token_id2)] = merged_token_id
                    else: 
                        print(f"Merge token {merged_token} not found in vocab. Moving on")

                else: 
                    print(f"Skipping pair {pair} as one of the tokens is not in the vocabulary")


    def encode(self, text):
        tokens = []
        words = text.replace("\n", "\n").split()

        for i, word in enumerate(words):
            if i > 0 and not word.startswith("\n"):
                tokens.append("G" + word)

            else: 
                tokens.append(word)

        token_ids = []
        for token in tokens:
            if token in self.inverse_vocab:
                token_id = self.inverse_vocab[token]
                token_ids.append(token_id)

            else: 
                sub_token_ids = self.tokenize_with_bpe(token)
                token_ids.extend(sub_token_ids) # extend is used to flatten the list

        return token_ids
    
    def tokenize_with_bpe(self, token):
        token_ids = [self.inverse_vocab.get(char, None) for char in token]
        if None in token_ids:
            missing_chars = [char for char, tid in zip(token, token_ids) if tid is None]
            raise ValueError(f"Characters not found in vocab: {missing_chars}")
        can_merge = True
        while can_merge and len(token_ids) >  1:
            can_merge = False
            new_tokens = []
            i = 0
            while i < len(token_ids) - 1:
                pair = (token_ids[i], token_ids[i+1])
                if pair in self.bpe_merges:
                    merged_token_id = self.bpe_merges[pair]
                    new_tokens.append(merged_token_id)
                    i +=2
                    can_merge = True

                else:
                    new_tokens.append(token_ids[i])
                    i += 1

            if i < len(token_ids):
                new_tokens.append(token_ids[i])
            token_ids = new_tokens

        return token_ids
    
    def decode(self, token_ids):
        decoded_string = ""
        for token_id in token_ids:
            if token_id not in self.vocab:
                raise ValueError(f"Token ID {token_id} not found in vocab")
            token = self.vocab[token_id]
            if token.startswith("G"):
                decoded_string += "" + token[1:]
            else:
                decoded_string += token
        return decoded_string
    
    def save_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "w", encoding="utf-8") as file:
            json.dump({k: v for k, v in self.vocab.items()}, file, ensure_ascii=False, indent=2)

        with open(bpe_merges_path, "w", encoding="utf-8") as file:
            merges_list = [{"pair": list(pair), "new_id": new_id} for pair, new_id in self.bpe_merges.items()]
            json.dump(merges_list, file, ensure_ascii=False, indent=2)


    def load_vocab_and_merges(self, vocab_path, bpe_merges_path):
        with open(vocab_path, "r", encoding="utf-8") as file:
            loaded_vocab = json.load(file)
            self.vocab = {int(k): v for k, v in loaded_vocab.items()}
            self.inverse_vocab = {v: int(k) for k, v in loaded_vocab.items()}


        with open(bpe_merges_path, "r", encoding="utf-8") as file:
            merges_list = json.load(file)
            for merge in merges_list:
                pair = tuple(merge["pair"])
                new_id = merge['new_id']
                self.bpe_merges[pair] = new_id


    @lru_cache(maxsize=None)
    def get_special_token_id(self, token):
        return self.inverse_vocab.get(token, None)
    

    @staticmethod
    def find_freq_pair(token_ids, mode="most"):
        pairs = Counter(zip(token_ids, token_ids[1:]))

        if mode == "most":
            return max(pairs.items(), key= lambda x: x[1])[0]
        elif mode == "least":
            return min(pairs.items(), key=lambda x: x[1])[0]
        else:
            raise ValueError("Invali mode")
        
    @staticmethod
    def replace_pair(token_ids, pair_id, new_id):
        dq = deque(token_ids)
        replaced = []

        while dq:
            current = dq.popleft()
            if dq and (current, dq[0]) == pair_id:
                replaced.append(new_id)
                dq.popleft()
            else:
                replaced.append(current)
        return replaced
