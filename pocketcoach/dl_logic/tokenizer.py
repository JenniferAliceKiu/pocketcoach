import pickle

TOKENIZER_NAME = 'tokenizer.pkl'

def save(tokenizer):
    print(f"Saving tokenizer as {TOKENIZER_NAME}")
    with open(TOKENIZER_NAME, 'wb') as f:
        pickle.dump(tokenizer, f)

def load_tokenizer():
    print(f"Loading tokenizer with name {TOKENIZER_NAME}")
    with open(TOKENIZER_NAME, 'rb') as f:
        return pickle.load(f)
