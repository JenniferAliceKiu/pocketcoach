from tensorflow.keras import Sequential, Input, layers

def base_model():

    tk = Tokenizer()
    tk.fit_on_texts(X)
    vocab_size = len(tk.word_index)

    embedding_size = 50

    model = Sequential()
    model.add(layers.Embedding(input_dim=vocab_size + 1, output_dim=embedding_size, mask_zero=True))
    model.add(layers.Conv1D(16, kernel_size=3))
    model.add(layers.Flatten())
    model.add(layers.Dense(5,))
    model.add(layers.Dense(6, activation='softmax'))
