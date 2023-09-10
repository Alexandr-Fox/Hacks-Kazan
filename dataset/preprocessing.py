import random
from typing import Tuple, Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn import preprocessing
import numpy as np


def get_tokenizer_model(model_name: str = "deepvk/deberta-v1-base") -> Tuple[Any, Any]:
    """
    Get tokenizer and model from model name
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    return tokenizer, model


def normalize(data: np.ndarray) -> np.ndarray:
    """
    Normalize the data
    """
    return preprocessing.normalize(data, norm='l2')


def standardize(data: np.ndarray) -> np.ndarray:
    """
    Standardize the data
    """
    scaler = preprocessing.StandardScaler()
    scaler.fit(data)
    return scaler.transform(data)


def classes_to_numbers(data: np.ndarray) -> np.ndarray:
    """
    Convert the classes to numbers
    """
    le = preprocessing.LabelEncoder()
    le.fit(data)
    return le.transform(data)


def get_word_idx(sent: str, word: str) -> int:
    """split sentences and add index to each word. Each word has its own index based on when it was added to the list first
    Args:
        sent (str): sentence in string
        word (str): word in string
    Returns:
        int: output the index of where the word correspond to in each sentence input
    """
    return sent.lower().split(" ").index(word)


# Helper func 2
def get_hidden_states(sent, tokenizer, model, layers):
    """Push input IDs through model. Stack and sum `layers` (last four by default).
       Select only those subword token outputs that belong to our word of interest
       and average them.
    Args:
        sent (str): Input sentence
        tokenizer : Tokenizer function
        model: bert model
        layers : last 4 model of model
    Returns:
        output: tensor torch
    """
    # encode without adding [CLS] and [SEP] tokens
    encoded = tokenizer.encode_plus(sent, return_tensors="pt", add_special_tokens=False)

    with torch.no_grad():
        output = model(**encoded)

    # Get all hidden states
    states = output.hidden_states
    # Stack and sum all requested layers
    output = torch.stack([states[i] for i in layers]).sum(0).squeeze()
    # Only select the tokens that constitute the requested word
    return output


# Helper func 3
def chunking(max_len, sent):
    """because the embedding function is trained on dim 512, so we have to limit the size of the sentences using max_len so the final chunked sentences wont exceed length 512
    Args:
        max_len (int): maximum number of tokens for each chunk
        sent (str): input sentence
    Returns:
        sent_chunk (List(str)): list of chunked sentences
    """
    tokenized_text = sent.lower().split(" ")
    # using list comprehension
    final = [
        tokenized_text[i * max_len : (i + 1) * max_len]
        for i in range((len(tokenized_text) + max_len - 1) // max_len)
    ]

    # join back to sentences for each of the chunks
    sent_chunk = []
    for item in final:
        # make sure the len(items) > 1 or else some of the embeddings will appear as len 1 instead of 768.
        assert len(item) > 1
        sent_chunk.append(" ".join(item))
    return sent_chunk


def embeddings_avg(sent: str, tokenizer, model, layers=None, chunk_size=300):
    """Gives the average word embedding per sentence

    Args:
        sent (str): The input sentence

    Returns:
        torch tensor: word embedding per sentence, dim = 768
    """
    # change all standard form numbers to decimal
    np.set_printoptions(formatter={"float_kind": "{:f}".format})

    # Use last four layers by default
    layers = [-4, -3, -2, -1] if layers is None else layers

    # chunking
    chunked_tokens = chunking(chunk_size, sent)  # helper func 3

    # initialise a outside chunk
    word_embedding_avg_collective = []
    # for each chunked token, we embed them separately
    for item in chunked_tokens:
        # adding tensors
        word_embedding_torch = get_hidden_states(
            item, tokenizer, model, layers
        )  # helper fun 2

        # convert torch tensor to numpy array
        word_embedding_avg_np = word_embedding_torch.cpu().detach().numpy()
        word_embedding_avg_chunks = np.mean(word_embedding_avg_np, axis=0)
        word_embedding_avg_collective.append(word_embedding_avg_chunks)
    word_embedding_avg = np.mean(word_embedding_avg_collective, axis=0)
    assert len(word_embedding_avg) == 768
    return word_embedding_avg


def merge_vectors_by_avg(vectors: np.ndarray) -> np.ndarray:
    """
    Merge vectors by average
    """
    return np.mean(vectors, axis=0)


def get_popular(data: pd.DataFrame) -> np.ndarray:
    popular = data["item_id"].to_numpy()
    popular = np.reshape(popular, (10, 10))
    result = []
    for i in range(10):
        result.append(popular[i][random.randint(0, 9)])
    result = np.random.permutation(result)
    return result
