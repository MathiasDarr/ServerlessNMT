import tensorflow as tf



import unicodedata
import re
import numpy as np
import os
import io
import time



def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping
    # -punctuation
    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.strip()
    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w


# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, FRENCH]
def create_dataset(path, num_examples):
    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[preprocess_sentence(w) for w in l.split('\t')] for l in lines[:num_examples]]
    dataset_columns = list(zip(*word_pairs))
    return dataset_columns[0], dataset_columns[1]
    # return [[s[0], s[1]] for s in word_pairs]

# return list(zip(*word_pairs))
def tokenize(lang):
    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
    lang_tokenizer.fit_on_texts(lang)
    tensor = lang_tokenizer.texts_to_sequences(lang)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, lang_tokenizer


def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    targ_lang, inp_lang = create_dataset(path, num_examples)
    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)
    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer




# Try experimenting with the size of that dataset
num_examples = 30000
# dataset = create_dataset('data/fra-eng/fra.txt', num_examples)


input_tensor, target_tensor, inp_lang, targ_lang = load_dataset('data/fra-eng/fra.txt', num_examples)
#
# # Calculate max_length of the target tensors
# max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1]
#
# # Creating training and validation sets using an 80-20 split
# input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
#                                                                                                 target_tensor,
#                                                                                                 test_size=0.2)
#
# # Show length
# print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
