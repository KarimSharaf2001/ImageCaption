import tensorflow as tf
import unicodedata
import re
import numpy as np

# Load and clean data
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_sentence(sentence, lang='eng'):
    sentence = unicode_to_ascii(sentence.lower().strip())
    sentence = re.sub(r"([?.!,؟،])", r" \1 ", sentence)
    sentence = re.sub(r'[" "]+', " ", sentence)
    sentence = re.sub(r"[^a-zA-Z?.!,؟،ء-ي ]+", "", sentence) if lang == 'eng' else sentence
    sentence = sentence.strip()
    sentence = '<start> ' + sentence + ' <end>'
    return sentence

def load_dataset(path, num_examples=None):
    with open(path, encoding='utf-8') as f:
     lines = f.read().strip().split('\n')
     arabic, english = [], []
    for line in lines[:num_examples]:
        ar, en = line.split('\t')
        arabic.append(preprocess_sentence(ar, lang='ar'))
        english.append(preprocess_sentence(en, lang='eng'))

    return arabic, english

def tokenize(lang_sentences):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>')
    tokenizer.fit_on_texts(lang_sentences)
    tensor = tokenizer.texts_to_sequences(lang_sentences)
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
    return tensor, tokenizer
# Load and preprocess
num_examples = 10000  # change this to None to load all
arabic_sentences, english_sentences = load_dataset("./translate/ara_eng.txt", num_examples)

input_tensor, inp_lang_tokenizer = tokenize(english_sentences)
target_tensor, targ_lang_tokenizer = tokenize(arabic_sentences)
# Save tokenizer for later use
import pickle
with open('inp_lang_tokenizer.pkl', 'wb') as f:
    pickle.dump(inp_lang_tokenizer, f)
with open('targ_lang_tokenizer.pkl', 'wb') as f:
    pickle.dump(targ_lang_tokenizer, f)

# Save as numpy arrays
np.save('input_tensor.npy', input_tensor)
np.save('target_tensor.npy',target_tensor)