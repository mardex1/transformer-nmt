
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast
device = 'cuda' if torch.cuda.is_available() else 'cpu'

MAX_LEN = 30
UNK_TOKEN = 3
SOS_TOKEN = 2
EOS_TOKEN = 1
PAD_TOKEN = 0

def train_tokenizers(files_path, lang1, lang2):
    files1 = [f"{files_path}{split}.{lang1}" for split in ["train", "val"]]
    files2 = [f"{files_path}{split}.{lang2}" for split in ["train", "val"]]

    tokenizer1 = Tokenizer(BPE(unk_token='[UNK]'))
    tokenizer2 = Tokenizer(BPE(unk_token='[UNK]'))

    trainer1 = BpeTrainer(special_tokens=["[PAD]", "[EOS]", "[SOS]", "[UNK]"])
    trainer2 = BpeTrainer(special_tokens=["[PAD]", "[EOS]", "[SOS]", "[UNK]"])   

    tokenizer1.pre_tokenizer = Whitespace()
    tokenizer2.pre_tokenizer = Whitespace() 

    tokenizer1.train(files1, trainer1)
    tokenizer2.train(files2, trainer2)

    tokenizer1.save('Tokenizers/tokenizer_en.json')
    tokenizer2.save('Tokenizers/tokenizer_pt.json')

    tokenizer1.post_processor = TemplateProcessing(
    single="$A [EOS]",
    special_tokens=[
        ("[EOS]", tokenizer1.token_to_id("[EOS]"))
    ]
    )
    tokenizer1.post_processor = TemplateProcessing(
    single="$A [EOS]",
    special_tokens=[
        ("[EOS]", tokenizer1.token_to_id("[EOS]"))
    ]
    )

    src_vocab_size = tokenizer1.get_vocab_size()
    tgt_vocab_size = tokenizer2.get_vocab_size()

    return tokenizer1, tokenizer2, src_vocab_size, tgt_vocab_size

def read_inference(lang1, lang2):
    with open('Datasets/opus-100_en-pt/text_files/val.pt', 'r', encoding='utf-8') as f:
        val_tgt_text = f.read()

    with open('Datasets/opus-100_en-pt/text_files/val.en', 'r', encoding='utf-8') as f:
        val_src_text = f.read()
    
    
    tokenizer1, tokenizer2, src_vocab_size, tgt_vocab_size = train_tokenizers('Datasets/opus-100_en-pt/text_files/', lang1, lang2)
    
    def filter(sentence_src, sentence_tgt):
        return len(tokenizer1.encode(sentence_src).ids) < MAX_LEN and len(tokenizer2.encode(sentence_tgt).ids) < MAX_LEN
    
    print('Len of validation tensors before shrinking', len(val_src_text.split('\n')), len(val_tgt_text.split('\n')))

    encoded_sentences_val = [[tokenizer1.encode(sentence_src).ids, tokenizer2.encode(sentence_tgt).ids] for (sentence_src, sentence_tgt) in zip(val_src_text.split('\n'), val_tgt_text.split('\n')) if filter(sentence_src, sentence_tgt)]

    val_len = len(encoded_sentences_val)

    input_val = np.zeros((val_len, MAX_LEN), dtype=np.int32)
    output_val = np.zeros((val_len, MAX_LEN), dtype=np.int32)

    for idx, (src, tgt) in enumerate(encoded_sentences_val):
        input_val[idx, :len(src)] = src
        output_val[idx, :len(tgt)] = tgt

    input_tensor_v = torch.tensor(input_val, dtype=torch.int64, device=device)
    output_tensor_v = torch.tensor(output_val, dtype=torch.int64, device=device)

    print('Len of validation tensors after shrinking: ', input_tensor_v.shape[0], output_tensor_v.shape[0])

    val_data = TensorDataset(input_tensor_v, output_tensor_v)

    val_dataloader = DataLoader(val_data, batch_size=1, shuffle=True)

    return  val_dataloader, src_vocab_size, tgt_vocab_size, tokenizer1, tokenizer2

def read_opus(batch_size, lang1, lang2):
    with open('Datasets/opus-100_en-pt/text_files/train.pt', 'r', encoding='utf-8') as f:
        train_tgt_text = f.read()

    with open('Datasets/opus-100_en-pt/text_files/train.en', 'r', encoding='utf-8') as f:
        train_src_text = f.read()

    with open('Datasets/opus-100_en-pt/text_files/val.pt', 'r', encoding='utf-8') as f:
        val_tgt_text = f.read()

    with open('Datasets/opus-100_en-pt/text_files/val.en', 'r', encoding='utf-8') as f:
        val_src_text = f.read()

    with open('Datasets/opus-100_en-pt/text_files/test.en', 'r', encoding='utf-8') as f:
        test_src_text = f.read()
    
    with open('Datasets/opus-100_en-pt/text_files/test.pt', 'r', encoding='utf-8') as f:
        test_tgt_text = f.read()
    
    tokenizer_pt, tokenizer_en, src_vocab_size, tgt_vocab_size = train_tokenizers('Datasets/opus-100_en-pt/text_files/', 'en', 'pt')

    def filter(sentence_src, sentence_tgt):
        return len(tokenizer_en.encode(sentence_src).ids) < MAX_LEN and len(tokenizer_pt.encode(sentence_tgt).ids) < MAX_LEN

    encoded_sentences_train = [[tokenizer_en.encode(sentence_src).ids, tokenizer_pt.encode(sentence_tgt).ids] for (sentence_src, sentence_tgt) in zip(train_src_text.split('\n'), train_tgt_text.split('\n')) if filter(sentence_src, sentence_tgt)]

    encoded_sentences_val = [[tokenizer_en.encode(sentence_src).ids, tokenizer_pt.encode(sentence_tgt).ids] for (sentence_src, sentence_tgt) in zip(val_src_text.split('\n'), val_tgt_text.split('\n')) if filter(sentence_src, sentence_tgt)]

    encoded_sentences_test = [[tokenizer_en.encode(sentence_src).ids, tokenizer_pt.encode(sentence_tgt).ids] for sentence_src, sentence_tgt in zip(test_src_text.split('\n'), test_tgt_text.split('\n')) if filter(sentence_src, sentence_tgt)]

    train_len = len(encoded_sentences_train)
    val_len = len(encoded_sentences_val)
    test_len = len(encoded_sentences_test)

    input_train = np.zeros((train_len, MAX_LEN), dtype=np.int32)
    output_train = np.zeros((train_len, MAX_LEN), dtype=np.int32)
    input_val = np.zeros((val_len, MAX_LEN), dtype=np.int32)
    output_val = np.zeros((val_len, MAX_LEN), dtype=np.int32)
    input_test = np.zeros((test_len, MAX_LEN), dtype=np.int32)
    output_test = np.zeros((test_len, MAX_LEN), dtype=np.int32)

    for idx, (src, tgt) in enumerate(encoded_sentences_train):
        input_train[idx, :len(src)] = src
        output_train[idx, :len(tgt)] = tgt

    for idx, (src, tgt) in enumerate(encoded_sentences_val):
        input_val[idx, :len(src)] = src
        output_val[idx, :len(tgt)] = tgt

    for idx, (src, tgt) in enumerate(encoded_sentences_test):
        input_test[idx, :len(src)] = src
        output_test[idx, :len(tgt)] = tgt

    input_tensor_t = torch.tensor(input_train, dtype=torch.int64, device=device)
    output_tensor_t = torch.tensor(output_train, dtype=torch.int64, device=device)
    input_tensor_v = torch.tensor(input_val, dtype=torch.int64, device=device)
    output_tensor_v = torch.tensor(output_val, dtype=torch.int64, device=device)
    input_tensor_test = torch.tensor(input_test, dtype=torch.int64, device=device)
    output_tensor_test = torch.tensor(output_test, dtype=torch.int64, device=device)

    print('Shape of train tensors: ', input_tensor_t.shape, output_tensor_t.shape)
    print('Shape of validation tensors: ', input_tensor_v.shape, output_tensor_v.shape)
    print('Shape of test tensor: ', input_tensor_test.shape, output_tensor_test.shape)

    train_data = TensorDataset(input_tensor_t, output_tensor_t)
    val_data = TensorDataset(input_tensor_v, output_tensor_v)
    test_data = TensorDataset(input_tensor_test, output_tensor_test)

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    return train_dataloader, val_dataloader, test_dataloader, src_vocab_size, tgt_vocab_size, tokenizer_en, tokenizer_pt

def add_padding(sentence_ids):
    while len(sentence_ids) < MAX_LEN:
        sentence_ids.append(0)
    return sentence_ids

def shift_right(labels):
    decoder_ids = torch.empty((labels.size(0), 1)).fill_(SOS_TOKEN).type(torch.int64).to(device)
    decoder_ids = torch.cat((decoder_ids, labels[:, :-1]), dim=1)
    return decoder_ids