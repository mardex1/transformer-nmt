import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import matplotlib.pyplot as plt
import numpy as np
from model import Transformer
from data import shift_right, read_opus, read_inference
PATH = 'Models/'

num_layers = 6
dropout = 0.1
d_model = 512
n_heads = 8
EOS_TOKEN = 1
SOS_TOKEN = 2
PAD_TOKEN = 0
learning_rate = 1e-4
num_epochs = 20
MAX_LENGTH = 30
batch_size = 128

def train(model, train_dataloader, val_dataloader, pre_train, tgt_vocab_size):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    if pre_train == True:
        model.load_state_dict(torch.load('Models/model_opus.pt'))
        optimizer.load_state_dict(torch.load('Models/optim_opus.pt'))
    
    lossi = []
    best_loss = float('inf')
    for epoch in range(num_epochs):
        total_loss = 0
        iteration = 0
        for data in train_dataloader:
            encoder_ids, labels = data
            decoder_ids = shift_right(labels)
            logits = model(encoder_ids, decoder_ids)
            logits = logits.contiguous().view(-1, logits.shape[-1])
            labels = labels.contiguous().view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=0)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            lossi.append(loss.item())
            print(f'{iteration} out of {len(train_dataloader)}')
            iteration += 1
        
        val_loss, val_losses = validation(model, val_dataloader, tgt_vocab_size, epoch)

        train_loss = total_loss/len(train_dataloader)
        val_loss = sum(val_losses)/len(val_dataloader)
        print(f"train loss: {train_loss}, epoch: {epoch}")
        print(f"val loss: {val_loss}, epoch: {epoch}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), "Models/model_opus.pt")
            torch.save(optimizer.state_dict(), "Models/optim_opus.pt")

def validation(model, val_dataloader, tgt_vocab_size, epoch):
    model.eval()
    lossi = []
    with torch.no_grad():
        for data in val_dataloader:
            encoder_ids, labels = data
            decoder_ids = shift_right(labels)
            logits = model(encoder_ids, decoder_ids)
            logits = logits.contiguous().view(-1, logits.shape[-1])
            labels = labels.contiguous().view(-1)
            loss = F.cross_entropy(logits, labels, ignore_index=0)
            print(loss.item())
            lossi.append(loss.item())
    model.train()
    return loss, lossi
    
def translate(model, test_dataloader, src_tokenizer, tgt_tokenizer):
    model.load_state_dict(torch.load('Models/model_opus.pt'))

    model.eval()
    
    encoder_input, label = next(iter(test_dataloader))

    decoder_input_tensor = torch.tensor([[SOS_TOKEN]], dtype=torch.int64).to(device)

    with torch.no_grad():
        for _ in range(MAX_LENGTH):

            logits = model(encoder_input, decoder_input_tensor)
            next_item = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(0)
            decoder_input_tensor = torch.cat((decoder_input_tensor, next_item), dim=1)

            if next_item.view(-1).item() == EOS_TOKEN:
                break

    list_translated = tgt_tokenizer.decode(decoder_input_tensor.view(-1).tolist())
    list_label = tgt_tokenizer.decode(label.view(-1).tolist())
    list_src = src_tokenizer.decode(encoder_input.view(-1).tolist())   
    
    print(f"Frase Original: {''.join(list_src)}")
    print(f"Frase Traduzida pelo modelo: {''.join(list_translated)}")
    print(f"Frase Traduzida correta: {''.join(list_label)}")

def translate_beam(model, test_dataloader, src_tokenizer, tgt_tokenizer, beam_width):
    model.load_state_dict(torch.load('Models/model_opus.pt'))

    model.eval()
    with torch.no_grad():
        encoder_input, label = next(iter(test_dataloader))

        decoder_input_tensor = torch.tensor([[SOS_TOKEN]], dtype=torch.int64).to(device)
        logits = model(encoder_input, decoder_input_tensor)
        next_probabilities = logits[:, -1, :]
        log_probs = F.log_softmax(next_probabilities.squeeze(), dim=-1)
        final_probs, idx = log_probs.topk(k = beam_width, dim=-1)
        decoder_input_tensor = decoder_input_tensor.repeat((beam_width, 1))
        next_items = idx.view(-1, 1)
        # Loop

        decoder_input_tensor = torch.cat((decoder_input_tensor, next_items), dim=-1)
        result = []
        while(len(result) < beam_width):
            logits = model(encoder_input, decoder_input_tensor) 
            next_probabilities = logits[:, -1, :]
            log_probs = F.log_softmax(next_probabilities, dim=-1)
            probs, idxs = log_probs.topk(k=beam_width, dim=-1)
            next_items_indices = torch.topk(probs.flatten(), decoder_input_tensor.size(0)).indices.tolist()
            next_items_sorted = sorted(next_items_indices)
            dec_idxs = []
            for idx in next_items_sorted:
                dec_idx = int(idx / beam_width) 
                dec_idxs.append(dec_idx)
            
            for i, idx in enumerate(dec_idxs):   
                if i == 0:
                    temp = torch.tensor(decoder_input_tensor[idx], dtype=torch.int64, device=device).unsqueeze(0)
                else:
                    temp = torch.cat((temp, decoder_input_tensor[idx].unsqueeze(0)), dim=0)
            decoder_input_tensor = temp

            next_ids = idxs.flatten()[next_items_sorted].view(-1, 1)
            
            decoder_input_tensor = torch.cat((decoder_input_tensor, next_ids), dim=-1)
            final_probs = final_probs + torch.topk(probs.flatten(), beam_width).values
            idx_to_remove = []
            for idx, sequence in enumerate(decoder_input_tensor):
                if EOS_TOKEN in sequence:
                    result.append([sequence, final_probs[idx].item()])
                    idx_to_remove.append(idx)
            for idx in sorted(idx_to_remove, reverse=True):
                temp = decoder_input_tensor.tolist()
                del temp[idx]
                decoder_input_tensor = torch.tensor(temp, dtype=torch.int64, device=device)
            
        # Normalize
        for elem in result:
            elem[1] = elem[1]/len(elem[0])
        
        # decode
        for elem in result:
            elem[0] = ''.join(tgt_tokenizer.decode(elem[0].tolist()))
        
        sorted(result, key=lambda item:item[1])
        tgt_pred = result[0][0]
        src_sentence = ''.join(src_tokenizer.decode(encoder_input.squeeze().tolist()))
        label = ''.join(tgt_tokenizer.decode(label.squeeze().tolist()))

        print(f"Frase Original: {src_sentence}")
        print(f"Frase Traduzida pelo modelo: {tgt_pred}")
        print(f"Frase Traduzida correta: {label}")
    

def run(mode):
    if mode == 'inference':
        dataloader, src_vocab_size, tgt_vocab_size, src_tokenizer, tgt_tokenizer = read_inference(lang1='en', lang2='pt')
        model = Transformer(src_vocab_size, tgt_vocab_size, d_model, MAX_LENGTH, dropout, num_layers, n_heads, PAD_TOKEN, SOS_TOKEN).to(device)
        translate_beam(model, dataloader, src_tokenizer, tgt_tokenizer, 10)
        
    if mode == "train":
        train_dataloader, val_dataloader, test_dataloader, src_vocab_size, tgt_vocab_size, src_tokenizer, tgt_tokenizer = read_opus(batch_size)
        model = Transformer(src_vocab_size, tgt_vocab_size, d_model, MAX_LENGTH, dropout, num_layers, n_heads, PAD_TOKEN, SOS_TOKEN).to(device)
        train(model, train_dataloader, val_dataloader, pre_train=True, tgt_vocab_size=tgt_vocab_size)

def main():
    run('inference')

if __name__ == '__main__':
    main()