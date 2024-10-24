import torch
import torch.nn as nn
import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout):
        super(TransformerEmbedding, self).__init__()
        self.max_len = max_len 
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        b, t = x.size()
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(0, t, dtype=torch.long, device=device))
        return self.dropout(tok_emb + pos_emb)

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.single_head_dim = int(d_model / self.n_heads)
        self.query = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.key = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.value = nn.Linear(self.single_head_dim, self.single_head_dim, bias=False)
        self.w_o = nn.Linear(self.n_heads * self.single_head_dim, d_model) # (d_model, d_model)
    
    def forward(self, q, k, v, mask=None):#(batchsize,seq_len,d_model)
        batch_size, seq_length, d_model = k.shape

        query_seq_length = q.size(1)
        key = k.view(-1, seq_length, self.n_heads, self.single_head_dim)
        query = q.view(-1, query_seq_length, self.n_heads, self.single_head_dim)
        value = v.view(-1, seq_length, self.n_heads, self.single_head_dim)

        key = self.key(key)
        query = self.query(query)
        value = self.value(value)

        k = key.transpose(1, 2)
        q = query.transpose(1, 2)
        v = value.transpose(1, 2)

        k_t = k.transpose(-1, -2)
        score = (q @ k_t) * d_model ** -0.5
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-1e20'))
        score = F.softmax(score, dim=-1) 
        out = score @ v # (batch_size, head, length, single_head_dim)
        b, h, s, h_d = out.shape
        out = out.transpose(1, 2).contiguous().view(b, s, h*h_d)
        out = self.w_o(out)

        return out

class MLP(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4*d_model),
            nn.ReLU(),
            nn.Linear(4*d_model, d_model),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout)
    def forward(self, x, src_mask=None):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), mask=src_mask) # Residual connection
        x = x + self.mlp(self.ln2(x))
        return x
    
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.cross_attn = MultiHeadSelfAttention(d_model, n_heads)
        self.ln3 = nn.LayerNorm(d_model)
        self.ff = MLP(d_model, dropout)
    def forward(self, dec_input, enc_out, tgt_mask, src_mask):
        _x = dec_input
        x = self.ln1(dec_input)
        x = _x + self.attn(q=x, k=x, v=x, mask=tgt_mask)

        if enc_out is not None:
            _x = x
            x = self.ln2(x)
            enc_out = self.ln2(enc_out)
            x = _x + self.cross_attn(q=x, k=enc_out, v=enc_out, mask=src_mask)
        
        x = x + self.ff(self.ln3(x))

        return x

class Encoder(nn.Module):
    def __init__(self, src_vocab_size, d_model, seq_len, dropout, n_layers, n_heads):
        super().__init__()
        self.emb = TransformerEmbedding(src_vocab_size, d_model, seq_len, dropout)
        self.layers = nn.ModuleList([EncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
    def forward(self, x, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
    
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, d_model, seq_len, dropout, n_layers, n_heads):
        super().__init__()
        self.emb = TransformerEmbedding(tgt_vocab_size, d_model, seq_len, dropout)
        self.layers = nn.ModuleList([DecoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.linear = nn.Linear(d_model, tgt_vocab_size)
    
    def forward(self, x, enc_out, tgt_mask, src_mask):
        x = self.emb(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        out = self.linear(x)
        return out
    
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model, seq_len, dropout, n_layers, n_heads, pad_idx, tgt_sos_idx):
        super().__init__()
        self.pad_idx = pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.encoder = Encoder(src_vocab_size, d_model, seq_len, dropout, n_layers, n_heads)
        self.decoder = Decoder(tgt_vocab_size, d_model, seq_len, dropout, n_layers, n_heads)
    
    def forward(self, src, tgt):
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)
        enc_src = self.encoder(src, src_mask)
        output = self.decoder(tgt, enc_src, tgt_mask, src_mask)
        return output

    def make_src_mask(self, src):
        src_mask = (src != self.pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask
    
    def make_tgt_mask(self, tgt): # (batch, seq_len)
        tgt_pad_mask = (tgt != self.pad_idx).unsqueeze(1).unsqueeze(2)
        seq_len = tgt.size(1)
        tgt_tril = torch.tril(torch.ones(seq_len, seq_len)).type(torch.ByteTensor).to(device)
        return tgt_pad_mask & tgt_tril


