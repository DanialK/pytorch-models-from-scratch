import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embedding_size, num_heads):
        super(SelfAttention, self).__init__()
        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads

        assert (self.head_dim * num_heads == embedding_size, f"embedding_size must be divisible by num_heads")

        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.dense_out = nn.Linear(embedding_size, embedding_size)

    def forward(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]

        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.values(values)
        keys = self.keys(keys)
        queries = self.queries(queries)

        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e20"))

        # normalise across the source sentence (keys)
        # attention shape: n, heads, query_len, key_len
        attention = torch.softmax(energy / (self.embedding_size ** 0.5), dim=3)

        # keys and values are the same length, hence "l" is used to represent the common dim size
        out = torch.einsum("nhql,nlhd->nqhd", [attention, values])

        # concatenation/flattening step
        out = out.reshape(N, query_len, self.num_heads * self.head_dim)

        out = self.dense_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()

        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_size, forward_expansion * embedding_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embedding_size, embedding_size)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embedding_size, num_layers, num_heads, device, forward_expansion, dropout, max_length):
        super(Encoder, self).__init__()
        self.embedding_size = embedding_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embedding_size, num_heads, dropout, forward_expansion)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout(self.word_embedding(x) + self.position_embedding(positions))

        for layer in self.layers:
            # in encoder, value, key and query are all the same
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embedding_size, num_heads, device, forward_expansion, dropout):
        super(DecoderBlock, self).__init__()
        self.attention = SelfAttention(embedding_size, num_heads)
        self.norm = nn.LayerNorm(embedding_size)
        self.transformer_block = TransformerBlock(
            embedding_size, num_heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    # value and key are going to be the same thing which are the outputs from the encoder
    def forward(self, x, value, key, src_mask, trg_mask):
        attention = self.attention(x, x, x, trg_mask)
        query = self.dropout(self.norm(attention + x))
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(self, trg_vocab_size, embedding_size, num_layers, num_heads, device, forward_expansion, dropout, max_length):
        super(Decoder, self).__init__()
        self.device = device
        self.word_embedding = nn.Embedding(trg_vocab_size, embedding_size)
        self.position_embedding = nn.Embedding(max_length, embedding_size)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embedding_size, num_heads, device, forward_expansion, dropout)
                for _ in range(num_layers)
            ]
        )

        self.fc_out = nn.Linear(embedding_size, trg_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        x = self.dropout((self.word_embedding(x) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, encoder_out, encoder_out, src_mask, trg_mask)

        out = self.fc_out(x)

        return out


class Transformer(nn.Module):
    def __init__(self,
                 src_vocab_size,
                 trg_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 embedding_size=256,
                 num_layers=6,
                 forward_expansion=4,
                 num_heads=8,
                 dropout=0,
                 device="cuda",
                 max_length=100
                 ):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, embedding_size, num_layers, num_heads, device, forward_expansion, dropout, max_length)
        self.decoder = Decoder(trg_vocab_size, embedding_size, num_layers, num_heads, device, forward_expansion, dropout, max_length)
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # (N, 1, 1, src_len)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(N, 1, trg_len, trg_len)
        return trg_mask.to(self.device)

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_src, src_mask, trg_mask)
        return out


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    x = torch.tensor([[1, 5, 6, 4, 3, 9, 5, 2, 0], [1, 8, 7, 3, 4, 5, 6, 7, 2]]).to(
        device
    )
    trg = torch.tensor([[1, 7, 4, 3, 5, 9, 2, 0], [1, 5, 6, 2, 4, 7, 6, 2]]).to(device)

    src_pad_idx = 0
    trg_pad_idx = 0
    src_vocab_size = 10
    trg_vocab_size = 10
    model = Transformer(src_vocab_size, trg_vocab_size, src_pad_idx, trg_pad_idx, device=device).to(
        device
    )
    out = model(x, trg[:, :-1])
    print(out.shape)