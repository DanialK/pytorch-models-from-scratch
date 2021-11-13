import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchtext.datasets import Multi30k
import random
from torchtext.data.utils import get_tokenizer
from collections import Counter, OrderedDict
from torchtext.vocab import vocab
from torch.nn.utils.rnn import pad_sequence
import math
import time


de_tokenizer = get_tokenizer('spacy', 'de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', 'en_core_web_sm')


def build_vocab(data, src_tokenizer, target_tokenizer, specials=['<unk>', '<SOS>', '<EOS>', '<PAD>'], special_first=True, min_freq=2):
    src_counter = Counter()
    target_counter = Counter()

    for (src, target) in data:
        src_counter.update(src_tokenizer(src))
        target_counter.update(target_tokenizer(target))

    def get_vocab(counter, specials):
        if specials is not None:
            for tok in specials:
                del src_counter[tok]
                del target_counter[tok]

        sorted_by_freq_tuples = sorted(counter.items(), key=lambda x: x[0])
        sorted_by_freq_tuples.sort(key=lambda x: x[1], reverse=True)
        ordered_dict = OrderedDict(sorted_by_freq_tuples)

        if specials is not None:
            if special_first:
                specials = specials[::-1]
            for symbol in specials:
                ordered_dict.update({symbol: min_freq})
                ordered_dict.move_to_end(symbol, last=not special_first)
        result = vocab(ordered_dict, min_freq=min_freq)
        result.set_default_index(result['<unk>'])
        return result

    src_vocab = get_vocab(src_counter, specials)
    target_vocab = get_vocab(target_counter, specials)

    return src_vocab, target_vocab


train_data = Multi30k(split='train', language_pair=('de', 'en'))

de_vocab, en_vocab = build_vocab(train_data, de_tokenizer, en_tokenizer)


class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)

    def forward(self, x):
        # x.shape = (seq_length, N)

        embedding = self.dropout(self.embedding(x))  # embedding.shape = (seq_length, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, dropout):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size, num_layers, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, hidden, cell):
        # shape of x: (N) but we want (1, N) since prediction happens one word at the time
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))  # embedding.shape = (1, N, embedding_size)

        outputs, (hidden, cell) = self.rnn(embedding, (hidden, cell))  # shape of output: (1, N, hidden_size)

        predictions = self.fc(outputs)  # (1, N, length of vocab)

        predictions = predictions.squeeze(0)  # (N, length of vocab)

        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(en_vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(self.device)

        hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(x, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


def train(model: nn.Module,
          iterator: torch.utils.data.DataLoader,
          optimizer: optim.Optimizer,
          criterion: nn.Module,
          clip: float):

    model.train()

    epoch_loss = 0
    N = 0
    for (src, trg) in list(iterator):
        src, trg = src.to(device), trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        output = output[1:].view(-1, output.shape[-1])
        trg = trg[1:].view(-1)

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()
        N += 1

    return epoch_loss / N


def evaluate(model: nn.Module,
             iterator: torch.utils.data.DataLoader,
             criterion: nn.Module):

    model.eval()

    epoch_loss = 0

    with torch.no_grad():

        for _, (src, trg) in enumerate(iterator):
            src, trg = src.to(device), trg.to(device)

            output = model(src, trg, 0) #turn off teacher forcing

            output = output[1:].view(-1, output.shape[-1])
            trg = trg[1:].view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time: int,
               end_time: int):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":

    num_epochs = 20
    learning_rate = 0.001
    batch_size = 64

    load_model = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_size_encoder = len(de_vocab)
    input_size_decoder = len(en_vocab)
    output_size = len(en_vocab)
    encoder_embedding_size = 300
    decoder_embedding_size = 300
    hidden_size = 1024
    num_layers = 2
    enc_dropout = 0.5
    dec_dropout = 0.5
    clip = 1
    PAD_IDX = de_vocab['<PAD>']
    BOS_IDX = de_vocab['<SOS>']
    EOS_IDX = de_vocab['<EOS>']

    def generate_batch(data_batch):
        de_batch, en_batch = [], []
        for (raw_de, raw_en) in data_batch:
            de_item = torch.tensor([de_vocab[token] for token in de_tokenizer(raw_de)], dtype=torch.long)
            en_item = torch.tensor([en_vocab[token] for token in en_tokenizer(raw_en)], dtype=torch.long)
            de_batch.append(torch.cat([torch.tensor([BOS_IDX]), de_item, torch.tensor([EOS_IDX])], dim=0))
            en_batch.append(torch.cat([torch.tensor([BOS_IDX]), en_item, torch.tensor([EOS_IDX])], dim=0))
        de_batch = pad_sequence(de_batch, padding_value=PAD_IDX)
        en_batch = pad_sequence(en_batch, padding_value=PAD_IDX)
        return de_batch, en_batch

    encoder_net = Encoder(input_size_encoder, encoder_embedding_size, hidden_size, num_layers, enc_dropout).to(device)
    decoder_net = Decoder(input_size_decoder, decoder_embedding_size, hidden_size, output_size, num_layers, dec_dropout).to(device)
    model = Seq2Seq(encoder_net, decoder_net, device).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    pad_idx = en_vocab["<PAD>"]
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(num_epochs):

        start_time = time.time()

        train_data, val_data = Multi30k(split=('train', 'valid'), language_pair=('de', 'en'))
        train_iter = DataLoader(train_data, batch_size=batch_size, collate_fn=generate_batch)
        valid_iter = DataLoader(val_data, batch_size=batch_size, collate_fn=generate_batch)

        train_loss = train(model, train_iter, optimizer, criterion, clip)
        valid_loss = evaluate(model, valid_iter, criterion)

        end_time = time.time()

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

    test_data = Multi30k(split='test', language_pair=('de', 'en'))
    test_iter = DataLoader(train_data, batch_size=batch_size, collate_fn=generate_batch)
    test_loss = evaluate(model, test_iter, criterion)

    print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')