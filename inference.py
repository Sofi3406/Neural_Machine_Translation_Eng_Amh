import re
import pickle
from pathlib import Path

import torch
import torch.nn as nn

PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN = "<pad>", "<sos>", "<eos>", "<unk>"
_en_token_re = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[.,!?;:\"()\-]")


def tokenize_en(text):
    return _en_token_re.findall(text.lower())


class Vocab:
    def __init__(self):
        self.itos = []
        self.stoi = {}

    def encode(self, tokens):
        unk = self.stoi[UNK_TOKEN]
        return [self.stoi[SOS_TOKEN]] + [self.stoi.get(t, unk) for t in tokens] + [self.stoi[EOS_TOKEN]]

    def decode(self, ids, strip_special=True):
        toks = [self.itos[i] for i in ids]
        if strip_special:
            toks = [t for t in toks if t not in (PAD_TOKEN, SOS_TOKEN, EOS_TOKEN)]
        return toks

    def __len__(self):
        return len(self.itos)


class VocabUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "Vocab":
            return Vocab
        return super().find_class(module, name)


def load_vocab(path):
    with open(path, "rb") as file:
        return VocabUnpickler(file).load()


class AttnEncoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim, enc_hid_dim, n_layers, bidirectional=True,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_hidden = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.fc_cell = nn.Linear(enc_hid_dim * 2, dec_hid_dim)
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)

        def combine(state, fc):
            state = state.view(self.n_layers, 2, state.shape[1], -1)
            fwd, bwd = state[:, 0], state[:, 1]
            combined = torch.cat((fwd, bwd), dim=2)
            return torch.tanh(fc(combined))
        return outputs, combine(hidden, self.fc_hidden), combine(cell, self.fc_cell)


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear(enc_hid_dim * 2 + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)

    def forward(self, hidden_top, encoder_outputs):
        src_len = encoder_outputs.shape[0]
        hidden_rep = hidden_top.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden_rep, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return torch.softmax(attention, dim=1)


class AttnDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, n_layers, dropout, attention, pad_idx):
        super().__init__()
        self.output_dim = output_dim
        self.attention = attention
        self.embedding = nn.Embedding(output_dim, emb_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(emb_dim + enc_hid_dim * 2, dec_hid_dim, n_layers,
                            dropout=dropout if n_layers > 1 else 0)
        self.fc_out = nn.Linear(emb_dim + enc_hid_dim * 2 + dec_hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tok, hidden, cell, encoder_outputs):
        input_tok = input_tok.unsqueeze(0)
        embedded = self.dropout(self.embedding(input_tok))
        attn_weights = self.attention(hidden[-1], encoder_outputs)
        attn_weights_u = attn_weights.unsqueeze(1)
        enc_out_b = encoder_outputs.permute(1, 0, 2)
        context = torch.bmm(attn_weights_u, enc_out_b).permute(1, 0, 2)
        rnn_input = torch.cat((embedded, context), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        embedded, output, context = embedded.squeeze(0), output.squeeze(0), context.squeeze(0)
        prediction = self.fc_out(torch.cat((output, context, embedded), dim=1))
        return prediction, hidden, cell, attn_weights


class AttnSeq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder, self.decoder, self.device = encoder, decoder, device

    @torch.no_grad()
    def translate(self, src, trg_vocab, max_len=60):
        self.eval()
        encoder_outputs, hidden, cell = self.encoder(src)
        input_tok = torch.tensor([trg_vocab.stoi[SOS_TOKEN]], device=self.device)
        unk_id = trg_vocab.stoi[UNK_TOKEN]
        result_ids = []
        for _ in range(max_len):
            output, hidden, cell, _ = self.decoder(input_tok, hidden, cell, encoder_outputs)
            output[0, unk_id] = float("-inf")
            top1 = output.argmax(1)
            token_id = top1.item()
            if token_id == trg_vocab.stoi[EOS_TOKEN]:
                break
            result_ids.append(token_id)
            input_tok = top1
        return trg_vocab.decode(result_ids, strip_special=True)


class Translator:
    """Loads all artifacts once and exposes a simple translate(text) -> str method."""

    def __init__(self, artifact_dir="deployment_artifacts", device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        artifact_dir = Path(artifact_dir)

        self.src_vocab = load_vocab(artifact_dir / "src_vocab.pkl")
        self.trg_vocab = load_vocab(artifact_dir / "trg_vocab.pkl")
        with open(artifact_dir / "config.pkl", "rb") as f:
            cfg = pickle.load(f)

        pad_idx_src = self.src_vocab.stoi[PAD_TOKEN]
        pad_idx_trg = self.trg_vocab.stoi[PAD_TOKEN]

        attn = Attention(cfg["ENC_HID_DIM"], cfg["DEC_HID_DIM"])
        enc = AttnEncoder(len(self.src_vocab), cfg["EMB_DIM"], cfg["ENC_HID_DIM"],
                           cfg["DEC_HID_DIM"], cfg["N_LAYERS"], cfg["DROPOUT"], pad_idx_src)
        dec = AttnDecoder(len(self.trg_vocab), cfg["EMB_DIM"], cfg["ENC_HID_DIM"],
                           cfg["DEC_HID_DIM"], cfg["N_LAYERS"], cfg["DROPOUT"], attn, pad_idx_trg)
        self.model = AttnSeq2Seq(enc, dec, self.device).to(self.device)
        self.model.load_state_dict(torch.load(artifact_dir / "attention_model.pt", map_location=self.device))
        self.model.eval()

    def translate(self, text: str, max_len: int = 60) -> str:
        tokens = tokenize_en(text)
        src_ids = torch.tensor(self.src_vocab.encode(tokens), dtype=torch.long).unsqueeze(1).to(self.device)
        pred_tokens = self.model.translate(src_ids, self.trg_vocab, max_len=max_len)
        return " ".join(pred_tokens)
