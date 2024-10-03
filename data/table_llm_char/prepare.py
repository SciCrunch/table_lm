"""
Prepare the PMC OAI 2024 June table dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import argparse
import pickle
import numpy as np
from pathlib import Path
from datasets import load_dataset
from tqdm import tqdm


class Vocabulary:
    def __init__(self, meta_pk_path: Path):
        with open(meta_pk_path, 'rb') as f:
            meta = pickle.load(f)
        self.vocab_size = meta['vocab_size']
        self.stoi = meta['stoi']
        self.itos = meta['itos']

    def encode(self, s: str):
        return [self._encode_char(c) for c in s]

    def _encode_char(self, c):
        return self.stoi[c] if c in self.stoi else self.stoi['<UNK>']

    def decode(self, l: list[int]):
        return [self.itos[i] for i in l]

    def get_eos_encoded(self):
        return self.stoi['<EOS>']


def prep_vocab(pretrain_text_corpus_path: Path, out_dir: Path):
    chars_set = set()
    with open(pretrain_text_corpus_path) as f:
        for line in f:
            nc_list = [c for c in line.rstrip() if c not in chars_set]
            if len(nc_list) > 0:
                chars_set.update(nc_list)

    chars = sorted(list(chars_set))
    chars.append('<EOS>')
    chars.append('<UNK>')
    vocab_size = len(chars)
    print("all the unique characters:", ''.join(chars))
    print(f"vocab size: {vocab_size:,}")
    out_dir.mkdir(parents=True, exist_ok=True)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    meta = {
        'vocab_size': vocab_size,
        'itos': itos,
        'stoi': stoi,
    }
    meta_file_path = out_dir / "meta.pkl"
    with open(meta_file_path, 'wb') as f:
        pickle.dump(meta, f)
        print(f"wrote {meta_file_path}")


def prep_train_val_sets(pretrain_text_corpus_dir: Path, out_dir: Path):
    meta_file_path = out_dir / "meta.pkl"
    assert meta_file_path.exists()
    vocab = Vocabulary(meta_file_path)
    data_files = {"train": "train.txt", "val": "val.txt"}
    dataset = load_dataset("text", data_dir=str(pretrain_text_corpus_dir), data_files=data_files)
    # print(len(dataset["train"]))

    def process(example):
        ids = vocab.encode(example['text'])
        ids.append(vocab.get_eos_encoded())
        return {'ids': ids, 'len': len(ids)}

    tokenized = dataset.map(process, remove_columns=['text'],
                            desc="tokenizing corpus",
                            num_proc=2)
    # print(len(tokenized['train']))

    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        out_path = out_dir / f'{split}.bin'
        dtype = np.uint16  # (can do since enc.max_token_value == 11,947 is < 2**16)
        arr = np.memmap(str(out_path), dtype=dtype, mode="w+", shape=(arr_len,))
        total_batches = 1024
        if arr_len < 1024:
            total_batches = 1
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {out_path}"):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True)
            arr_batch = np.concatenate(batch['ids'])
            # write into mmap
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="encodes Tables text data for pretraining")
    parser.add_argument("-c", metavar="<command (one of [vocab, encode])>", required=True)
    parser.add_argument("-i", metavar="<input-text-corpus-file>", required=True)
    parser.add_argument("-od", metavar="<output-dir>", required=True)

    args = parser.parse_args()

    cmd = args.c
    corpus_text_file = Path(args.i)
    output_dir = Path(args.od)

    if cmd == 'vocab':
        prep_vocab(corpus_text_file, output_dir)
    elif cmd == 'encode':
        prep_train_val_sets(corpus_text_file.parent, output_dir)
        # print('Not implemented yet!')
