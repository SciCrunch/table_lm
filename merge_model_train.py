import os
import json
import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from pathlib import Path

from model import GPTConfig
from table_cell_merge_model import TableCellMergeModel
from data.table_llm_char.prepare import Vocabulary

batch_size = 64


class CellMergeDataset(Dataset):
    def __init__(self, instances, vocab: Vocabulary):
        self.vocab = vocab
        n_instances = len(instances)
        xs = np.zeros((n_instances, 256), dtype=np.int32)
        ys = [int(inst['label']) for inst in instances]
        print(f"ys: {len(ys)}")
        for i, inst in enumerate(instances):
            el = []
            el.extend(vocab.encode(inst['l1']))
            el.append(vocab.get_eos_encoded())
            el.extend(vocab.encode(inst['l2']))
            xs[i, :len(el)] = el
        self.X = torch.tensor(xs, dtype=torch.int32)
        ys = np.asarray(ys, dtype=np.float32)
        self.y = torch.tensor(ys, dtype=torch.float32).reshape(-1, 1)
        print(f"X: {self.X.shape}")
        print(f"y: {self.y.shape}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        features = self.X[idx]
        target = self.y[idx]
        return features, target


def create_model(char_llm_model_path, vocab: Vocabulary, device):
    model_args = dict(n_layer=6, n_head=6, n_embd=384, block_size=256,
                      bias=False, vocab_size=vocab.vocab_size, dropout=0.1)
    gpt_conf = GPTConfig(**model_args)
    checkpoint = torch.load(char_llm_model_path)
    state_dict = checkpoint['model']
    model = TableCellMergeModel(gpt_conf, state_dict)
    model = model.to(device)
    return model


def model_train(model, loss_fn, train_dataloader, device):
    size = len(train_dataloader.dataset)
    optimizer = optim.Adam(model.parameters(), lr=0.00002)
    n_epochs = 2
    model.train()
    for epoch in range(n_epochs):
        print(f"Epoch {epoch + 1}\n-------------------------------")
        for batch, (X, y) in enumerate(train_dataloader):
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # backprop
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if batch % 100 == 0:
                loss, current = loss.item(), batch * batch_size + len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def model_eval(model, loss_fn, test_dataloader, device):
    model.eval()
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in test_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.round() == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


def train_test(model, train_dataset, test_dataset, device, model_output_path: Path):
    loss_fn = torch.nn.BCELoss()
    tr_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    tst_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    model_train(model, loss_fn, tr_dataloader, device)
    model_eval(model, loss_fn, tst_dataloader, device)
    torch.save(model.state_dict(), model_output_path)


def load_instances(json_path: Path):
    with open(json_path) as f:
        data = json.load(f)
        return data['instances']


def cli():
    parser = argparse.ArgumentParser(description="training row merging model")
    parser.add_argument("-d", metavar="<data-dir>", required=True)
    parser.add_argument("-p", metavar="<table-lm-checkpoint-file>", required=True)
    parser.add_argument("-v", metavar="<vocabulary-meta-file>", required=True)
    parser.add_argument("-o", metavar="<model-output-dir>", required=True)
    args = parser.parse_args()
    llm_cpkt_path = args.p
    data_dir = args.d
    meta_path = args.v
    model_out_path = args.o
    vocabulary = Vocabulary(Path(meta_path))
    print(f"vocab size: {vocabulary.vocab_size}")
    tr_instances = load_instances(data_dir / "cell_merge_tr_instances.json")
    tr_dataset = CellMergeDataset(tr_instances, vocabulary)
    print(f"training dataset size:{len(tr_dataset)}")
    tr_instances = None
    tst_instances = load_instances(data_dir / "cell_merge_test_instances.json")
    tst_dataset = CellMergeDataset(tst_instances, vocabulary)
    tst_instances = None
    print(f"test dataset size:{len(tst_dataset)}")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_model = create_model(llm_cpkt_path, vocabulary, _device)
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    train_test(cm_model, tr_dataset, tst_dataset, _device, model_out_path)



def test_driver():
    HOME = os.path.expanduser('~')
    meta_path = "data/table_llm_char/meta.pkl"
    vocabulary = Vocabulary(Path(meta_path))
    print(f"vocab size: {vocabulary.vocab_size}")
    data_dir = Path(HOME, "data/table_llm/cell_merge")
    tr_instances = load_instances(data_dir / "cell_merge_tr_instances.json")
    tr_dataset = CellMergeDataset(tr_instances, vocabulary)
    print(f"training dataset size:{len(tr_dataset)}")
    tr_instances = None
    tst_instances = load_instances(data_dir / "cell_merge_test_instances.json")
    tst_dataset = CellMergeDataset(tst_instances, vocabulary)
    tst_instances = None
    print(f"test dataset size:{len(tst_dataset)}")

    llm_cpkt_path = Path("out/ckpt.pt")
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cm_model = create_model(llm_cpkt_path, vocabulary, _device)
    model_out_path = Path(HOME, "models/tc_merge/model.pth")
    model_out_path.parent.mkdir(parents=True, exist_ok=True)
    train_test(cm_model, tr_dataset, tst_dataset, _device, model_out_path)



if __name__ == '__main__':
    cli()
