import torch
from transformers import BertTokenizer, BertForSequenceClassification, RobertaTokenizer, RobertaForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, Subset
from transformers import AdamW
import tqdm
import argparse
from sklearn.metrics import f1_score
import numpy as np
import os, sys

lib_path = os.path.abspath(os.path.join(r"C:\Users\hhjimhhj\Desktop\usc\ckids\GPTIndex\bert\bert.ipynb", '..', '..'))
sys.path.append(lib_path)

from generated_questions.question_utils import extract_q_a_pairs
from backend.config import QUESTION_ABS_PATH

def train(arg):
    # Load the tokenizer and BERT model
    if arg.model == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    elif arg.model == 'roberta':
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

    train_texts, train_labels = extract_q_a_pairs(os.path.join(QUESTION_ABS_PATH, 'qa_pairs_train.json'))
    val_texts, val_labels = extract_q_a_pairs(os.path.join(QUESTION_ABS_PATH, 'qa_pairs.json'))

    filter = lambda x: 1 if x == 'Yes' else 0
    train_labels = list(map(filter, train_labels))
    val_labels = list(map(filter, val_labels))

    train_encodings = tokenizer(train_texts, truncation=True, padding=True)
    val_encodings = tokenizer(val_texts, truncation=True, padding=True)

    print(train_encodings['input_ids'][0])
    print(val_encodings['input_ids'][0])
    print(type(train_encodings['input_ids']))

    print(train_encodings['attention_mask'][0])
    print(val_encodings['attention_mask'][0])

    train_encodings = {'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask'], 'labels': train_labels}
    val_encodings = {'input_ids': val_encodings['input_ids'], 'attention_mask': val_encodings['attention_mask'], 'labels': val_labels}

    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, encodings, r=None):
            self.encodings = {}
            if r:
                self.encodings['input_ids'] = torch.tensor(encodings['input_ids'])[r[0]:r[1]]
                self.encodings['attention_mask'] = torch.tensor(encodings['attention_mask'])[r[0]:r[1]]
                self.encodings['labels'] = torch.tensor(encodings['labels'])[r[0]:r[1]]
            else:
                self.encodings['input_ids'] = torch.tensor(encodings['input_ids'])
                self.encodings['attention_mask'] = torch.tensor(encodings['attention_mask'])
                self.encodings['labels'] = torch.tensor(encodings['labels'])

        def __len__(self):
            return self.encodings['input_ids'].shape[0]

        def __getitem__(self, i):
            return {key: tensor[i] for key, tensor in self.encodings.items()}
        
    train_dataset = CustomDataset(train_encodings, r=[0,1000])
    val_dataset = CustomDataset(val_encodings)

    train_loader = DataLoader(train_dataset, sampler=RandomSampler(train_dataset), batch_size=16)
    val_loader = DataLoader(val_dataset, sampler=SequentialSampler(val_dataset), batch_size=16)

    device = torch.device('cpu')

    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(50):
        for batch in tqdm.tqdm(train_loader, desc=f"Epoch {epoch+1}/{3}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            val_acc = 0
            batch_count = 0
            label_pred = np.array([])
            label_test = np.array([])
            for batch in val_loader:
                batch_count += 1
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                val_loss += outputs.loss
                val_acc += (outputs.logits.argmax(dim=-1) == labels).float().mean()
                label_pred = np.concatenate((label_pred, outputs.logits.argmax(dim=-1).numpy()))
                label_test = np.concatenate((label_test, labels.numpy()))
            val_loss /= batch_count
            val_acc /= batch_count

        print(f"Epoch {epoch+1}: Val loss {val_loss:.4f}, Val acc {val_acc:.4f}, f1 score {f1_score(label_test, label_pred):.4f}")

        model.save_pretrained(arg.model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', choices=['bert', 'roberta'], help='train which model')

    args = parser.parse_args()

    train(args)
    
