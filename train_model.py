import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from tqdm import tqdm

# 1. Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Load dataset
df = pd.read_csv("manipulative_nonmanipulative_allproducts_1000_unique.csv")  # Replace with your dataset filename
print(f"Dataset Shape: {df.shape}")

# 3. Encode labels
label_mapping = {"Manipulative": 0, "Non-Manipulative": 1}
df["label"] = df["label"].map(label_mapping)

# 4. Train-test split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Training Set Size: {train_df.shape}, Testing Set Size: {test_df.shape}")

# 5. Save test set for evaluation
test_df.to_csv("test_data.csv", index=False)

# 6. Dataset Class
class ProductDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# 7. Initialize tokenizer and model
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)
model.to(device)

# 8. DataLoaders
train_dataset = ProductDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

test_dataset = ProductDataset(test_df["text"].tolist(), test_df["label"].tolist(), tokenizer)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# 9. Optimizer & Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
total_steps = len(train_loader) * 3
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# 10. Training Loop
for epoch in range(3):
    model.train()
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/3")

    for batch in progress_bar:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1} completed. Average Loss: {avg_loss:.4f}")

# 11. Evaluation Metrics
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")

#  12. Safe Model Saving
save_dir = 'distilbert_manipulation_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

try:
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print("Model saved successfully using save_pretrained.")
except Exception as e:
    print(f" save_pretrained failed due to {e}. Trying torch.save as fallback...")
    torch.save(model.state_dict(), os.path.join(save_dir, 'pytorch_model.bin'))
    tokenizer.save_pretrained(save_dir)
    print(" Model saved successfully using torch.save fallback.")
