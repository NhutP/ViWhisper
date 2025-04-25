from transformers import WhisperProcessor
from transformers import WhisperForConditionalGeneration

from datasets import load_from_disk
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import random
import torch


class Classifier(nn.Module):
  def __init__(self, model):
    super(Classifier, self).__init__()
    self.encoder = model.get_encoder()
    self.encoder._freeze_parameters()

    self.flatten = nn.Flatten()
    self.ln1 = nn.Linear(576000, 10)
    self.ac1 = nn.ReLU()
    self.ln2 = nn.Linear(10 ,1)
     
    # self.softmax = nn.Softmax(dim=1)


  def forward(self, x):
    x = self.encoder(x).last_hidden_state
    x = self.flatten(x)
    x = self.ln1(x)
    x = self.ac1(x)
    x = self.ln2(x)
    x = torch.sigmoid(x)
    return x
  

class CustomDataset(Dataset):
  def __init__(self, dataset, processor):
    self.dataset = dataset
    self.processor = processor

  def __len__(self):
      return len(self.dataset)

  def __getitem__(self, idx):
    sample = self.dataset[idx]
    input_features = [{"input_features": sample["input_features"]}]
    array = self.processor.feature_extractor.pad(input_features, return_tensors="pt")['input_features'][0]

    label = float(sample['label'])
    return array, label


def train_model(model, train_loader, criterion, optimizer, scheduler, epochs=5):
  model.train()
  for epoch in range(epochs):
      running_loss = 0.0
      for arrays, labels in train_loader:
          arrays = arrays.float()
          labels = labels.float()
          labels = labels.unsqueeze(0)

          optimizer.zero_grad()
          outputs = model(arrays)
          loss = criterion(outputs, labels)
          loss.backward()
          optimizer.step()

          running_loss += loss.item()

      # Step the scheduler
      scheduler.step()

      print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")


dataset = load_from_disk(r"C:\Users\quang\Desktop\000000001")
dataset = dataset.add_column("label", [random.randint(0, 1) for _ in range(len(dataset))])
dataset.set_format("torch")


model_id = r"C:\Users\quang\Desktop\deply_checkpoint\tiny"
model = WhisperForConditionalGeneration.from_pretrained(model_id, use_cache=False)
processor = WhisperProcessor.from_pretrained("openai/whisper-" + 'tiny', task="transcribe")

train_dataset = CustomDataset(dataset, processor)


data_loader = DataLoader(train_dataset, batch_size=1)

gender_classifier = Classifier(model)
criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=1e-1)
scheduler = optim.lr_scheduler.LinearLR(optimizer)

# Train the model
train_model(gender_classifier, data_loader, criterion, optimizer, scheduler, epochs=10)