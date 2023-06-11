import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

from tqdm import tqdm

import os

from include.data_loader import read_txt_data, read_file


device = "cpu"

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=35, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)


model = torch.load("weights/model_state_dict_8000.pth", torch.device(device))
model.to(device)
model.eval()


data = read_txt_data("data/Fylhtq/")
for key in data:
    data[key] = torch.from_numpy(data[key]).unsqueeze(0)

commands = data.keys()
labels = sorted(['backward', 'forward', 'left', 'off', 'on', 'right', 'stop'])


def get_likely_index(tensor):
    # find most likely label index for each element in the batch
    return tensor.argmax(dim=-1)


def index_to_label(index):
    # Return the word corresponding to the index in labels
    # This is the inverse of label_to_index
    return labels[index]


def predict(tensor):
    # Use the model to predict the label of the waveform
    tensor = tensor.to(device)
    tensor = transform(tensor)
    tensor = tensor.unsqueeze(0)
    tensor = model(tensor)
    tensor = get_likely_index(tensor)
    tensor = index_to_label(tensor.squeeze())
    return tensor


# command = read_file("/home/cailen/Data/commands_2770/left.txt")
# command = torch.from_numpy(command).unsqueeze(0)

# base_sample_rate = command.shape[-1]
# new_sample_rate = 8000
# transform = torchaudio.transforms.Resample(orig_freq=base_sample_rate, new_freq=new_sample_rate)

# pred = predict(command)
# print(pred)


print(model)
# print(sum(p.numel() for p in model.parameters() if p.requires_grad))
for key in data:
    base_sample_rate = data[key].shape[-1]
    new_sample_rate = 8000
    transform = torchaudio.transforms.Resample(orig_freq=base_sample_rate, new_freq=new_sample_rate)
    pred = predict(data[key])
    print(pred)