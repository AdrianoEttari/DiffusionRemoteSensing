import os
import torch
from torchvision import transforms
import torchvision
from torch.utils.data import DataLoader

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

import torch.nn as nn

batch_size=32

init_process_group(backend="nccl")

transform = transforms.Compose(
[transforms.ToTensor()])

train_dataset = torchvision.datasets.CIFAR10(root='./Cifar10', train=True,
                                    download=True, transform=transform)

image_size = 32
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, pin_memory=True, shuffle=False, sampler=DistributedSampler(train_dataset))
num_classes = len(train_loader.dataset.classes)

gpu_id = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(int(gpu_id))
device = gpu_id

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.fc2 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
model = SimpleCNN().to(device)
print("Num params: ", sum(p.numel() for p in model.parameters()))
model = DDP(model, device_ids=[device], find_unused_parameters=True)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(10):
    running_loss = 0.0
    train_loader.sampler.set_epoch(epoch)
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if i % 100 == 99:    # stampa ogni 100 batch
            print(f'[Epoch {epoch + 1}, Batch {i + 1}] loss: {running_loss / 100:.3f}, gpu_id: {device}')
            running_loss = 0.0
        


destroy_process_group()
