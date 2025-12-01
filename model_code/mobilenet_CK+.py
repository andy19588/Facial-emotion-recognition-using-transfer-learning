import os
import torch
import numpy as np
from torch.amp import autocast, GradScaler
from torchvision.models import mobilenet_v2
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter

# ======= 裝置與參數 =======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
num_epochs = 100
val_ratio = 0.2
best_val_acc = 0.0
patience = 10
counter = 0

train_acc_list, val_acc_list = [], []
train_loss_list, val_loss_list = [], []

# ======= 資料前處理 =======
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ======= 載入資料集 =======
full_dataset = datasets.ImageFolder('data_4/CK+/train', transform=train_transform)
num_classes = len(full_dataset.classes)
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

class_names = full_dataset.classes

# ======= 標籤分布圖 =======
train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
val_labels = [full_dataset.targets[i] for i in val_dataset.indices]
train_counter = Counter(train_labels)
val_counter = Counter(val_labels)
train_counts = [train_counter[i] for i in range(len(class_names))]
val_counts = [val_counter[i] for i in range(len(class_names))]

x = np.arange(len(class_names))
width = 0.35
plt.figure(figsize=(10, 5))
plt.bar(x - width/2, train_counts, width, label='Train')
plt.bar(x + width/2, val_counts, width, label='Validation')
plt.xticks(x, class_names, rotation=45)
plt.xlabel("Emotion Class")
plt.ylabel("Count")
plt.title("Train vs Validation Label Distribution")
plt.legend()
plt.tight_layout()
plt.savefig("label_distribution.png")
plt.show()

# ======= 模型載入與遷移學習設定 =======
# 初始化模型（不使用預訓練）
model = mobilenet_v2(weights=None).to(device)

# 修改分類層
in_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(in_features, num_classes).to(device)

# 載入你訓練好的模型權重（略過分類層）
state_dict = torch.load("mobilenetv2_emotion_JAFFE.pth", map_location=device)
state_dict.pop("classifier.1.weight", None)
state_dict.pop("classifier.1.bias", None)
model.load_state_dict(state_dict, strict=False)
print("✅ 已載入自訓模型權重（略過分類層）")

# 凍結參數，只訓練分類層
for param in model.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# 動態解凍設定
unfreeze_start_epoch = 3
unfreeze_rate = 2
mobilenet_blocks = list(model.features)

def get_layerwise_lr_params(model, base_lr, decay_factor=0.8):
    params = []
    decay = 1.0
    for layer in reversed(mobilenet_blocks):
        for param in layer.parameters():
            if param.requires_grad:
                params.append({'params': param, 'lr': base_lr * decay})
        decay *= decay_factor
    for param in model.classifier.parameters():
        if param.requires_grad:
            params.append({'params': param, 'lr': base_lr})
    return params

# ======= Optimizer, Scheduler, Loss =======
current_lr = 1e-4
optimizer = optim.Adam(get_layerwise_lr_params(model, base_lr=current_lr), weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-4)
scaler = GradScaler(device="cuda")
criterion = nn.CrossEntropyLoss()

# ======= 訓練迴圈 =======
for epoch in range(num_epochs):
    # 動態解凍 block
    if epoch >= unfreeze_start_epoch and (epoch - unfreeze_start_epoch) % unfreeze_rate == 0:
        for block in mobilenet_blocks:
            if all(not param.requires_grad for param in block.parameters()):
                for param in block.parameters():
                    param.requires_grad = True
                print(f">> 第 {epoch+1} 個 epoch，解凍 block：{block}")
                break
        optimizer = optim.Adam(get_layerwise_lr_params(model, base_lr=current_lr), weight_decay=1e-4)

    # 訓練模式
    model.train()
    running_loss, correct_train, total_train = 0.0, 0, 0

    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        with autocast("cuda" if torch.cuda.is_available() else "cpu"):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct_train += (preds == labels).sum().item()
        total_train += labels.size(0)

    train_loss = running_loss / len(train_loader)
    train_acc = correct_train / total_train
    train_loss_list.append(train_loss)
    train_acc_list.append(train_acc)
    print(f"[Epoch {epoch+1}] 訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")

    # 驗證
    model.eval()
    val_running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_loss = val_running_loss / len(val_loader)
    val_acc = correct / total
    val_loss_list.append(val_loss)
    val_acc_list.append(val_acc)
    print(f"→ 驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
    scheduler.step(val_acc)

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'mobilenetv2_emotion_CK+.pth')
        print("✅ 驗證準確率提升，已儲存模型")
    else:
        counter += 1
        print(f"驗證準確率未提升：{counter}/{patience}")
        if counter >= patience:
            print("⏹️ 提前停止訓練（Early Stopping）")
            break

# ======= 畫圖 =======
epochs = range(1, len(train_acc_list) + 1)
if len(train_loss_list) != len(epochs) or len(val_loss_list) != len(epochs):
    print("⚠️ 損失列表長度與 Epoch 數不一致，跳過畫圖")
else:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_list, 'r', label='train accuracy')
    plt.plot(epochs, val_acc_list, 'b', label='val accuracy')
    plt.title('Train vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig("accuracy.png")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_list, 'r', label='train loss')
    plt.plot(epochs, val_loss_list, 'b', label='val loss')
    plt.title('Train vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss.png")

    plt.tight_layout()
    plt.show()
