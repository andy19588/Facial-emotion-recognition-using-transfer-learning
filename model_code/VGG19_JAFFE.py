import os
import torch
import numpy as np
from torch.amp import autocast, GradScaler
from torchvision.models import VGG19_Weights
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from collections import Counter
import matplotlib.pyplot as plt


# 檢查是否有 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

batch_size = 32
num_epochs = 100
val_ratio = 0.2
best_val_acc = 0.0
patience = 10
counter = 0

# 學習曲線記錄用
train_acc_list = []
val_acc_list = []
train_loss_list = []
val_loss_list = []

# 資料預處理 (保持不變)
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

# 載入資料 (保持不變)
full_dataset = datasets.ImageFolder('data_3/jaffe/train', transform=train_transform)
num_classes = len(full_dataset.classes)
train_size = int((1 - val_ratio) * len(full_dataset))
val_size = len(full_dataset) - train_size
generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
all_paths = [s[0] for s in full_dataset.samples]
train_indices = train_dataset.indices
val_indices = val_dataset.indices
train_paths = set([all_paths[i] for i in train_indices])
val_paths = set([all_paths[i] for i in val_indices])
overlap = train_paths.intersection(val_paths)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

class_names = full_dataset.classes

# 擷取 label
train_labels = [full_dataset.targets[i] for i in train_dataset.indices]
val_labels = [full_dataset.targets[i] for i in val_dataset.indices]

# 統計
train_counter = Counter(train_labels)
val_counter = Counter(val_labels)

# 轉成 list 並依照 label 排序
train_counts = [train_counter[i] for i in range(len(class_names))]
val_counts = [val_counter[i] for i in range(len(class_names))]

# 畫圖
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

# 載入模型 (保持不變)
model = models.vgg19(weights=VGG19_Weights.DEFAULT).to(device)
model.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Dropout(0.5),
    nn.Linear(4096, num_classes)
).to(device)

model.load_state_dict(torch.load("vgg19_emotion_RAF-DB.pth", map_location=device))

# 更細緻的解凍策略參數
unfreeze_start_epoch = 3
unfreeze_rate = 2


def get_layerwise_lr_params(model, base_lr, decay_factor=0.8):
    """
    為 features 的每層設定遞減學習率；classifier 則使用 base_lr。
    """
    params = []
    decay = 1.0
    for layer in model.features:
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.ReLU):
            layer_params = list(layer.parameters())
            if layer_params:
                for param in layer_params:
                    if param.requires_grad:
                        params.append({'params': param, 'lr': base_lr * decay})
            decay *= decay_factor  # 每經過一層就衰減
    # classifier 使用固定學習率
    for param in model.classifier.parameters():
        if param.requires_grad:
            params.append({'params': param, 'lr': base_lr})
    return params

# 初始化 optimizer 和學習率排程器
current_lr = 1e-4
optimizer = optim.Adam(get_layerwise_lr_params(model, base_lr=current_lr), weight_decay=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, threshold=1e-4, threshold_mode='rel')
scaler = GradScaler(device="cuda")
criterion = nn.CrossEntropyLoss()

# 訓練開始
for param in model.features.parameters():
    param.requires_grad = False

# 2. 訓練迴圈
for epoch in range(num_epochs):
    # 解凍邏輯
    if epoch >= unfreeze_start_epoch and (epoch - unfreeze_start_epoch) % unfreeze_rate == 0:
        num_unfrozen = sum(p.requires_grad for p in model.features.parameters())
        num_to_unfreeze = min(8, len(list(model.features.parameters())) - num_unfrozen)
        if num_to_unfreeze > 0:
            print(f">> 第 {epoch+1} 個 epoch，解凍 {num_to_unfreeze} 個 features 層參數")
            for param in list(model.features.parameters())[::-1]:
                if not param.requires_grad:
                    param.requires_grad = True
                    num_to_unfreeze -= 1
                    if num_to_unfreeze == 0:
                        break

        # 重新建 optimizer
        optimizer = optim.Adam(get_layerwise_lr_params(model, base_lr=current_lr),
                               weight_decay=1e-4)
        # 印出以檢查
        num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"目前可訓練參數總數：{num_trainable}")
        for i, group in enumerate(optimizer.param_groups):
            print(f"  Group {i}: lr = {group['lr']:.2e}")

    # === 訓練 ===
    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f"train_loader 批次數量: {len(train_loader)}")

    for i, (inputs, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", ncols=100)):
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

    train_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0
    train_acc = correct_train / total_train if total_train > 0 else 0
    train_acc_list.append(train_acc)
    train_loss_list.append(train_loss)

    print(f"[Epoch {epoch+1}/{num_epochs}] 訓練損失: {train_loss:.4f}, 訓練準確率: {train_acc:.4f}")

    # === 驗證 ===
    model.eval()
    correct = 0
    total = 0
    val_running_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_loss = val_running_loss / len(val_loader) if len(val_loader) > 0 else 0
    val_acc = correct / total if total > 0 else 0
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss)
    print(f"→ 驗證損失: {val_loss:.4f}, 驗證準確率: {val_acc:.4f}")
    print(f"train_loss_list 的長度: {len(train_loss_list)}") # 添加這行
    print(f"val_loss_list 的長度: {len(val_loss_list)}")   # 添加這行
    scheduler.step(val_acc)

    # Early stopping (保持不變)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        counter = 0
        torch.save(model.state_dict(), 'vgg19_emotion_JAFFE.pth')
        print("驗證準確率提升，已儲存模型")
    else:
        counter += 1
        print(f"驗證準確率未提升：{counter}/{patience}")
        if counter >= patience:
            print("提前停止訓練（Early Stopping）")
            break


epochs = range(1, len(train_acc_list) + 1)

print(f"\n📊 Epoch 數: {len(epochs)}")
print(f"📉 train_loss_list 長度: {len(train_loss_list)}")
print(f"📉 val_loss_list 長度: {len(val_loss_list)}")

if len(train_loss_list) != len(epochs) or len(val_loss_list) != len(epochs):
    print("⚠️ 損失列表長度與 Epoch 數不一致，跳過畫圖以避免錯誤")
else:
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_acc_list, 'r', label='train accuracy')
    plt.plot(epochs, val_acc_list, 'b', label='val accuracy')
    plt.title('train accuracy and val accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('accurzcy')
    plt.savefig("accuracy.png")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_loss_list, 'r', label='train loss')
    plt.plot(epochs, val_loss_list, 'b', label='val loss')
    plt.title('train loss and val loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig("loss.png")

    plt.tight_layout()
    plt.show()
