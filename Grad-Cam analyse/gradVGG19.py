import torch
import torch.nn.functional as F
from torchvision import models, transforms, datasets
from torchvision.models import VGG19_Weights
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


# 設定裝置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 載入模型架構與參數
model = models.vgg19(weights=VGG19_Weights.DEFAULT)
model.classifier[6] = torch.nn.Linear(4096, 7)  # 7 類對應 FER+ 訓練好的模型
model.load_state_dict(torch.load("resnet50_emotion_FERplus.pth", map_location=device))
model.to(device)
model.eval()

# 設定 GradCAM 用的 hook
features = None
gradients = None

def save_feature_hook(module, input, output):
    global features
    features = output

def save_gradient_hook(module, grad_in, grad_out):
    global gradients
    gradients = grad_out[0]

target_layer = model.features[-1]
target_layer.register_forward_hook(save_feature_hook)
target_layer.register_full_backward_hook(save_gradient_hook)

# 預處理
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# 載入測試集
test_dataset = datasets.ImageFolder('data_1/FERplus/test', transform=val_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)
classes = test_dataset.classes

# 測試集預測與真實標籤
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 評估
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='weighted')
precision = precision_score(all_labels, all_preds, average='weighted')
recall = recall_score(all_labels, all_preds, average='weighted')

print(f"\n✅ Test Accuracy: {acc:.4f}")
print(f"📊 Weighted Precision: {precision:.4f}")
print(f"📊 Weighted Recall:    {recall:.4f}")
print(f"📊 Weighted F1 Score:  {f1:.4f}")

# 顯示每一類的 precision/recall/f1
print("\n📄 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=classes))

# 混淆矩陣
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("gradcam_outputs/confusion_matrix.png")
plt.show()

# ----- Grad-CAM 視覺化（保留原本） -----
indices = random.sample(range(len(test_dataset)), 4)
for idx in indices:
    img_tensor, label = test_dataset[idx]
    input_tensor = img_tensor.unsqueeze(0).to(device)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    model.zero_grad()
    class_score = output[0, pred_class]
    class_score.backward()

    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    feature_map = features.squeeze().detach()
    for i in range(pooled_gradients.shape[0]):
        feature_map[i, :, :] *= pooled_gradients[i]

    heatmap = feature_map.mean(dim=0).cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    def denormalize(tensor):
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = std * img + mean
        img = np.clip(img, 0, 1)
        return img

    img = denormalize(img_tensor)
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = heatmap_colored / 255.0
    overlay = heatmap_colored * 0.5 + img * 0.5

    plt.figure(figsize=(6, 3))
    plt.suptitle(f"True: {classes[label]}, Pred: {classes[pred_class]}")
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(overlay)
    plt.title("Grad-CAM")
    plt.axis("off")

    os.makedirs("gradcam_outputs", exist_ok=True)
    plt.savefig(f"gradcam_outputs/sample_{idx}.png")
    plt.show()
