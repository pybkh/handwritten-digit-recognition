import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
import matplotlib.pyplot as plt

# 1. 必须重新定义模型结构（确保和训练时完全一致）
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.hidden1 = nn.Linear(28 * 28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.hidden2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.hidden3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.hidden4 = nn.Linear(128, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU()
        self.output = nn.Linear(32, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.bn1(self.hidden1(x)))
        x = self.relu(self.bn2(self.hidden2(x)))
        x = self.relu(self.bn3(self.hidden3(x)))
        x = self.relu(self.bn4(self.hidden4(x)))
        x = self.output(x)
        return x

def predict(image_path, model_path='mnist_model.pth'):
    # 配置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型参数
    model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval() # 切换到评估模式

    # 3. 处理你拍的照片
    # MNIST是黑底白字，如果你是白纸黑字拍的，必须经过以下处理：
    try:
        raw_img = Image.open(image_path).convert('L') # 转为灰度图
        # 如果背景是白的，字是黑的，需要反色
        # 你可以根据实际情况决定是否注释掉下面这一行
        raw_img = ImageOps.invert(raw_img) 
    except Exception as e:
        print(f"读取图片失败: {e}")
        return

    # 定义预处理流程
    transform = transforms.Compose([
        transforms.Resize((28, 28)),          # 强制缩放到 28x28
        transforms.ToTensor(),                # 转为张量
        transforms.Normalize((0.5,), (0.5,))  # 归一化（需与训练一致）
    ])

    img_tensor = transform(raw_img).unsqueeze(0).to(device) # 增加 Batch 维度

    # 4. 推理
    with torch.no_grad():
        output = model(img_tensor)
        probabilities = torch.softmax(output, dim=1) # 得到概率分布
        confidence, predicted = torch.max(probabilities, 1)

    # 5. 结果展示
    print(f"--- 预测结果 ---")
    print(f"图片路径: {image_path}")
    print(f"预测数字: {predicted.item()}")
    print(f"置信度: {confidence.item()*100:.2f}%")

    # 可视化看看模型眼里的图片长啥样
    plt.imshow(raw_img, cmap='gray')
    plt.title(f"Predicted: {predicted.item()} ({confidence.item()*100:.2f}%)")
    plt.show()

if __name__ == "__main__":
    # 在这里输入你照片的文件名，比如 'test.jpg'
    image_name = './images.png' 
    predict(image_name)