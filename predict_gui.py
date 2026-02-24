import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox

# --- 1. 定义模型结构 (必须与训练时完全一致) ---
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

# --- 2. GUI 界面类 ---
class PredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.geometry("400x200")

        # 加载模型
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = NeuralNetwork().to(self.device)
        try:
            self.model.load_state_dict(torch.load('mnist_model.pth', map_location=self.device))
            self.model.eval()
        except FileNotFoundError:
            messagebox.showerror("错误", "找不到 mnist_model.pth，请先运行训练脚本！")

        self.btn_select = tk.Button(root, text="选择图片", command=self.select_and_predict, width=20, height=2)
        self.btn_select.pack(pady=10)

        self.label_result = tk.Label(root, text="预测结果: ---", font=("Arial", 16, "bold"))
        self.label_result.pack(pady=20)

    def select_and_predict(self):
        # 调起文件选择器
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        
        if not file_path:
            return

        # 2. 预处理并预测
        try:
            # MNIST 格式化处理
            img = Image.open(file_path).convert('L')
            img = ImageOps.invert(img) # 白纸黑字转黑底白字
            
            transform = transforms.Compose([
                transforms.Resize((28, 28)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
            
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(img_tensor)
                prob = torch.softmax(output, dim=1)
                confidence, predicted = torch.max(prob, 1)

            # 3. 更新界面结果
            res_text = f"预测结果: {predicted.item()}"
            self.label_result.config(text=res_text)
            
        except Exception as e:
            messagebox.showerror("预测失败", f"处理图片时出错: {e}")

# 启动程序
if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorApp(root)
    root.mainloop()