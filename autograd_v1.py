import torch

# 建立 Tensor
a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.ones_like(a)   # 全 1，形狀與 a 相同

# 基本運算
c = a + b
d = a @ b.T  # 矩陣乘法

# GPU 加速
if torch.cuda.is_available():
    a = a.to("cuda")
    b = b.to("cuda")


x = torch.tensor([2.0], requires_grad=True)
y = x**2 + 3 * x
y.backward()
print(x.grad)  # dy/dx = 2x + 3


x = torch.tensor([5.0], requires_grad=True)
lr = 0.1

for i in range(20):
    y = x**2 - 4*x + 3
    y.backward()
    with torch.no_grad():
        x -= lr * x.grad
        x.grad.zero_()
    print(f"Step {i+1}: x={x.item():.4f}, y={y.item():.4f}")