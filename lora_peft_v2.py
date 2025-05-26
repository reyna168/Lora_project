import torch
import torch.nn as nn
from torch.optim import Adam
from copy import deepcopy
from peft import get_peft_model, LoraConfig, PeftModel

# 固定随机数种子，确保结果可复现
torch.manual_seed(42)

# 定义一个简单的线性模型
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, x):
        return self.linear(x)

# 实例化线性模型
model = LinearModel(input_size=10, output_size=1)

# 在应用 LoRA 之前深拷贝原始模型，确保后续公平比较
original_model = deepcopy(model)

# 配置 LoRA 参数
config = LoraConfig(
    inference_mode=False,
    r=4,
    lora_alpha=16,
    target_modules=['linear'],
)

# 将 LoRA 应用到模型中
lora_model = get_peft_model(model, config)

# 定义一个简单的损失函数和优化器
criterion = nn.MSELoss()
optimizer = Adam(lora_model.parameters(), lr=1e-3)

# 生成一些模拟的训练数据
input_data = torch.randn(100, 10)  # 100 个样本，每个样本有 10 个特征
target_data = torch.randn(100, 1)  # 对应的目标值

# 训练一个回合
lora_model.train()
for epoch in range(1):  # 训练 1 个回合
    optimizer.zero_grad()
    outputs = lora_model(input_data)
    loss = criterion(outputs, target_data)
    loss.backward()
    optimizer.step()

# 训练后保存 LoRA 权重
lora_model.save_pretrained('linear_lora_model')

# 方法 1：先使用 get_peft_model，再加载 LoRA 权重
model1 = PeftModel.from_pretrained(get_peft_model(deepcopy(original_model), config), 'linear_lora_model')

# 方法 2：直接加载 LoRA 权重
model2 = PeftModel.from_pretrained(deepcopy(original_model), 'linear_lora_model')

# 生成相同的输入数据以进行输出比较
test_input = torch.randn(1, 10)

# 比较四个模型的输出（原始模型，LoRA，方法1，方法2）
def compare_model_outputs(input_data):
    # 原始模型
    original_output = original_model(input_data)
    print("原始模型输出:", original_output.detach().numpy())

    # 训练后的 LoRA 模型
    lora_output = lora_model(input_data)
    print("训练后的 LoRA 模型输出:", lora_output.detach().numpy())

    # 方法 1：先使用 get_peft_model，再加载 LoRA
    output1 = model1(input_data)
    print("方法 1（先使用 get_peft_model，再加载 LoRA）输出:", output1.detach().numpy())

    # 方法 2：直接加载 LoRA
    output2 = model2(input_data)
    print("方法 2（直接加载 LoRA）输出:", output2.detach().numpy())

    if torch.allclose(original_output, output1):
        print("\n原始模型和方法 1 输出相同。")
    if torch.allclose(lora_output, output2):
        print("训练后的 LoRA 模型和方法 2 输出相同。\n")

# 比较两个模型的参数
def compare_params(m1, m2):
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        if n1 != n2 or not torch.allclose(p1, p2):
            print(f"参数不匹配: \n{n1}\n{n2}")
            return False
    return True

# 比较四个模型的输出
compare_model_outputs(test_input)

# 检查方法 1 和方法 2 的参数是否一致
if compare_params(model1, model2):
    print("方法 1 和方法 2 的 LoRA 模型参数一致！")
else:
    print("方法 1 和方法 2 的 LoRA 模型参数不一致！")