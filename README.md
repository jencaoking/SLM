# 问答系统模型训练脚本

基于PyTorch和Transformer架构的问答系统模型训练脚本，专为中国大陆网络环境优化，支持离线训练和多种数据格式。

## 🌟 核心特性

- **完整的Transformer架构**：自实现的编码器-解码器模型，专门针对问答任务优化
- **多格式数据支持**：支持CSV、JSON、JSONL、TSV、Parquet等多种数据格式
- **中国大陆网络适配**：支持HuggingFace镜像源、离线模式、魔搭ModelScope
- **智能数据处理**：自动格式检测、字段映射、数据验证和清洗
- **灵活的分词器**：支持中英文混合分词，基于jieba的中文处理
- **全面的评估指标**：EM、F1、BLEU、ROUGE等多种评估指标
- **检查点管理**：自动保存、最佳模型选择、断点续训
- **推理接口**：支持单个推理、批量推理、交互模式

## 📁 项目结构

```
LLM/
├── configs/                    # 配置文件
│   ├── config.yaml            # 主配置文件
│   └── data_configs/          # 数据集配置
│       ├── custom_data.yaml   # 自定义数据配置
│       └── squad_data.yaml    # SQuAD数据配置
├── models/                    # 模型架构
│   ├── __init__.py
│   ├── transformer.py         # 主模型
│   ├── encoder.py            # 编码器
│   ├── decoder.py            # 解码器
│   └── attention.py          # 注意力机制
├── utils/                     # 工具模块
│   ├── __init__.py
│   ├── data_loader.py        # 数据加载器
│   ├── data_processor.py     # 数据处理器
│   ├── tokenizer.py          # 分词器
│   ├── format_detector.py    # 格式检测器
│   └── metrics.py            # 评估指标
├── data/                      # 数据目录
│   └── raw/                  # 原始数据
│       └── custom/           # 自定义数据文件
│           ├── 1.jsonl       # 示例数据文件1
│           └── 2.jsonl       # 示例数据文件2
├── train.py                  # 训练脚本
├── inference.py              # 推理脚本
├── data_validator.py         # 数据验证脚本
├── test_system.py            # 系统测试脚本
├── requirements.txt          # 依赖包
└── README.md                 # 项目说明
```

> **注意**: 运行时会自动创建必要的目录（如 checkpoints/、logs/、outputs/、data/processed/、data/cache/ 等）

### 项目清理

为了保持项目整洁，建议定期清理以下目录：

```bash
# 清理Python缓存文件
find . -type d -name "__pycache__" -exec rm -rf {} +
# 或在Windows上：
for /d /r . %d in (__pycache__) do @if exist "%d" rd /s /q "%d"

# 清理空的输出目录（可选）
# 这些目录在训练过程中会重新创建
```

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目（如果需要）
git clone <repository_url>
cd LLM

# 创建虚拟环境（推荐）
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# 或者
.venv\Scripts\activate     # Windows

# 安装依赖（使用国内镜像源）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装PyTorch（根据CUDA版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 准备数据

将你的问答数据放在 `data/raw/custom/` 目录下，支持以下格式：

> **数据目录**: 项目中已包含示例数据文件 `1.jsonl` 和 `2.jsonl`，你可以参考其格式或替换为自己的数据。

#### CSV格式
```csv
question,context,answer,answer_start
"什么是机器学习？","机器学习是人工智能的一个重要分支...","人工智能的一个重要分支",5
```

#### JSON格式
```json
{
  "data": [
    {
      "question": "什么是机器学习？",
      "context": "机器学习是人工智能的一个重要分支...",
      "answer": "人工智能的一个重要分支",
      "answer_start": 5
    }
  ]
}
```

#### JSONL格式
```jsonl
{"question": "什么是机器学习？", "context": "机器学习是人工智能...", "answer": "人工智能的一个重要分支", "answer_start": 5}
```

### 3. 查看示例数据

项目中已包含两个示例数据文件：
- `data/raw/custom/1.jsonl`: 小规模示例数据 (21.9KB)
- `data/raw/custom/2.jsonl`: 大规模示例数据 (17MB)

你可以查看这些文件來了解数据格式，或者直接使用它们进行测试。

### 4. 数据验证

在训练之前，验证数据格式和质量：

```bash
python data_validator.py --file data/raw/custom/qa_data.csv --output validation_report.txt
```

### 5. 训练模型

```bash
# 使用默认配置训练
python train.py

# 使用自定义配置
python train.py --config configs/custom_config.yaml --seed 42
```

### 6. 模型推理

```bash
# 交互模式
python inference.py --model_path <训练完成的模型路径> --mode interactive

# 单个问题推理
python inference.py --model_path <训练完成的模型路径> --mode single \
    --question "什么是机器学习？" \
    --context "机器学习是人工智能的一个重要分支..."

# 批量推理
python inference.py --model_path <训练完成的模型路径> --mode file \
    --input_file test_data.json --output_file results.json
```

## 📊 系统测试

运行综合测试验证系统功能：

```bash
python test_system.py
```

测试包括：
- 数据格式检测
- 数据加载功能
- 数据验证功能
- 分词器功能
- 数据处理功能
- 模型创建
- 评估指标
- 配置加载

## ⚙️ 配置说明

主要配置文件 `configs/config.yaml` 包含以下部分：

### 模型配置
```yaml
model:
  vocab_size: 30000          # 词汇表大小
  d_model: 512              # 模型维度
  n_heads: 8                # 注意力头数
  n_layers: 6               # 编码器层数
  d_ff: 2048                # 前馈网络维度
  dropout: 0.1              # Dropout率
```

### 训练配置
```yaml
training:
  batch_size: 16            # 批次大小
  learning_rate: 5e-4       # 学习率
  num_epochs: 10            # 训练轮数
  gradient_accumulation_steps: 4  # 梯度累积步数
  warmup_steps: 1000        # 预热步数
```

### 数据配置
```yaml
data:
  datasets:
    - name: "custom_dataset"
      path: "data/raw/custom/qa_data.csv"
      format: "csv"
      weight: 1.0
      enabled: true
  
  max_length: 512           # 最大序列长度
  train_split: 0.8          # 训练集比例
  val_split: 0.1            # 验证集比例
  test_split: 0.1           # 测试集比例
```

### 网络环境配置
```yaml
network:
  offline_mode: true        # 离线模式
  use_mirror: true          # 使用镜像源
  mirror_endpoint: "https://hf-mirror.com"
  use_proxy: false          # 是否使用代理
  proxy_url: "http://127.0.0.1:7890"
```

## 🌐 中国大陆网络环境适配

### 1. HuggingFace镜像源
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### 2. 离线模式
```bash
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

### 3. 魔搭ModelScope
```python
from modelscope import MsDataset
dataset = MsDataset.load('squad', subset_name='plain_text')
```

### 4. 代理配置
```bash
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
```

## 📈 模型性能

### 模型架构
- **编码器**: 6层Transformer编码器，支持问题和上下文的独立编码
- **解码器**: 专门的问答解码器，包含问题-上下文交互和答案指针网络
- **注意力机制**: 多头自注意力和交叉注意力
- **位置编码**: 支持最大512长度的位置编码

### 训练策略
- **优化器**: AdamW with weight decay
- **学习率调度**: Cosine Annealing with warmup
- **梯度累积**: 支持大batch size训练
- **混合精度**: FP16训练减少内存占用
- **早停机制**: 防止过拟合

### 评估指标
- **精确匹配 (EM)**: 完全匹配的答案比例
- **F1分数**: 基于token重叠的F1分数
- **BLEU分数**: 生成质量评估
- **ROUGE分数**: 召回率导向评估

## 🔧 高级用法

### 自定义数据集

1. 创建数据配置文件：
```yaml
# configs/data_configs/my_dataset.yaml
dataset_info:
  name: "my_qa_dataset"
  description: "我的问答数据集"
  
field_mappings:
  question:
    primary: "question"
    alternatives: ["query", "q", "问题"]
  context:
    primary: "context"
    alternatives: ["passage", "text", "上下文"]
  answer:
    primary: "answer"
    alternatives: ["response", "答案"]
```

2. 在主配置中引用：
```yaml
data:
  datasets:
    - name: "my_dataset"
      path: "data/raw/my_data.json"
      format: "json"
      config: "configs/data_configs/my_dataset.yaml"
```

### 分布式训练

```bash
# 多GPU训练
python -m torch.distributed.launch --nproc_per_node=2 train.py

# 使用accelerate（推荐）
accelerate config
accelerate launch train.py
```

### 模型微调

```bash
# 从预训练模型开始微调
python train.py --config configs/finetune_config.yaml \
    --pretrained_model <预训练模型路径>
```

### API服务部署

```python
# app.py
from fastapi import FastAPI
from inference import QAInference

app = FastAPI()
# 使用训练完成的模型路径
qa_model = QAInference("<训练完成的模型路径>")

@app.post("/predict")
async def predict(question: str, context: str):
    result = qa_model.predict_single(question, context)
    return result
```

## 🐛 常见问题

### Q: 训练时内存不足怎么办？
A: 尝试以下方法：
- 减小batch_size
- 减小max_length
- 启用gradient_accumulation_steps
- 使用mixed_precision训练

### Q: 数据加载失败？
A: 检查：
- 文件格式是否正确
- 字段名是否匹配
- 文件编码是否为UTF-8
- 运行数据验证脚本检查问题

### Q: 模型训练不收敛？
A: 尝试：
- 调整学习率
- 增加warmup_steps
- 检查数据质量
- 调整模型大小

### Q: 推理结果不理想？
A: 检查：
- 训练数据质量和数量
- 模型是否收敛
- 推理时的confidence_threshold
- 答案长度限制

## 📝 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📧 联系方式

如有问题，请提交Issue或联系维护者。

---

**注意**: 本项目专为中国大陆网络环境优化，包含离线模式和镜像源配置，确保在网络受限环境下也能正常使用。