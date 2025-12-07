import torch

# 设备
device = (
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)

# 训练超参数
block_size = 128
batch_size = 256
max_iters = 3000
eval_interval = 200
eval_iters = 50
learning_rate = 3e-4
seed = 42

# 模型超参数
n_layer = 4
n_head = 4
n_embed = 128
dropout = 0.1

# Tokenization level: 'char' | 'word'
token_level = "word"
# 数据源：'local' | 'wikitext2' 
data_source = "wikitext2"
wikitext_config = "wikitext-2-raw-v1"

# 在线语料抽样与截断，避免一次性加载过大
online_take_n = 200000            # 最多拿多少条样本（文章/段落）
online_max_text_chars = 2_000_000  # 最多拼接多少字符