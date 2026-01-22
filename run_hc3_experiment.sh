#!/bin/bash
# HC3数据集对比实验快速启动脚本

echo "=========================================="
echo "HC3 Dataset Comparison Experiment"
echo "=========================================="
echo ""

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 第一步：准备数据集
echo -e "${YELLOW}Step 1: Preparing HC3 Dataset...${NC}"
python experiments/hc3_loader.py

# 检查数据集是否成功创建
if [ ! -f "datasets/hc3/hc3_english_qa.jsonl" ]; then
    echo "Error: Failed to prepare HC3 dataset"
    exit 1
fi

echo -e "${GREEN}✓ Dataset prepared successfully${NC}"
echo ""

# 第二步：快速验证实验（使用小样本）
echo -e "${YELLOW}Step 2: Running quick validation (100 samples)...${NC}"
python experiments/run_hc3_comparison.py \
    --dataset_path datasets/hc3/hc3_english_qa.jsonl \
    --max_samples 100 \
    --run_custom \
    --custom_observer gpt2 \
    --custom_performer gpt2-medium \
    --batch_size 8 \
    --output_dir results/hc3_quick_test

echo -e "${GREEN}✓ Quick validation completed${NC}"
echo ""

# 第三步：询问是否运行完整实验
echo -e "${YELLOW}Do you want to run the full experiment? (y/n)${NC}"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo ""
    echo -e "${YELLOW}Step 3: Running full comparison experiment...${NC}"
    echo "This may take a while depending on your GPU..."
    
    python experiments/run_hc3_comparison.py \
        --dataset_path datasets/hc3/hc3_english_qa.jsonl \
        --run_original \
        --run_custom \
        --original_observer tiiuae/falcon-7b \
        --original_performer tiiuae/falcon-7b-instruct \
        --custom_observer gpt2 \
        --custom_performer gpt2-large \
        --batch_size 16 \
        --output_dir results/hc3_full_comparison
    
    echo -e "${GREEN}✓ Full experiment completed${NC}"
else
    echo "Skipping full experiment."
fi

echo ""
echo "=========================================="
echo "Experiment finished!"
echo "Check results in the 'results/' directory"
echo "=========================================="
