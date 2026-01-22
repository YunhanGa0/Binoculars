@echo off
REM HC3数据集评估快速启动脚本 (Windows版本)

echo ==========================================
echo HC3 Dataset Evaluation - Binoculars
echo ==========================================
echo.

REM 第一步：准备数据集
echo Step 1: Preparing HC3 English Dataset...
python experiments/hc3_loader.py

REM 检查数据集是否成功创建
if not exist "datasets\hc3\hc3_english_qa.jsonl" (
    echo Error: Failed to prepare HC3 dataset
    exit /b 1
)

echo [OK] Dataset prepared successfully
echo.

REM 第二步：快速验证实验（使用小样本）
echo Step 2: Running quick validation (100 samples)...
echo Using default models: gpt-neo-1.3B + gpt-neo-2.7B (high performance, 8GB GPU)...
python experiments/run_hc3_evaluation.py ^
    --dataset_path datasets/hc3/hc3_english_qa.jsonl ^
    --max_samples 100 ^
    --output_dir results/hc3_quick_test

echo [OK] Quick validation completed
echo.

REM 第三步：询问是否运行完整评估
set /p response="Do you want to run the full evaluation on complete dataset? (y/n): "

if /i "%response%"=="y" goto :full_eval
if /i "%response%"=="yes" goto :full_eval
goto :skip_full

:full_eval
echo.
echo Step 3: Running full evaluation on HC3 dataset...
echo Using default models: gpt-neo-1.3B + gpt-neo-2.7B (optimized for RTX 2070S/8GB)...

python experiments/run_hc3_evaluation.py ^
    --dataset_path datasets/hc3/hc3_english_qa.jsonl ^
    --output_dir results/hc3_full_evaluation

echo [OK] Full evaluation completed
goto :end

:skip_full
echo Skipping full evaluation.

:end
echo.
echo ==========================================
echo Evaluation finished!
echo Check results in the 'results/' directory
echo ==========================================
pause
