@echo off
REM HC3数据集对比实验快速启动脚本 (Windows版本)

echo ==========================================
echo HC3 Dataset Comparison Experiment
echo ==========================================
echo.

REM 第一步：准备数据集
echo Step 1: Preparing HC3 Dataset...
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
python experiments/run_hc3_comparison.py ^
    --dataset_path datasets/hc3/hc3_english_qa.jsonl ^
    --max_samples 100 ^
    --run_custom ^
    --custom_observer gpt2 ^
    --custom_performer gpt2-medium ^
    --batch_size 8 ^
    --output_dir results/hc3_quick_test

echo [OK] Quick validation completed
echo.

REM 第三步：询问是否运行完整实验
set /p response="Do you want to run the full experiment? (y/n): "

if /i "%response%"=="y" goto :full_experiment
if /i "%response%"=="yes" goto :full_experiment
goto :skip_full

:full_experiment
echo.
echo Step 3: Running full comparison experiment...
echo This may take a while depending on your GPU...

python experiments/run_hc3_comparison.py ^
    --dataset_path datasets/hc3/hc3_english_qa.jsonl ^
    --run_original ^
    --run_custom ^
    --original_observer tiiuae/falcon-7b ^
    --original_performer tiiuae/falcon-7b-instruct ^
    --custom_observer gpt2 ^
    --custom_performer gpt2-large ^
    --batch_size 16 ^
    --output_dir results/hc3_full_comparison

echo [OK] Full experiment completed
goto :end

:skip_full
echo Skipping full experiment.

:end
echo.
echo ==========================================
echo Experiment finished!
echo Check results in the 'results/' directory
echo ==========================================
pause
