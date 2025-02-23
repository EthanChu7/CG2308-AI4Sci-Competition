# AI4Sci Competition: CG2308
## 题目: 基于预训练模型的分子毒性预测

### 模型 checkpoints参见res文件夹:
[best_fine_tune_ChemBERTa_task1_seed_1.pt](res/best_fine_tune_ChemBERTa_task1_seed_1.pt)

[best_fine_tune_ChemBERTa_task2_seed_1.pt](res/best_fine_tune_ChemBERTa_task2_seed_1.pt)

由于模型checkpoints文件超出结果文件上传限制，请使用百度网盘链接下载checkpoints文件，并放置于`res`文件夹内。

百度网盘：https://pan.baidu.com/s/1NM4HB7Q1_ScXgKuEVKi9_A?pwd=klyh ，提取码：klyh。

### 实验结果参见res文件夹:
[fine_tune_ChemBERTa_task1.xlsx](res/fine_tune_ChemBERTa_task1.xlsx)

[fine_tune_ChemBERTa_task2.xlsx](res/fine_tune_ChemBERTa_task2.xlsx)

### 复现子任务1, 请运行:
    python Experiment_task_1.py

[Experiment_task_1.py](Experiment_task_1.py)

### 复现子任务2, 请运行: 
    python Experiment_task_2.py
[Experiment_task_2.py](Experiment_task_2.py)


### Shap可解释性可视化参见：
[shapley.ipynb](shapley.ipynb)

### GNN baselines checkpoints参见res文件夹:
[best_gat_task1_seed_1.pt](res/best_gat_task1_seed_1.pt)

[best_gat_task2_seed_1.pt](res/best_gat_task2_seed_1.pt)

[best_gcn_task1_seed_1.pt](res/best_gcn_task1_seed_1.pt)

[best_gcn_task2_seed_1.pt](res/best_gcn_task2_seed_1.pt)

### GNN baselines实验结果参见res文件夹:
[gat_task1.xlsx](res/gat_task1.xlsx)

[gcn_task2.xlsx](res/gcn_task2.xlsx)

[gat_task2.xlsx](res/gat_task2.xlsx)

[gcn_task1.xlsx](res/gcn_task1.xlsx)

### 复现 GNN baslines子任务1, 请运行:
    python Experiment_gnn_baseline_task1.py --model_name ['gcn' or 'gat]
[Experiment_gnn_baseline_task1.py](Experiment_gnn_baseline_task1.py)

### 复现 GNN baslines子任务2, 请运行:
    python Experiment_gnn_baseline_task2.py --model_name ['gcn' or 'gat]
[Experiment_gnn_baseline_task2.py](Experiment_gnn_baseline_task2.py)

### 实验默认参数设置参见:
[config.py](config.py)

### 模型架构参见：
[models.py](models.py)

### 数据预处理参见：
[ultis.py](ultis.py)

