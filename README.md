# NLP Coursework Project

本仓库为《自然语言处理》课程大作业项目，主题为“针对多选任务的模型 SFT”。

项目以 `Qwen2.5-0.5B` 为基础模型，基于 `LLaMA-Factory` 完成 CommonsenseQA 多选任务的监督微调实验，并围绕不同输出路线、超参数消融以及 `LoRA vs Full` 拓展对比展开分析。

## 项目内容

- 任务类型：生成式多选任务
- 数据集：CommonsenseQA
- 基础模型：`Qwen2.5-0.5B`
- 训练框架：`LLaMA-Factory`
- 微调方式：以 `LoRA` 为主，补充 `Full fine-tuning` 对照

## 主要实验路线

- `fullset`：保留较自然的生成式输出形式
- `strict`：强化输出约束，要求输出标准选项
- `tmpl`：进一步约束为 `The correct answer is X.` 模板句式

其中，报告正文主要以 `strict` 主线为核心展开，`fullset` 与 `tmpl` 路线作为对比补充。

## 目录说明

- `Python脚本/01_数据处理`
  - 数据集转换与不同路线的数据构造脚本
- `Python脚本/02_评估脚本`
  - 普通评估、strict 评估、tmpl 评估及补充评估脚本
- `Python脚本/03_可视化与监控`
  - 训练曲线绘图、消融图生成、表格导出、显存监控等脚本
- `base_result/01_严格主线`
  - strict 主实验、消融实验及训练输出
- `base_result/02_严格主线_补充评估`
  - strict 路线 first_valid 补充评估结果
- `base_result/03_旧版Fullset路线`
  - 旧版 fullset 路线训练与评估结果
- `base_result/04_模板输出路线`
  - tmpl 路线训练、评估与可视化结果
- `base_result/05_对比图`
  - 跨路线对比图与 strict 消融图
- `base_result/06_导出表格` / `base_result/06_导出表格_最新版`
  - 报告中使用的 CSV 表格
- `LLaMA-Factory`
  - 训练框架与本项目使用的训练配置文件

## 结果入口

建议优先查看以下内容：

- 报告文档：
  - `人工智能2302-张诗琪-20236517-作业二.docx`
  - `人工智能2302-张诗琪-20236517-作业二.pdf`
- 结果索引：
  - `base_result/00_说明文档/结果索引_按文件夹整理.md`
- 关键表格：
  - `base_result/06_导出表格_最新版/00_当前主要结果总表.csv`
  - `base_result/06_导出表格_最新版/09_拓展部分_LoRA_vs_Full.csv`
- 关键图表：
  - `base_result/05_对比图/strict主线_学习率消融_柱状折线.png`
  - `base_result/05_对比图/旧版Fullset_vs_严格主线_vs_模板路线_training_loss_三线对比.png`
  - `base_result/05_对比图/旧版Fullset_vs_严格主线_vs_模板路线_validation_loss_三线对比.png`

## 说明

- 为避免 GitHub 仓库过大，模型权重、压缩包和 checkpoint 等大文件未纳入版本控制。
- 仓库中保留了完成课程作业所需的脚本、配置、评估结果、可视化图和报告文件。
- 若需复现实验，可结合 `LLaMA-Factory/examples/train_lora` 与 `LLaMA-Factory/examples/train_full` 下的配置文件进行训练。
