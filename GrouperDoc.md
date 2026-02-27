# 🖼️ Image Grouper AI - Core

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)
![AI](https://img.shields.io/badge/Model-CLIP%20(OpenAI)-brightgreen.svg)
![Backend](https://img.shields.io/badge/Backend-OpenVINO%20%7C%20PyTorch-red.svg)

**Image Grouper AI** 是一款基于语义理解的现代化图像管理工具。它利用 OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 模型，将传统的“文件名分类”提升为“视觉特征分类”，让海量图片的整理工作变得智能且高效。

---

## ✨ 核心特性

* **🧠 三大智能分组模式**：
    * **准分类模式 (Text-Guided)**：通过自然语言描述（如“猫”、“海滩”、“赛博朋克”）自动匹配并归类图片。
    * **AI 发现模式 (Auto-Cluster)**：利用 **DBSCAN 算法** 自动分析视觉特征，无需人工干预即可发现相似图片组。
    * **专属规则 (SVM 进化)**：通过手动微调结果进行“学习”，训练专属于您的 **SVM (支持向量机)** 分类模型。
* **⚡ 多后端推理优化**：
    * **OpenVINO™**：针对 Intel CPU 的极速优化方案，适合无显卡的办公设备。
    * **PyTorch (CUDA/CPU)**：支持 NVIDIA 显卡加速，适合高性能计算。
* **🖱️ 交互式体验**：
    * **拖拽微调**：支持在不同分组间通过鼠标拖拽图片，AI 会实时更新分类归属。
    * **增量缓存**：自动生成 `.embeddings_cache.pkl`，避免重复扫描图片，提升二次加载速度。
    * **内置工具**：提供双击大图预览及智能回收站（Trash）管理功能。

---

## 🛠️ 环境准备

运行本项目需要 Python 3.8+ 环境，请先安装以下依赖库：

```bash
pip install PyQt6 numpy pillow torch transformers scikit-learn