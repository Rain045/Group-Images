# 🖼️ Image Grouper AI - Core

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)
![AI](https://img.shields.io/badge/Model-CLIP%20(OpenAI)-brightgreen.svg)
![Backend](https://img.shields.io/badge/Backend-OpenVINO%20%7C%20PyTorch-red.svg)
![Algorithm](https://img.shields.io/badge/Algorithm-HDBSCAN%20%7C%20SVM-yellow.svg)

**Image Grouper AI** 是一款基于语义理解的现代化图像管理工具。它利用 OpenAI 的 **CLIP (Contrastive Language-Image Pre-training)** 模型，将传统的“文件名分类”提升为“视觉特征分类”，让海量图片的整理工作变得智能且高效。

---

## ✨ 核心特性

* **🧠 三大智能分组模式**：
    * **准分类模式 (Text-Guided)**：通过自然语言描述（如“猫”、“海滩”、“建筑”）自动匹配并归类图片。基于 CLIP 的跨模态特征对齐。
    * **AI 发现模式 (Auto-Cluster)**：利用 **HDBSCAN/DBSCAN 算法** 自动分析视觉特征，无需人工干预即可发现相似图片组。支持细粒度、平衡、粗粒度三种灵敏度调节。
    *   **专属规则 (SVM 进化)**：通过手动微调结果进行“学习”，训练专属于您的 **SVM (支持向量机)** 分类模型。支持“吸纳经验”、“保存/导入规则”及“洗脑（清空记忆）”。
* **⚡ 多后端推理优化**：
    * **OpenVINO™**：针对 Intel CPU 的极速优化方案，适合无显卡的办公设备。
    * **PyTorch (CUDA/CPU)**：支持 NVIDIA 显卡加速，适合高性能计算。
* **🖱️ 交互式体验**：
    * **响应式布局**：基于 PyQt6 构建，支持 Windows 11 原生深色模式适配（沉浸式标题栏）。
    * **拖拽微调**：支持在不同分组间通过鼠标拖拽图片，实时调整分类并将规则反馈给 AI 引擎。
    * **增量缓存**：自动生成 `.embeddings_cache.pkl`，避免重复扫描图片，极大地提升二次处理速度。
    * **内置工具**：提供双击即开的图像预览器及带有“撤回（Restore）”能力的垃圾回收站（Trash）。

---

## 🚀 快速上手

1. **选择目录**：点击“浏览目标文件夹”加载图片项目。
2. **提取特征**：选择推理引擎（推荐 CPU 用户使用 OpenVINO），点击“1. 提取全量特征”。程序将计算每张图片的 CLIP Embedding 并进行本地缓存。
3. **模式选择**：
    * 输入关键词进行**准分类**。
    * 使用**AI 发现**进行无监督聚类。
    * 点击“2. 执行 AI 分组”查看结果。
4. **进化与调优**：
    * 如果 AI 分类有误，直接将其拖入正确的组。
    * 在“我的专属规则”模式下，点击**🧠 吸收经验**，AI 将记住您的手动修订偏好，并在之后以此规则进行预测。

---

## 🛠️ 技术栈与算法

* **特征提取**：`openai/clip-vit-base-patch32`
* **聚类算法**：优先使用 `HDBSCAN`（自适应密度），当环境不支持时自动降级为 `DBSCAN`。
* **分类模型**：线性核 `SVC` (Support Vector Classifier)，支持类别权重平衡（Balanced Weight）。
* **数据持久化**：使用 `pickle` 模块存储高维特征向量与训练好的模型权重。
* **UI 框架**：PyQt6，搭配自定义深色样式表（QSS）。

---

## 📦 环境依赖

运行前请安装以下依赖库（推荐使用中科大或阿里云镜像）：

```bash
pip install PyQt6 numpy pillow torch transformers scikit-learn
```

若需使用 OpenVINO 加速，请额外安装：
```bash
pip install openvino
```

> [!TIP]
> 针对国内网络环境，程序已内置 `HF_ENDPOINT=https://hf-mirror.com` 以加速模型权重下载过程。

---

## 📄 存储结构

* 目录内 `.embeddings_cache.pkl`：存储该文件夹下所有图片的特征向量。
* 程序同级 `custom_ai_rules.pkl`：存储 SVM 进化出的全局分类经验。
* 目录内 `Trash/`：存放被删除的图片，可在软件内随时还原。