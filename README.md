# 🖼️ Image Processing Toolkit (Group-Images)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange.svg)](https://www.riverbankcomputing.com/software/pyqt/)
[![AI-Powered](https://img.shields.io/badge/AI-CLIP%20%7C%20OpenVINO-brightgreen.svg)](https://openai.com/blog/clip/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Group-Images** 是一套專為攝影師、設計師及多媒體創作者打造的現代化桌面工具集。它整合了尖端的深度學習模型（如 OpenAI CLIP）與傳統電腦視覺（CV）技術，旨在解決海量圖片的**智能分類、重複篩選與高效壓縮**三大核心需求。

本工具集採用基於 **PyQt6** 的增強版深色模式設計，提供沉浸式的專業視覺體驗。

---

## 🚀 核心組件

本項目包含三個互補的獨立應用程式，共同構成完整的圖像管理工作流：

### 1. 🧠 Image Grouper AI — 語義分組專家
基於內容理解而非簡單的文件名，自動整理您的相冊。
*   **三大模式**：自然語言導向的「准分類」、基於 HDBSCAN 的「AI 自動發現」、以及結合 SVM 的「專屬規則進化」。
*   **推理優化**：支持 OpenVINO™ (Intel CPU 加速) 與 PyTorch (CUDA GPU 加速)。
*   **互動微调**：支持拖拽調整與「吸收經驗」，讓 AI 越用越懂你。
*   [查看詳細文檔 (GrouperDoc.md)](GrouperDoc.md)

### 2. 📸 Image Similarity Pro — 美感篩選工具
快速清理連拍照片，挑選出構圖與畫質最佳的傑作。
*   **混合評分系統**：結合 CLIP 語義評分 (美學) 與 CV 技術評分 (清晰度、對比度、色彩豐富度)。
*   **多維比對**：提供 AI 特徵向量、OpenCV ORB 特徵點及感知哈希 (pHash) 三種分析引擎。
*   **安全清理**：自動標記「👑 最佳」與「🗑️ 待刪」，並提供撤回能力的垃圾桶機制。
*   [查看詳細文檔 (SimilarityDoc.md)](SimilarityDoc.md)

### 3. ⚡ Compressor — 圖像視頻壓縮工具
在保持極高畫質的前提下，極大縮減文件體積。
*   **現代格式支持**：全面支持 WebP 與 AVIF 等新一代高效編碼格式。
*   **即時畫質監控**：計算 PSNR 與 SSIM 仿真指標，動態顯示畫質損失情況 (0.0 - 1.0)。
*   **專業級 UI**：雙擊即開的比對窗口，支持原圖與壓縮效果的並排細節放大。
*   [查看詳細文檔 (CompressorDoc.md)](CompressorDoc.md)

---

## 🛠️ 技術棧

*   **GUI 框架**：PyQt6 (支持 Windows 11 沉浸式標題欄)
*   **AI 模型**：OpenAI CLIP (`openai/clip-vit-base-patch32`), ResNet18, MobileNet_V2
*   **圖像處理**：Pillow (PIL), OpenCV, ImageHash
*   **數據與算法**：NumPy, Scikit-learn (HDBSCAN, SVM)
*   **硬體加速**：OpenVINO, CUDA, macOS Metal (MPS)

---

## 📦 安裝與準備

### 環境要求
*   Python 3.8 或更高版本
*   建議配備 NVIDIA GPU (Windows/Linux) 或 Apple Silicon (macOS) 以獲得最佳 AI 性能

### 快速安裝
```bash
# 克隆倉庫
git clone https://github.com/YourUsername/Group-Images.git
cd Group-Images

# 安裝核心依賴
pip install PyQt6 Pillow ImageHash opencv-python numpy scikit-learn torch torchvision transformers

# (可選) 安裝 OpenVINO 以加速 Intel CPU 推理
pip install openvino

# (可選) 安裝 AVIF 支持插件
pip install pillow-avif-plugin
```

---

## 📖 快速上手

您可以根據需求運行對應的應用程式：

1.  **整理圖片**：`python GrouperAPP.py`
2.  **筛选相似圖/美感評分**：`python SimilarityAPP.py`
3.  **批量壓縮**：`python CompressorAPP.py`

---

## 🎨 設計理念

*   **Premium Aesthetics**：採用類似 Adobe Premiere 的深色工業風格，減少長時間操作的視覺疲勞。
*   **Local First**：所有 AI 運算均在本地執行，保護隱私且無需依賴網絡服務 (初次模型下載除外)。
*   **Fault Tolerance**：提供 `Trash/` 文件夾緩衝與增量快取機制，確保數據安全與處理效率。

---

## 📄 授權協議

本項目採用 [MIT License](LICENSE)。

> [!NOTE]
> 初次運行 AI 組件時，程式會從雲端自動下載模型權重檔案（約 300MB - 600MB），請確保您的網路連線暢通。
