# Compressor - 图像视频压缩工具技术文档

`Compressor` 是一款基于 **PyQt6** 开发的桌面端多媒体压缩工具，旨在提供高效、美观且易用的图像（及初步视频）压缩方案。该工具集成了多种现代图像格式（WebP, AVIF）支持，并在处理过程中提供实时的画质损失监控。

---

## 🎨 设计理念与 UI 风格

- **视觉风格**：采用增强版深色模式 (Enhanced Dark Mode)，设计灵感来源于现代专业后期软件（如 Adobe Premiere, DaVinci Resolve）。
- **交互逻辑**：
  - **无边框设计**：沉浸式窗口体验，支持自定义标题栏及拖拽。
  - **实时预览**：缩略图即时生成，支持双击对比原图与效果图。
  - **响应式布局**：使用 `QSplitter` 和动态布局，适配不同窗口尺寸。

---

## 🚀 核心架构

项目采用典型的 **MVC (Model-View-Controller)** 与 **多线程处理架构**：

### 1. 处理引擎 (Engine)
- **ImageWorker (`QRunnable`)**: 核心压缩任务。
  - **格式支持**：利用 `Pillow` 和 `pillow-avif-plugin` 支持 WebP, AVIF, JPEG。
  - **画质评估**：通过计算 PSNR (峰值信噪比) 模拟 SSIM (结构相似性)，量化压缩损失。
  - **缩略图生成**：自动提取 120x80 的预览图以供 UI 展示。
- **线程池 (`QThreadPool`)**: 确保在处理大批量文件时，主界面保持流畅不卡顿。

### 2. 界面组件 (Components)
- **`MediaItemWidget`**: 高度定制的列表项，集成进度条、状态图标及动态文本（显示压缩百分比和画质指数）。
- **`CompareDialog`**: 提供大图并排放大的对比视图。
- **`Compressor`**: 主控类，负责业务逻辑流转、文件 IO 及全局样式管理。

---

## 🛠️ 技术关键点

### 图像压缩逻辑
```python
# 动态导出逻辑段落
if fmt.upper() == "AVIF": 
    orig_img.save(buf, format="AVIF", quality=q)
elif fmt.upper() == "WEBP": 
    orig_img.save(buf, format="WEBP", quality=q)
```

### 画质仿真算法
采用 PSNR 作为基准对 SSIM 进行仿真映射，分值范围 0.0 ~ 1.0。
- `ssim_val = min(1.0, psnr / 50.0)`
- **绿色标志** (SSIM ≥ 0.95)：表示肉眼几乎不可察觉的损失。
- **红色标志** (SSIM < 0.95)：表示有可见画质下降。

---

## 📦 环境依赖

若要运行此程序，需安装以下 Python 库：

| 库名称 | 用途 |
| :--- | :--- |
| `PyQt6` | UI 框架 |
| `Pillow` | 基础图像处理 |
| `pillow-avif-plugin` | 提供 AVIF 编码支持 |
| `numpy` | SSIM/PSNR 矩阵计算 |

```bash
pip install PyQt6 Pillow pillow-avif-plugin numpy
```

---

## 📖 使用指南

1. **添加资源**：支持通过“添加文件”或“导入文件夹”批量载入任务。
2. **配置选项**：
   - **格式切换**：由于采用响应式引擎，切换格式（WebP/AVIF）将立即触发后台重新渲染。
   - **压缩率设置**：`0.0` 为最大压缩，`1.0` 为无损/极高质量。
3. **预览对比**：在列表项上**双击**即可弹出对比窗口，通过观察细节决定最终导出设置。
4. **批量导出**：确认导出目录后，一键保存所有已处理的任务。

---

> [!TIP]
> **关于 AVIF 格式**：AVIF 拥有极高的压缩率，但在某些旧设备上可能无法正常打开。为了兼容性，建议优先选择 **WebP**。

> [!IMPORTANT]
> **视频处理说明**：当前版本的视频处理选项为 UI 占位符，完整编解码功能（H.265/VP9）计划在后续版本中接入 FFmpeg 引擎实现。
