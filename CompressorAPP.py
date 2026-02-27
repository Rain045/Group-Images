import sys
import io
import os
import numpy as np
from PIL import Image
import pillow_avif

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QLineEdit, QComboBox, 
                             QPushButton, QListWidget, QListWidgetItem, QSplitter, 
                             QDialog, QFileDialog, QMessageBox, QProgressBar, 
                             QSizePolicy, QFrame)
from PyQt6.QtCore import Qt, QRunnable, QThreadPool, QObject, pyqtSignal, pyqtSlot, QTimer, QSize
from PyQt6.QtGui import QImage, QPixmap, QDoubleValidator

# ==========================================
# 增强版深色模式 QSS 样式表
# ==========================================
STYLESHEET = """
/* 修改位置：STYLESHEET 字符串中 */
QMainWindow { 
    background-color: #0F0F0F; 
}

/* 针对侧边栏顶部进行微调，让其在视觉上更靠近窗口顶端 */
#SidePanel { 
    background-color: #161616; 
    border-right: 1px solid #2D2D2D;
    /* 如果觉得顶部文字太靠下，可以调整这里的内边距 */
}
/* 全局窗口与基础背景 */
QMainWindow, QDialog { 
    background-color: #0F0F0F; 
}
QWidget { 
    color: #E0E0E0; 
    font-family: "Segoe UI", "PingFang SC", sans-serif; 
}

/* 左侧面板：稍微亮一点点以示区分 */
#SidePanel { 
    background-color: #161616; 
    border-right: 1px solid #2D2D2D; 
}

/* 解决 QListWidget 可能出现的白边或默认背景 */
QListWidget { 
    background-color: #0F0F0F; 
    border: none; 
    outline: none; 
}

/* 列表项背景 */
QListWidget::item { 
    background-color: #1A1A1A; 
    margin: 5px 10px; 
    border-radius: 8px; 
    border: 1px solid #262626; 
}
QListWidget::item:selected { 
    background-color: #262626; 
    border: 1px solid #0078D4; 
}

/* 标题样式 */
#SideTitle { 
    color: #FFFFFF; 
    font-size: 24px; 
    font-weight: bold; 
    margin-bottom: 10px; 
    padding: 5px;
}

/* 按钮样式：确保即便在非 Focus 状态下也是深色的 */
QPushButton { 
    background-color: #2D2D2D; 
    color: #FFFFFF; 
    border: 1px solid #3D3D3D; 
    border-radius: 6px; 
    padding: 8px; 
    font-weight: 500; 
}
QPushButton:hover { 
    background-color: #3D3D3D; 
}
QPushButton:disabled {
    background-color: #1A1A1A;
    color: #555555;
}

QLineEdit {
    background-color: #1A1A1A; 
    color: #FFFFFF; 
    border: 1px solid #333333;
    border-radius: 4px; 
    padding: 6px 10px; 
}

/* 针对 QComboBox (下拉框) 进行独立高对比度优化 */
QComboBox {
    background-color: #1A1A1A; 
    color: #FFFFFF; 
    border: 1px solid #333333;
    border-radius: 4px; 
    padding: 6px 10px; 
}

/* 鼠标悬浮时边框高亮 */
QComboBox:hover {
    border: 1px solid #555555;
}

/* 修复下拉框展开后的列表文字对比度和背景 */
QComboBox QAbstractItemView {
    background-color: #252525;    /* 列表背景色略微提亮，与主输入框区分 */
    color: #FFFFFF;               /* 确保列表文字为纯白 */
    border: 1px solid #3D3D3D;    /* 列表边框 */
    selection-background-color: #0078D4; /* 选中项的背景色（微软蓝） */
    selection-color: #FFFFFF;     /* 选中项的文字纯白 */
    outline: none;                /* 去除点击时默认的虚线框 */
}

/* 滚动条深色化 */
QScrollBar:vertical {
    border: none;
    background: #0F0F0F;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #333333;
    border-radius: 5px;
}

/* 标签提示色 */
QLabel#Hint { 
    color: #0078D4; 
    font-weight: bold; 
    text-transform: uppercase; 
    font-size: 11px; 
    margin-top: 15px; 
}

/* 修改 QSplitter 分割线，去除默认的刺眼白边 */
QSplitter::handle {
    background-color: #2D2D2D; /* 极细的深色分割线，如果想完全隐形可以改成 #0F0F0F */
    width: 1px;
}
"""

# ==========================================
# 处理引擎 (包含 SSIM 计算)
# ==========================================
class WorkerSignals(QObject):
    finished = pyqtSignal(int, int, int, float, bytes, QPixmap, QPixmap, int)

class ImageWorker(QRunnable):
    def __init__(self, item_id, path, fmt, quality, gen, signals):
        super().__init__()
        self.item_id, self.path, self.fmt, self.quality, self.gen, self.signals = item_id, path, fmt, quality, gen, signals

    @pyqtSlot()
    def run(self):
        try:
            orig_img = Image.open(self.path).convert("RGB")
            orig_size = os.path.getsize(self.path)
            buf = io.BytesIO()
            q = int(self.quality * 100)
            
            # 执行压缩
            if self.fmt.upper() == "AVIF": orig_img.save(buf, format="AVIF", quality=q)
            elif self.fmt.upper() == "WEBP": orig_img.save(buf, format="WEBP", quality=q)
            else: orig_img.save(buf, format="JPEG", quality=q)

            comp_bytes = buf.getvalue()
            comp_size = len(comp_bytes)
            
            # SSIM 仿真计算 (基于 PSNR)
            comp_img = Image.open(io.BytesIO(comp_bytes)).convert("RGB")
            o_arr = np.array(orig_img.resize((256, 256)), dtype=np.float32)
            c_arr = np.array(comp_img.resize((256, 256)), dtype=np.float32)
            mse = np.mean((o_arr - c_arr) ** 2)
            psnr = 100.0 if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))
            ssim_val = min(1.0, psnr / 50.0)

            # 生成预览略缩图
            orig_img.thumbnail((120, 80))
            comp_img.thumbnail((120, 80))
            o_pix = QPixmap.fromImage(self.pil_to_qimage(orig_img))
            c_pix = QPixmap.fromImage(self.pil_to_qimage(comp_img))

            self.signals.finished.emit(self.item_id, orig_size, comp_size, ssim_val, comp_bytes, o_pix, c_pix, self.gen)
        except Exception as e:
            print(f"Error processing {self.path}: {e}")

    def pil_to_qimage(self, pil_img):
        data = pil_img.tobytes("raw", "RGB")
        return QImage(data, pil_img.width, pil_img.height, pil_img.width * 3, QImage.Format.Format_RGB888)

# ==========================================
# 对比弹窗
# ==========================================
class CompareDialog(QDialog):
    def __init__(self, orig_path, comp_bytes, parent=None):
        super().__init__(parent)
        self.setStyleSheet("background-color: #0F0F0F;") 
        self.setWindowTitle("画质对比预览")
        self.resize(1000, 600)
        layout = QHBoxLayout(self)
        for content in [orig_path, io.BytesIO(comp_bytes)]:
            lbl = QLabel()
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("background-color: #000; border: 1px solid #333;")
            img = Image.open(content).convert("RGB")
            data = img.tobytes("raw", "RGB")
            qimg = QImage(data, img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
            lbl.setPixmap(QPixmap.fromImage(qimg).scaled(480, 550, Qt.AspectRatioMode.KeepAspectRatio))
            layout.addWidget(lbl)

# ==========================================
# 列表项组件 (人性化存储显示 + SSIM)
# ==========================================
class MediaItemWidget(QWidget):
    def __init__(self, filename, is_video, parent_app, item_id):
        super().__init__()
        self.parent_app, self.item_id, self.is_video = parent_app, item_id, is_video
        
        # 1. 主布局设置
        layout = QHBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15) # 增加左右内边距，提升呼吸感
        layout.setSpacing(20)
        # 核心修改：强制要求布局内的所有控件在垂直方向上居中
        layout.setAlignment(Qt.AlignmentFlag.AlignVCenter) 

        # 2. 缩略图组
        self.thumb_orig = QLabel()
        self.thumb_comp = QLabel()
        
        for lbl in (self.thumb_orig, self.thumb_comp):
            lbl.setFixedSize(100, 70)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("""
                background-color: #0A0A0A; 
                border-radius: 6px; 
                border: 1px solid #333;
            """)
        
        layout.addWidget(self.thumb_orig)
        
        arrow_lbl = QLabel("→")
        arrow_lbl.setStyleSheet("color: #0078D4; font-weight: bold; font-size: 18px;")
        layout.addWidget(arrow_lbl)
        
        layout.addWidget(self.thumb_comp)
        
        # 3. 右侧信息展示区
        info_layout = QVBoxLayout()
        info_layout.setSpacing(6)
        # 核心修改：让文字信息在垂直方向也居中
        info_layout.setAlignment(Qt.AlignmentFlag.AlignVCenter) 
        
        self.lbl_name = QLabel(filename)
        self.lbl_name.setStyleSheet("font-weight: bold; color: #FFFFFF; font-size: 14px;")
        
        self.lbl_status = QLabel("等待处理..." if not is_video else "视频模式: 导出时处理")
        self.lbl_status.setStyleSheet("color: #AAAAAA; font-size: 12px;")
        
        self.pbar = QProgressBar()
        self.pbar.setFixedHeight(4)
        self.pbar.setTextVisible(False)
        self.pbar.setStyleSheet("""
            QProgressBar { background-color: #2D2D2D; border-radius: 2px; border: none; }
            QProgressBar::chunk { background-color: #0078D4; border-radius: 2px; }
        """)
        self.pbar.hide()
        
        info_layout.addWidget(self.lbl_name)
        info_layout.addWidget(self.lbl_status)
        info_layout.addWidget(self.pbar)
        
        layout.addLayout(info_layout, stretch=1)

        # 4. 操作按钮
        if not is_video:
            self.btn_save = QPushButton("保存")
            self.btn_save.setFixedSize(70, 32)
            self.btn_save.setEnabled(False)
            self.btn_save.clicked.connect(lambda: parent_app.save_single(self.item_id))
            layout.addWidget(self.btn_save)

    # 核心修改：重写 sizeHint 确保 QListWidget 给予足够的行高度
    def sizeHint(self):
        return QSize(self.width(), 110)

    def mouseDoubleClickEvent(self, event):
        if not self.is_video:
            self.parent_app.show_comparison(self.item_id)

    def format_size(self, size_bytes):
        if size_bytes < 1024: return f"{size_bytes} B"
        elif size_bytes < 1048576: return f"{size_bytes/1024:.1f} KB"
        else: return f"{size_bytes/1048576:.2f} MB"

    def update_info(self, o_s, c_s, ssim, o_pix, c_pix):
        self.thumb_orig.setPixmap(o_pix)
        self.thumb_comp.setPixmap(c_pix)
        
        # 1. 处理体积变化比例与颜色
        if o_s > 0:
            if c_s > o_s:
                # 越压越大：计算增加比例，红色，带 + 号
                diff_ratio = ((c_s - o_s) / o_s) * 100
                ratio_str = f"<span style='color:#FF5252;'>+{diff_ratio:.1f}%</span>"
            else:
                # 成功压缩：计算减少比例，绿色，带 - 号
                diff_ratio = ((o_s - c_s) / o_s) * 100
                ratio_str = f"<span style='color:#4CAF50;'>-{diff_ratio:.1f}%</span>"
        else:
            ratio_str = "<span style='color:#AAAAAA;'>0.0%</span>"

        # 2. 画质颜色判断
        quality_color = "#4CAF50" if ssim >= 0.95 else "#FF5252"
        
        # 3. 拼接并更新文本
        self.lbl_status.setText(
            f"<span>{self.format_size(o_s)}</span> → "
            f"<b style='color:white;'>{self.format_size(c_s)}</b> "
            f"({ratio_str}) "
            f"| 画质: <span style='color:{quality_color};'>{ssim:.4f}</span>"
        )
        
        if not self.is_video: 
            self.btn_save.setEnabled(True)

# ==========================================
# 主程序: Compressor
# ==========================================
class Compressor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("Compressor")
        self.resize(1100, 780)
        self.setStyleSheet(STYLESHEET)
        
        self.threadpool = QThreadPool()
        self.image_data = {}
        self.current_gen = 0
        self._next_id = 1
        self.signals = WorkerSignals()
        self.signals.finished.connect(self.on_worker_done)
        
        self.init_ui()

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        
        # 唯一的主布局（垂直），用于上下排列 标题栏 和 主体内容
        master_layout = QVBoxLayout(central)
        master_layout.setContentsMargins(0, 0, 0, 0)
        master_layout.setSpacing(0)


        # --- 自定义标题栏 ---
        title_bar = QFrame()
        title_bar.setObjectName("TitleBar")  # 增加专属 ID
        title_bar.setFixedHeight(35)
        # 使用 #TitleBar 限制样式只作用于背景框，防止污染内部的 Label 导致重影遮挡
        title_bar.setStyleSheet("#TitleBar { background-color: #161616; border-bottom: 1px solid #2D2D2D; }")
        title_layout = QHBoxLayout(title_bar)
        title_layout.setContentsMargins(15, 0, 10, 0)

        # 标题文字
        title_label = QLabel("Compressor - 图像视频压缩工具")
        title_label.setStyleSheet("font-size: 12px; color: #888; border: none;")
        title_layout.addWidget(title_label)
        
        title_layout.addStretch()

        # 关闭按钮
        btn_close = QPushButton("✕")
        btn_close.setFixedSize(30, 30)
        btn_close.setStyleSheet("QPushButton { border:none; background:none; font-size: 16px; } "
                                "QPushButton:hover { background-color: #E81123; color: white; }")
        btn_close.clicked.connect(self.close)
        title_layout.addWidget(btn_close)

        # 将自定义标题栏加入主布局顶部
        master_layout.addWidget(title_bar)

        # --- 原有的 Splitter 部分 (主体内容) ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # --- 左侧控制面板 ---
        side = QFrame()
        side.setObjectName("SidePanel")
        side.setFixedWidth(280)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(20, 30, 20, 30)
        
        title = QLabel("Compressor")
        title.setObjectName("SideTitle")
        side_layout.addWidget(title)
        
        # 导入
        h1 = QLabel("资源导入")
        h1.setObjectName("Hint")
        side_layout.addWidget(h1)
        btn_add_file = QPushButton("添加文件 (图片/视频)")
        btn_add_file.clicked.connect(self.import_files)
        side_layout.addWidget(btn_add_file)
        btn_add_folder = QPushButton("导入文件夹")
        btn_add_folder.clicked.connect(self.import_folder)
        side_layout.addWidget(btn_add_folder)
        
        # 图片区
        h2 = QLabel("图片处理选项")
        h2.setObjectName("Hint")
        side_layout.addWidget(h2)
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["WEBP", "AVIF", "JPEG"])
        self.combo_fmt.currentIndexChanged.connect(self.reprocess_images)
        side_layout.addWidget(self.combo_fmt)
        
        self.edit_quality = QLineEdit("0.8")
        self.edit_quality.setValidator(QDoubleValidator(0.0, 1.0, 2))
        self.edit_quality.editingFinished.connect(self.reprocess_images)
        side_layout.addWidget(QLabel("图片压缩率 (0.0-1.0)"))
        side_layout.addWidget(self.edit_quality)
        
        # 视频区
        h3 = QLabel("视频处理选项")
        h3.setObjectName("Hint")
        side_layout.addWidget(h3)
        self.video_preset = QComboBox()
        self.video_preset.addItems(["H.264 (高效)", "H.265 (超小)", "VP9"])
        side_layout.addWidget(self.video_preset)
        
        side_layout.addStretch()
        
        self.btn_batch = QPushButton("🚀 批量导出全部任务")
        self.btn_batch.setObjectName("PrimaryBtn")
        self.btn_batch.setFixedHeight(50)
        self.btn_batch.clicked.connect(self.batch_export)
        side_layout.addWidget(self.btn_batch)
        
        # --- 右侧列表区 ---
        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.list_widget = QListWidget()
        right_layout.addWidget(QLabel("-----任务队列-----", alignment=Qt.AlignmentFlag.AlignCenter))
        right_layout.addWidget(self.list_widget)
        
        # 将左右面板加入 splitter
        splitter.addWidget(side)
        splitter.addWidget(right)
        
        # 最后，将装配好的 splitter 加入到主布局的下方
        master_layout.addWidget(splitter)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton:
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

    def import_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", "", "Media (*.png *.jpg *.jpeg *.webp *.mp4 *.mkv *.mov)")
        for p in paths: self.add_item(p)
        self.reprocess_images()

    def import_folder(self):
        path = QFileDialog.getExistingDirectory(self, "选择文件夹")
        if not path: return
        for f in os.listdir(path):
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp', '.mp4')):
                self.add_item(os.path.join(path, f))
        self.reprocess_images()

    def add_item(self, path):
        is_video = path.lower().endswith(('.mp4', '.mkv', '.mov'))
        item = QListWidgetItem(self.list_widget)
        widget = MediaItemWidget(os.path.basename(path), is_video, self, self._next_id)
        
        # 显式告知 Item 它的尺寸暗示
        item.setSizeHint(widget.sizeHint()) 
        
        self.list_widget.setItemWidget(item, widget)
        self.image_data[self._next_id] = {'path': path, 'widget': widget, 'is_video': is_video, 'bytes': None}
        self._next_id += 1

    def reprocess_images(self):
        self.current_gen += 1
        fmt = self.combo_fmt.currentText()
        q = float(self.edit_quality.text() or 0.8)
        for i_id, data in self.image_data.items():
            if not data['is_video']:
                data['widget'].pbar.show()
                self.threadpool.start(ImageWorker(i_id, data['path'], fmt, q, self.current_gen, self.signals))

    @pyqtSlot(int, int, int, float, bytes, QPixmap, QPixmap, int)
    def on_worker_done(self, i_id, o_s, c_s, ssim, data, o_p, c_p, gen):
        if gen == self.current_gen:
            item_data = self.image_data[i_id]
            item_data['bytes'] = data
            item_data['widget'].update_info(o_s, c_s, ssim, o_p, c_p)
            item_data['widget'].pbar.hide()

    def show_comparison(self, i_id):
        data = self.image_data.get(i_id)
        if data and data['bytes']:
            CompareDialog(data['path'], data['bytes'], self).exec()

    def save_single(self, i_id):
        data = self.image_data[i_id]
        if data['bytes']:
            path, _ = QFileDialog.getSaveFileName(self, "保存文件", f"zen_{os.path.basename(data['path'])}")
            if path:
                with open(path, 'wb') as f: f.write(data['bytes'])

    def batch_export(self):
        target = QFileDialog.getExistingDirectory(self, "选择导出目录")
        if not target: return
        count = 0
        for i_id, data in self.image_data.items():
            if not data['is_video'] and data['bytes']:
                name = f"batch_{os.path.basename(data['path'])}"
                with open(os.path.join(target, name), 'wb') as f:
                    f.write(data['bytes'])
                count += 1
        QMessageBox.information(self, "导出完成", f"已成功导出 {count} 个图片文件。")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = Compressor()
    window.show()
    sys.exit(app.exec())