import sys
import os
import time
import pickle
import shutil
import platform
import subprocess
import traceback

# ==========================================
# 1. 基础依赖与环境检测
# ==========================================
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

HAS_OPENVINO = False
HAS_TORCH = False
HAS_CUDA = False
HAS_SKLEARN = False

try:
    import openvino.runtime as ov
    HAS_OPENVINO = True
except ImportError:
    try:
        import openvino as ov
        HAS_OPENVINO = True
    except ImportError: pass

try:
    import torch
    from transformers import CLIPProcessor, CLIPModel
    HAS_TORCH = True
    HAS_CUDA = torch.cuda.is_available()
except ImportError: pass

try:
    from sklearn.cluster import DBSCAN
    from sklearn.svm import SVC
    HAS_SKLEARN = True
except ImportError: pass


# ==========================================
# 2. PyQt6 核心组件导入
# ==========================================
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QFrame, QPushButton, QComboBox, QLabel, QFileDialog, 
    QScrollArea, QGridLayout, QLineEdit, QProgressBar, 
    QStackedWidget, QGroupBox, QMessageBox, QDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QMimeData, QPoint
from PyQt6.QtGui import QPixmap, QFont, QImageReader, QDrag


# ==========================================
# 3. 模型全局缓存管理器
# ==========================================
class ModelManager:
    _processor = None
    _image_model = None
    _text_model = None
    _torch_model = None
    _current_backend = None
    _device = "cpu"
    _ov_path = None

    @classmethod
    def load_model(cls, backend_name, ov_path=None):
        if cls._processor is not None and cls._current_backend == backend_name and cls._ov_path == ov_path:
            return

        is_openvino = "OpenVINO" in backend_name
        cls._current_backend = backend_name
        cls._ov_path = ov_path
        cls._processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        if is_openvino:
            if not ov_path or not os.path.exists(ov_path):
                raise FileNotFoundError(f"OpenVINO 模型路径无效: {ov_path}")
            core = ov.Core()
            compiled_model = core.compile_model(ov_path, "CPU")
            cls._image_model = compiled_model
            cls._text_model = compiled_model
            cls._torch_model = None
        else:
            cls._device = "cuda" if "CUDA" in backend_name and HAS_CUDA else "cpu"
            cls._torch_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(cls._device)
            cls._image_model = None
            cls._text_model = None


# ==========================================
# 4. 异步任务处理器
# ==========================================
class Worker(QThread):
    progress = pyqtSignal(int, str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, task_type, params):
        super().__init__()
        self.task_type = task_type
        self.params = params
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        try:
            backend = self.params.get("backend", "")
            is_openvino = "OpenVINO" in backend
            
            if self.task_type == "PREPROC" or self.params.get("mode") == "text":
                self.progress.emit(0, "正在初始化模型引擎...")
                ModelManager.load_model(backend, self.params.get("ov_path"))
            
            processor = ModelManager._processor
            device = ModelManager._device
            torch_model = ModelManager._torch_model
            image_model = ModelManager._image_model
            text_model = ModelManager._text_model

            # --- 任务 A: 提取图像特征 ---
            if self.task_type == "PREPROC":
                images = self.params.get("images", [])
                total = len(images)
                embeddings = {}
                
                if total == 0:
                    self.finished.emit({"status": "success", "type": "preproc", "data": {}})
                    return

                for i, img_path in enumerate(images):
                    if self._is_cancelled: raise InterruptedError("任务已被手动中止")
                    try:
                        image = Image.open(img_path).convert("RGB")
                        
                        if is_openvino:
                            inputs = processor(text=[""], images=image, return_tensors="np", padding=True)
                            feed_dict = {}
                            for port in image_model.inputs:
                                for k, v in inputs.items():
                                    if any(k in n for n in port.get_names()): feed_dict[port.any_name] = v
                            if not feed_dict: feed_dict = {image_model.inputs[0]: inputs["pixel_values"]}

                            res = image_model(feed_dict)
                            img_features = None
                            for out_node, tensor in res.items():
                                if any("image_embed" in n for n in out_node.get_names()):
                                    img_features = tensor; break
                            if img_features is None: img_features = list(res.values())[0]

                            img_features = img_features / np.linalg.norm(img_features, axis=-1, keepdims=True)
                            embeddings[img_path] = img_features.flatten()
                        else:
                            inputs = processor(images=image, return_tensors="pt").to(device)
                            with torch.no_grad():
                                img_outputs = torch_model.get_image_features(**inputs)
                                img_features = img_outputs.pooler_output if getattr(img_outputs, "pooler_output", None) is not None else img_outputs[0]
                                if img_features.shape[-1] != 512 and hasattr(torch_model, "visual_projection"):
                                    img_features = torch_model.visual_projection(img_features)
                                img_features = img_features / img_features.norm(p=2, dim=-1, keepdim=True)
                            embeddings[img_path] = img_features.cpu().numpy().flatten()
                            
                    except Exception as e: print(f"⚠️ 无法处理 {img_path}: {e}")
                    self.progress.emit(int((i + 1) / total * 100), f"正在提取特征 ({i+1}/{total})...")
                
                self.finished.emit({"status": "success", "type": "preproc", "data": embeddings})

            # --- 任务 B: 分组 ---
            elif self.task_type == "GROUP":
                mode = self.params.get("mode")
                img_embeddings = self.params.get("embeddings", {})
                result_groups = {} 
                total = len(img_embeddings)

                self.progress.emit(10, "正在计算相似度与特征映射...")

                if mode == "text":
                    raw_tags = self.params.get("tags", "未分类").split(',')
                    tags = [t.strip() for t in raw_tags if t.strip()]
                    if not tags: tags = ["未分类"]

                    if is_openvino:
                        dummy_image = Image.new("RGB", (224, 224), (0, 0, 0))
                        inputs = processor(text=tags, images=dummy_image, return_tensors="np", padding=True)
                        feed_dict = {}
                        for port in text_model.inputs:
                            for k, v in inputs.items():
                                if any(k in n for n in port.get_names()): feed_dict[port.any_name] = v
                        if not feed_dict: feed_dict = {text_model.inputs[0]: inputs["input_ids"]}

                        res = text_model(feed_dict)
                        text_features = None
                        for out_node, tensor in res.items():
                            if any("text_embed" in n for n in out_node.get_names()):
                                text_features = tensor; break
                        if text_features is None: text_features = list(res.values())[0]
                    else:
                        inputs = processor(text=tags, return_tensors="pt", padding=True).to(device)
                        with torch.no_grad():
                            text_outputs = torch_model.get_text_features(**inputs)
                            text_features = text_outputs.pooler_output if getattr(text_outputs, "pooler_output", None) is not None else text_outputs[0]
                            if text_features.shape[-1] != 512 and hasattr(torch_model, "text_projection"):
                                text_features = torch_model.text_projection(text_features)
                        text_features = text_features.cpu().numpy()

                    text_features = text_features / np.linalg.norm(text_features, axis=-1, keepdims=True)

                    for tag in tags: result_groups[tag] = []
                    result_groups["其他 (不匹配)"] = []

                    for i, (img_path, img_emb) in enumerate(img_embeddings.items()):
                        if self._is_cancelled: raise InterruptedError("任务中止")
                        similarities = np.dot(text_features, np.array(img_emb).flatten()) 
                        best_idx = np.argmax(similarities)
                        if similarities[best_idx] > 0.22: result_groups[tags[best_idx]].append(img_path)
                        else: result_groups["其他 (不匹配)"].append(img_path)
                        self.progress.emit(int((i + 1) / total * 100), "进行语义比对...")

                elif mode == "ai":
                    eps_val = {0: 0.12, 1: 0.20, 2: 0.35}.get(self.params.get("eps_level", 1), 0.20)
                    paths = list(img_embeddings.keys())
                    matrix = np.array(list(img_embeddings.values()))
                    if len(matrix.shape) == 1: matrix = matrix.reshape(1, -1)
                        
                    dbscan = DBSCAN(eps=eps_val, min_samples=2, metric='cosine')
                    labels = dbscan.fit_predict(matrix)

                    result_groups["独立图片 (未归类)"] = []
                    for i, label in enumerate(labels):
                        if self._is_cancelled: raise InterruptedError("任务中止")
                        if label == -1: result_groups["独立图片 (未归类)"].append(paths[i])
                        else: result_groups.setdefault(f"智能发现组 {label + 1}", []).append(paths[i])
                        self.progress.emit(int((i + 1) / total * 100), "进行无监督聚类...")

                elif mode == "svm":
                    clf = self.params.get("svm_clf")
                    if not clf: raise ValueError("内存中未找到进化模型，请先学习经验或导入规则！")

                    paths = list(img_embeddings.keys())
                    matrix = np.array(list(img_embeddings.values()))
                    if len(matrix.shape) == 1: matrix = matrix.reshape(1, -1)
                    
                    predictions = clf.predict(matrix)
                    for i, label in enumerate(predictions):
                        if self._is_cancelled: raise InterruptedError("任务中止")
                        result_groups.setdefault(label, []).append(paths[i])
                        self.progress.emit(int((i + 1) / total * 100), "正在应用专属规则预测...")

                self.finished.emit({"status": "success", "type": "group", "data": result_groups})

        except Exception as e:
            traceback.print_exc()
            self.error.emit(str(e))


# ==========================================
# 5. 现代化 UI 组件与内置预览器
# ==========================================
class ImageViewerDialog(QDialog):
    def __init__(self, image_path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("内置图像预览")
        self.resize(900, 700)
        self.setStyleSheet("background-color: #1e1f22; color: white;")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.img_label)
        
        self.pixmap = QPixmap(image_path)
        self.update_image()

    def update_image(self):
        if not self.pixmap.isNull():
            scaled = self.pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.img_label.setPixmap(scaled)

    def resizeEvent(self, event):
        self.update_image()
        super().resizeEvent(event)


class ImageCard(QFrame):
    on_delete = pyqtSignal(object)
    double_clicked = pyqtSignal(str)
    
    def __init__(self, image_path, is_trash_mode=False):
        super().__init__()
        self.image_path = image_path
        self.is_trash_mode = is_trash_mode
        self.setFixedSize(150, 150)
        self.setToolTip("双击预览 | 长按拖拽以更换分组")
        self.drag_start_pos = None
        
        self.setStyleSheet("""
            ImageCard { 
                background-color: #2b2d31; /* 卡片默认深色 */
                border-radius: 10px; 
                border: 1px solid #1e1f22; 
            }
            ImageCard:hover { 
                border: 2px solid #5865F2; 
                background-color: #383a40; /* 鼠标悬浮时提亮一点点 */
            }
        """)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        reader = QImageReader(self.image_path)
        reader.setAutoTransform(True)
        if reader.size().isValid():
            reader.setScaledSize(reader.size().scaled(200, 200, Qt.AspectRatioMode.KeepAspectRatioByExpanding))
            img = reader.read()
            if not img.isNull():
                pixmap = QPixmap.fromImage(img)
                x = (pixmap.width() - 134) // 2
                y = (pixmap.height() - 134) // 2
                self.display_pixmap = pixmap.copy(x, y, 134, 134).scaled(134, 134, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                self.img_label.setPixmap(self.display_pixmap)
            else: self.img_label.setText("解析失败")
        else: self.img_label.setText("无效图片")
            
        layout.addWidget(self.img_label)

        self.action_btn = QPushButton("↺" if self.is_trash_mode else "×", self)
        color = "#23A559" if self.is_trash_mode else "#DA373C"
        self.action_btn.setStyleSheet(f"QPushButton {{ background-color: {color}; color: white; border-radius: 12px; font-weight: bold; border: none; }}")
        self.action_btn.setFixedSize(24, 24)
        self.action_btn.move(120, 6)
        self.action_btn.hide()
        self.action_btn.clicked.connect(self.process_action)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.drag_start_pos = event.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self.drag_start_pos or not (event.buttons() & Qt.MouseButton.LeftButton): return
        if (event.pos() - self.drag_start_pos).manhattanLength() < QApplication.startDragDistance(): return
            
        drag = QDrag(self)
        mime_data = QMimeData()
        mime_data.setText(self.image_path)
        drag.setMimeData(mime_data)
        
        if hasattr(self, 'display_pixmap'):
            drag.setPixmap(self.display_pixmap.scaled(80, 80, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
            drag.setHotSpot(QPoint(40, 40))
        drag.exec(Qt.DropAction.MoveAction)

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.double_clicked.emit(self.image_path)

    def process_action(self):
        try:
            base = os.path.basename(self.image_path)
            parent = os.path.dirname(self.image_path)
            
            if self.is_trash_mode: target_dir = os.path.dirname(parent)
            else: target_dir = os.path.join(parent, "Trash"); os.makedirs(target_dir, exist_ok=True)
                
            target_path = os.path.join(target_dir, base)
            if os.path.exists(target_path): 
                target_path = os.path.join(target_dir, f"{os.path.splitext(base)[0]}_{int(time.time())}{os.path.splitext(base)[1]}")
                
            shutil.move(self.image_path, target_path)
            self.delete_self()
        except Exception as e: QMessageBox.critical(self, "操作失败", str(e))

    def enterEvent(self, event): self.action_btn.show(); super().enterEvent(event)
    def leaveEvent(self, event): self.action_btn.hide(); super().leaveEvent(event)
    def delete_self(self): self.on_delete.emit(self); self.setParent(None); self.deleteLater()


class ResponsiveGridWidget(QWidget):
    image_dropped = pyqtSignal(str, str)

    def __init__(self, group_name=""):
        super().__init__()
        self.group_name = group_name
        self.setAcceptDrops(bool(group_name))
        self.grid = QGridLayout(self)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid.setSpacing(12)
        self.widgets = []

    def dragEnterEvent(self, event):
        if event.mimeData().hasText(): event.acceptProposedAction()

    def dropEvent(self, event):
        image_path = event.mimeData().text()
        if image_path:
            self.image_dropped.emit(image_path, self.group_name)
            event.acceptProposedAction()

    def add_widget(self, widget):
        self.widgets.append(widget)
        widget.on_delete.connect(self.remove_widget)
        self.rearrange()

    def remove_widget(self, widget):
        if widget in self.widgets:
            self.widgets.remove(widget); self.grid.removeWidget(widget); self.rearrange()                

    def rearrange(self):
        col_count = max(1, self.width() // 165) 
        for i, widget in enumerate(self.widgets):
            row, col = divmod(i, col_count)
            self.grid.addWidget(widget, row, col)

    def resizeEvent(self, event):
        self.rearrange(); super().resizeEvent(event)


# ==========================================
# 6. 主窗口逻辑
# ==========================================
class ImageGrouperApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Grouper AI - Core")
        self.resize(1280, 800)
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1f22; font-family: 'Segoe UI', 'Microsoft YaHei'; }
            QScrollBar:vertical { background: #2b2d31; width: 12px; }
            QScrollBar::handle:vertical { background: #4e5058; border-radius: 6px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #62656d; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)
        
        self.embeddings_cache = {}
        self.current_groups = {} 
        
        # AI进化记忆库
        self.memory_db = {}
        self.svm_clf = None
        self.global_rule_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "custom_ai_rules.pkl")
        self.load_global_rules()
        
        self.init_ui()
        self.check_hardware()

    def load_global_rules(self):
        if os.path.exists(self.global_rule_path):
            try:
                with open(self.global_rule_path, "rb") as f:
                    data = pickle.load(f)
                    self.memory_db = data.get("memory_db", {})
                    self.svm_clf = data.get("clf", None)
            except Exception as e: print("记忆库读取失败:", e)

    def init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ====== 侧边栏 ======
        sidebar = QFrame()
        sidebar.setFixedWidth(340)
        sidebar.setStyleSheet("background-color: #2b2d31; border-right: 1px solid #1e1f22;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(24, 28, 24, 28)
        side_layout.setSpacing(18)

        title = QLabel("AI 图像分组引擎")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet("color: #F2F3F5; border: none;")
        side_layout.addWidget(title)

        self.btn_select_dir = QPushButton("📁 浏览目标文件夹")
        self.btn_select_dir.setFixedHeight(44)
        self.btn_select_dir.setStyleSheet("QPushButton { background-color: #5865F2; color: white; border-radius: 6px; font-size: 14px; font-weight: bold; } QPushButton:hover { background-color: #4752C4; }")
        self.btn_select_dir.clicked.connect(self.select_directory)
        side_layout.addWidget(self.btn_select_dir)
        
        self.lbl_dir = QLabel("未选择文件夹")
        self.lbl_dir.setStyleSheet("color: #949BA4; font-size: 12px; border: none;")
        side_layout.addWidget(self.lbl_dir)

        combo_style = "QComboBox { padding: 8px; background: #1e1f22; border: 1px solid #383a40; border-radius: 6px; color: #DBDEE1; }"
        
        side_layout.addWidget(self.create_label("推理引擎 (Backend):"))
        self.combo_backend = QComboBox()
        self.combo_backend.addItems(["OpenVINO (Intel CPU 最优)", "PyTorch (CPU)", "PyTorch (CUDA)"])
        self.combo_backend.setStyleSheet(combo_style)
        self.combo_backend.currentIndexChanged.connect(self.toggle_ov_selector)
        side_layout.addWidget(self.combo_backend)

        # OpenVINO 路径选择UI (按需显示)
        self.ov_widget = QWidget()
        ov_layout = QVBoxLayout(self.ov_widget)
        ov_layout.setContentsMargins(0,0,0,0)
        ov_layout.addWidget(self.create_label("OpenVINO 模型路径 (.xml):"))
        ov_hbox = QHBoxLayout()
        self.inp_ov_path = QLineEdit()
        self.inp_ov_path.setPlaceholderText("选择或输入 openvino_model.xml 路径")
        self.inp_ov_path.setStyleSheet("QLineEdit { padding: 6px; background: #1e1f22; color: white; border: 1px solid #383a40; border-radius: 4px; }")
        btn_ov_browse = QPushButton("...")
        btn_ov_browse.setFixedSize(30, 30)
        btn_ov_browse.setStyleSheet("QPushButton { background-color: #383a40; color: white; border-radius: 4px; }")
        btn_ov_browse.clicked.connect(self.browse_ov_model)
        ov_hbox.addWidget(self.inp_ov_path)
        ov_hbox.addWidget(btn_ov_browse)
        ov_layout.addLayout(ov_hbox)
        side_layout.addWidget(self.ov_widget)

        self.btn_preproc = QPushButton("1. 提取全量特征")
        self.btn_preproc.setFixedHeight(40)
        self.btn_preproc.setStyleSheet("QPushButton { background-color: #383a40; color: #DBDEE1; border-radius: 6px; font-weight: bold; } QPushButton:hover { background-color: #404249; color: white; } QPushButton:disabled { background-color: #2b2d31; color: #5c5e66; border: 1px solid #383a40; }")
        self.btn_preproc.clicked.connect(self.run_preprocessing)
        side_layout.addWidget(self.btn_preproc)

        side_layout.addWidget(self.create_label("分组模式:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["准分类模式 (Text-Guided)", "AI 发现模式 (Auto-Cluster)", "我的专属规则 (SVM进化)"])
        self.combo_mode.setStyleSheet(combo_style)
        self.combo_mode.currentIndexChanged.connect(self.switch_mode_ui)
        side_layout.addWidget(self.combo_mode)
        
        self.stack_mode = QStackedWidget()
        
        # Page 0: Text
        page_text = QWidget(); l_text = QVBoxLayout(page_text); l_text.setContentsMargins(0, 0, 0, 0)
        self.inp_tags = QLineEdit()
        self.inp_tags.setPlaceholderText("如: 猫, 海滩, 建筑...")
        self.inp_tags.setStyleSheet("QLineEdit { padding: 10px; background: #1e1f22; color: white; border: 1px solid #383a40; border-radius: 6px; }")
        l_text.addWidget(self.create_label("目标类别 (逗号分隔):"))
        l_text.addWidget(self.inp_tags)
        self.stack_mode.addWidget(page_text)

        # Page 1: AI
        page_ai = QWidget(); l_ai = QVBoxLayout(page_ai); l_ai.setContentsMargins(0, 0, 0, 0)
        self.combo_eps = QComboBox()
        self.combo_eps.addItems(["细粒度", "平衡 (推荐)", "粗粒度"])
        self.combo_eps.setCurrentIndex(1)
        self.combo_eps.setStyleSheet(combo_style)
        l_ai.addWidget(self.create_label("聚类灵敏度 (DBSCAN):"))
        l_ai.addWidget(self.combo_eps)
        self.stack_mode.addWidget(page_ai)
        
        # Page 2: SVM
        page_svm = QWidget(); l_svm = QVBoxLayout(page_svm); l_svm.setContentsMargins(0, 0, 0, 0)
        svm_hbox = QHBoxLayout()
        self.btn_learn = QPushButton("🧠 吸收经验并进化")
        self.btn_learn.setFixedHeight(36)
        self.btn_learn.setStyleSheet("QPushButton { background-color: #E67E22; color: white; border-radius: 4px; font-weight: bold; } QPushButton:hover { background-color: #D35400; }")
        self.btn_learn.clicked.connect(self.learn_current_groups)
        
        self.btn_import_rule = QPushButton("📥 导入")
        self.btn_import_rule.setFixedSize(50, 36)
        self.btn_import_rule.setStyleSheet("QPushButton { background-color: #383a40; color: white; border-radius: 4px; }")
        self.btn_import_rule.clicked.connect(self.import_rules)
        
        # 🟢 新增：洗脑后悔药按钮
        self.btn_clear_mem = QPushButton("🧹 洗脑")
        self.btn_clear_mem.setFixedSize(50, 36)
        self.btn_clear_mem.setStyleSheet("QPushButton { background-color: #DA373C; color: white; border-radius: 4px; }")
        self.btn_clear_mem.clicked.connect(self.clear_memory)
        
        svm_hbox.addWidget(self.btn_learn)
        svm_hbox.addWidget(self.btn_import_rule)
        svm_hbox.addWidget(self.btn_clear_mem) # 🟢 加入布局
        l_svm.addWidget(self.create_label("持续学习与协同:"))
        l_svm.addLayout(svm_hbox)
        
        self.lbl_svm_info = QLabel(f"当前记忆体量: {len(self.memory_db)} 张特征样本")
        self.lbl_svm_info.setStyleSheet("color: #23A559; font-size: 11px;")
        l_svm.addWidget(self.lbl_svm_info)
        self.stack_mode.addWidget(page_svm)

        side_layout.addWidget(self.stack_mode)

        self.btn_group = QPushButton("2. 执行 AI 分组")
        self.btn_group.setFixedHeight(48)
        self.btn_group.setStyleSheet("QPushButton { background-color: #23A559; color: white; border-radius: 6px; font-weight: bold; font-size: 14px; } QPushButton:hover { background-color: #1D8749; } QPushButton:disabled { background-color: #2b2d31; color: #5c5e66; }")
        self.btn_group.clicked.connect(self.run_grouping)
        side_layout.addWidget(self.btn_group)
        
        side_layout.addWidget(self.create_label("快捷操作:"))
        self.btn_view_trash = QPushButton("🗑️ 垃圾回收站")
        self.btn_view_trash.setFixedHeight(34)
        self.btn_view_trash.setStyleSheet("QPushButton { background-color: #383a40; color: #DBDEE1; border-radius: 6px; }")
        self.btn_view_trash.clicked.connect(self.view_trash)
        side_layout.addWidget(self.btn_view_trash)

        side_layout.addStretch()
        
        # 底部状态布局：Label + 悬浮中止图标
        status_layout = QHBoxLayout()
        self.lbl_status = QLabel("就绪")
        self.lbl_status.setStyleSheet("color: #949BA4; font-size: 12px; font-weight: bold;")
        
        self.btn_stop_icon = QPushButton("⏹️")
        self.btn_stop_icon.setFixedSize(22, 22)
        self.btn_stop_icon.setToolTip("强行中止任务")
        self.btn_stop_icon.setStyleSheet("QPushButton { background-color: #DA373C; color: white; border-radius: 11px; font-weight: bold; border: none; } QPushButton:hover { background-color: #A1282D; }")
        self.btn_stop_icon.clicked.connect(self.stop_worker)
        self.btn_stop_icon.hide()
        
        status_layout.addWidget(self.lbl_status)
        status_layout.addStretch()
        status_layout.addWidget(self.btn_stop_icon)
        side_layout.addLayout(status_layout)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setFixedHeight(6)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setStyleSheet("QProgressBar { background-color: #1e1f22; border: none; border-radius: 3px; } QProgressBar::chunk { background-color: #5865F2; }")
        side_layout.addWidget(self.progress_bar)

        main_layout.addWidget(sidebar)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; background-color: #1e1f22; }")
        self.viewport = QWidget()
        # 🟢 新增下面这一行，强制锁定内部面板为深色，防止浅色穿透
        self.viewport.setStyleSheet("background-color: #18191c;")
        self.view_layout = QVBoxLayout(self.viewport)
        self.view_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.view_layout.setContentsMargins(30, 30, 30, 30)
        self.view_layout.setSpacing(25)
        self.scroll_area.setWidget(self.viewport)
        main_layout.addWidget(self.scroll_area)

        self.combo_mode.setCurrentIndex(1)
        self.toggle_ov_selector()

    def create_label(self, text):
        lbl = QLabel(text)
        lbl.setStyleSheet("color: #B5BAC1; font-size: 12px; font-weight: bold;")
        return lbl
        
    def toggle_ov_selector(self):
        is_ov = "OpenVINO" in self.combo_backend.currentText()
        self.ov_widget.setVisible(is_ov)

    def browse_ov_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择 OpenVINO 模型", "", "XML 模型文件 (*.xml)")
        if path: self.inp_ov_path.setText(path)

    def stop_worker(self):
        if hasattr(self, 'worker') and self.worker.isRunning():
            self.worker.cancel()
            self.btn_stop_icon.setEnabled(False)
            self.lbl_status.setText("正在中止...")

    def check_hardware(self):
        if not HAS_SKLEARN:
            model_mode = self.combo_mode.model()
            model_mode.item(1).setEnabled(False)
            model_mode.item(2).setEnabled(False)
            self.combo_mode.setItemText(1, "AI 发现 (未安装 sklearn)")
            self.combo_mode.setItemText(2, "规则进化 (未安装 sklearn)")
            self.combo_mode.setCurrentIndex(0)

    def refresh_directory_state(self):
        if not hasattr(self, 'current_folder') or not self.current_folder: return
            
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        self.target_files = [os.path.join(self.current_folder, f) for f in os.listdir(self.current_folder) if f.lower().endswith(valid_exts)]
        
        keys_to_remove = [k for k in self.embeddings_cache.keys() if k not in self.target_files]
        for k in keys_to_remove: del self.embeddings_cache[k]
            
        # 强制复写 pkl，保证硬盘数据实时性
        if keys_to_remove:
            try:
                with open(os.path.join(self.current_folder, ".embeddings_cache.pkl"), "wb") as f:
                    pickle.dump(self.embeddings_cache, f)
            except Exception: pass
            
        self.missing_in_cache = [f for f in self.target_files if f not in self.embeddings_cache]

        if self.missing_in_cache:
            self.btn_preproc.setText(f"1. 提取新特征 ({len(self.missing_in_cache)}张增量)")
            self.btn_preproc.setStyleSheet("QPushButton { background-color: #5865F2; color: white; border-radius: 6px; font-weight: bold; }")
            self.btn_preproc.setEnabled(True)
        else:
            self.btn_preproc.setText("特征已最新 ✓ (可直接分组)")
            self.btn_preproc.setStyleSheet("QPushButton { background-color: #2b2d31; color: #23A559; border: 1px solid #23A559; border-radius: 6px; font-weight: bold; }")
            self.btn_preproc.setEnabled(False)

    def select_directory(self):
        folder = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if not folder: return
        self.current_folder = folder
        self.lbl_dir.setText(folder if len(folder) < 35 else "..." + folder[-32:])
        
        cache_file = os.path.join(self.current_folder, ".embeddings_cache.pkl")
        self.embeddings_cache = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f: self.embeddings_cache = pickle.load(f)
            except Exception as e: print("缓存读取失败:", e)
            
        self.refresh_directory_state()
        if not self.target_files: QMessageBox.warning(self, "提示", "没有找到支持的图片！")

    @pyqtSlot(int)
    def switch_mode_ui(self, index): self.stack_mode.setCurrentIndex(index)

    def run_preprocessing(self):
        self.refresh_directory_state()
        self.btn_preproc.setEnabled(False)
        self.btn_stop_icon.setEnabled(True)
        self.btn_stop_icon.show()
        self.progress_bar.setRange(0, 0) 
        
        params = {"images": getattr(self, 'missing_in_cache', []), "backend": self.combo_backend.currentText()}
        if "OpenVINO" in params["backend"]: params["ov_path"] = self.inp_ov_path.text()
        
        self.worker = Worker("PREPROC", params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    def update_progress(self, value, msg):
        if self.progress_bar.maximum() == 0 and value > 0: self.progress_bar.setRange(0, 100)
        self.lbl_status.setText(msg)
        self.progress_bar.setValue(value)

    def run_grouping(self):
        self.refresh_directory_state() 
        if not self.embeddings_cache: return QMessageBox.warning(self, "提示", "请先提取特征")
        
        self.btn_group.setEnabled(False)
        self.btn_stop_icon.setEnabled(True)
        self.btn_stop_icon.show()
        self.progress_bar.setRange(0, 0)
        
        modes = ["text", "ai", "svm"]
        mode = modes[self.combo_mode.currentIndex()]
        params = {"mode": mode, "embeddings": self.embeddings_cache, "backend": self.combo_backend.currentText()}
        
        if mode == "text": params["tags"] = self.inp_tags.text()
        elif mode == "ai": params["eps_level"] = self.combo_eps.currentIndex()
        elif mode == "svm": params["svm_clf"] = self.svm_clf
        if "OpenVINO" in params["backend"]: params["ov_path"] = self.inp_ov_path.text()

        self.worker = Worker("GROUP", params)
        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.error.connect(self.on_worker_error)
        self.worker.start()

    @pyqtSlot(str)
    def on_worker_error(self, err_msg):
        self.btn_stop_icon.hide()
        self.progress_bar.setRange(0, 100); self.progress_bar.setValue(0)
        self.lbl_status.setText("执行异常中止")
        self.btn_preproc.setEnabled(True)
        self.btn_group.setEnabled(True)
        QMessageBox.critical(self, "错误", str(err_msg))

    @pyqtSlot(dict)
    def on_worker_finished(self, result):
        self.btn_stop_icon.hide()
        self.lbl_status.setText("任务已完成 ✓")
        
        if result["type"] == "preproc":
            self.embeddings_cache.update(result["data"])
            self.missing_in_cache = []
            self.btn_preproc.setText("特征已最新 ✓")
            self.btn_preproc.setStyleSheet("QPushButton { background-color: #2b2d31; color: #23A559; border: 1px solid #23A559; border-radius: 6px; font-weight: bold; }")
            self.progress_bar.setValue(100)
            
            if hasattr(self, 'current_folder'):
                try:
                    with open(os.path.join(self.current_folder, ".embeddings_cache.pkl"), "wb") as f:
                        pickle.dump(self.embeddings_cache, f)
                except Exception: pass
            
        elif result["type"] == "group":
            self.btn_group.setEnabled(True)
            self.current_groups = result["data"] 
            self.render_grouped_results(self.current_groups)
            self.progress_bar.setValue(100)

   # =============== 进化与导入逻辑 ===============
    def learn_current_groups(self):
        if not self.current_groups:
            return QMessageBox.warning(self, "提示", "当前还没有分类结果可供学习！\n请先使用准分类或拖拽分好组。")
            
        # 🟢 1. 预统计即将学习的有效数据，给用户确认
        learn_summary = []
        valid_count = 0
        for group_name, images in self.current_groups.items():
            if any(x in group_name for x in ["其他", "未归类"]): continue
            if images:
                learn_summary.append(f" - {group_name}: {len(images)} 张")
                valid_count += len(images)
                
        if valid_count == 0:
            return QMessageBox.warning(self, "提示", "当前没有有效的标准分组可供学习！")

        # 🟢 2. 强制二次确认弹窗
        reply = QMessageBox.question(
            self, "学习前人工核对确认", 
            f"即将把以下 {valid_count} 张图片的特征吸纳入大脑：\n" + 
            "\n".join(learn_summary) + 
            "\n\n⚠️ 警告：请确保您已经人工检查过上述图片，如果 AI 分错了，请【取消】并用鼠标把错图拖到正确的组里再来学习！\n\n确认无误并开始进化吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.No:
            return

        # 3. 开始录入记忆
        for group_name, images in self.current_groups.items():
            if any(x in group_name for x in ["其他", "未归类"]): continue
            for img_path in images:
                if img_path in self.embeddings_cache:
                    base_name = os.path.basename(img_path)
                    self.memory_db[base_name] = (self.embeddings_cache[img_path], group_name)
                    
        X = [item[0] for item in self.memory_db.values()]
        y = [item[1] for item in self.memory_db.values()]
        
        if len(set(y)) < 2:
            return QMessageBox.warning(self, "提示", "有效学习组别少于 2 个！\nAI至少需要知道两个事物之间的区别。")
            
        try:
            self.svm_clf = SVC(kernel='linear', class_weight='balanced')
            self.svm_clf.fit(X, y)
            
            with open(self.global_rule_path, "wb") as f:
                pickle.dump({"memory_db": self.memory_db, "clf": self.svm_clf}, f)
                
            self.lbl_svm_info.setText(f"当前记忆体量: {len(self.memory_db)} 张特征样本")
            QMessageBox.information(self, "进化成功", f"AI 已吸纳当前分类经验！\n当前大脑累计包含 {len(self.memory_db)} 张特征样本。")
        except Exception as e:
            QMessageBox.critical(self, "训练失败", str(e))

    # 🟢 新增：洗脑重置方法
    def clear_memory(self):
        reply = QMessageBox.warning(
            self, "危险操作", 
            "这将彻底清空 AI 当前积累的所有分类记忆（记忆体量归零）。\n确定要给 AI 洗脑吗？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self.memory_db = {}
            self.svm_clf = None
            if os.path.exists(self.global_rule_path):
                try:
                    os.remove(self.global_rule_path)
                except Exception as e:
                    print("删除记忆库失败:", e)
            self.lbl_svm_info.setText("当前记忆体量: 0 张特征样本")
            QMessageBox.information(self, "已清空", "AI 大脑已格式化，随时准备重新学习！")

    def import_rules(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择记忆库文件", "", "Pickle 文件 (*.pkl)")
        if not path: return
        try:
            with open(path, "rb") as f: imported_data = pickle.load(f)
            if "memory_db" not in imported_data: raise ValueError("格式不支持，请选择本工具导出的 pkl")
            
            # 融合记忆：合并两个字典
            self.memory_db.update(imported_data["memory_db"])
            
            X = [item[0] for item in self.memory_db.values()]
            y = [item[1] for item in self.memory_db.values()]
            
            self.svm_clf = SVC(kernel='linear', class_weight='balanced')
            self.svm_clf.fit(X, y)
            
            with open(self.global_rule_path, "wb") as f:
                pickle.dump({"memory_db": self.memory_db, "clf": self.svm_clf}, f)
                
            self.lbl_svm_info.setText(f"当前记忆体量: {len(self.memory_db)} 张特征样本")
            QMessageBox.information(self, "融合成功", "外部记忆已与当前大脑融合完毕，请切至专属规则模式享用！")
        except Exception as e:
            QMessageBox.critical(self, "导入失败", str(e))
    # ============================================

    @pyqtSlot(str, str)
    def handle_image_drop(self, image_path, target_group):
        old_group = None
        for g, imgs in self.current_groups.items():
            if image_path in imgs:
                old_group = g; break
                
        if old_group and old_group != target_group:
            self.current_groups[old_group].remove(image_path)
            self.current_groups[target_group].append(image_path)
            self.render_grouped_results(self.current_groups)

    def render_grouped_results(self, groups_dict):
        for i in reversed(range(self.view_layout.count())): 
            w = self.view_layout.itemAt(i).widget()
            if w: w.setParent(None); w.deleteLater()

        for group_name, images in groups_dict.items():
            if not images: continue
            
            group_box = QGroupBox(f"{group_name} ({len(images)} 张)")
            group_box.setStyleSheet("""
                QGroupBox { 
                    border: 1px solid #2b2d31; 
                    border-radius: 8px; 
                    margin-top: 18px; 
                    background-color: #1e1f22; /* 分组块的底色 */
                    font-weight: bold; 
                    color: #DBDEE1; 
                }
                QGroupBox::title { subcontrol-origin: margin; left: 20px; padding: 0 8px; color: #5865F2; }
            """)
            
            box_layout = QVBoxLayout(group_box)
            box_layout.setContentsMargins(15, 25, 15, 15)
            
            grid = ResponsiveGridWidget(group_name)
            grid.image_dropped.connect(self.handle_image_drop)
            
            for img_path in images:
                card = ImageCard(img_path)
                card.double_clicked.connect(lambda path=img_path: ImageViewerDialog(path, self).exec())
                # 若被删除进回收站，毫秒级要求主线程复写对账
                card.on_delete.connect(lambda c: self.refresh_directory_state())
                grid.add_widget(card)
                
            box_layout.addWidget(grid)
            self.view_layout.addWidget(group_box)
            
        self.view_layout.addStretch()

    def view_trash(self):
        if not hasattr(self, 'current_folder') or not self.current_folder:
            return QMessageBox.warning(self, "提示", "请先选择目标文件夹！")
        
        trash_dir = os.path.join(self.current_folder, "Trash")
        if not os.path.exists(trash_dir) or not os.listdir(trash_dir):
            return QMessageBox.information(self, "提示", "回收站是空的。")
            
        dialog = QDialog(self)
        dialog.setWindowTitle("🗑️ 回收站")
        dialog.resize(800, 600)
        dialog.setStyleSheet("background-color: #1e1f22; color: white;")
        layout = QVBoxLayout(dialog)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("border: none;")
        viewport = QWidget()
        view_layout = QVBoxLayout(viewport)
        
        grid = ResponsiveGridWidget()
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        for f in os.listdir(trash_dir):
            if f.lower().endswith(valid_exts):
                card = ImageCard(os.path.join(trash_dir, f), is_trash_mode=True)
                card.on_delete.connect(lambda c: self.refresh_directory_state())
                grid.add_widget(card)
                
        view_layout.addWidget(grid)
        view_layout.addStretch()
        scroll.setWidget(viewport)
        layout.addWidget(scroll)
        dialog.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    font = app.font()
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)
    window = ImageGrouperApp()
    window.show()
    sys.exit(app.exec())