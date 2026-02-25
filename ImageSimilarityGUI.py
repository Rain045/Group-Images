import os
import shutil
import sys
import numpy as np

# PyQt6 模組
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QScrollArea, QGroupBox, 
                             QMessageBox, QRadioButton, QComboBox, QStackedWidget,
                             QProgressBar, QStyle, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap

# ImageHash
from PIL import Image
import imagehash

# OpenCV
import cv2

# AI / Deep Learning
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity


# --- 自訂可點擊的圖片標籤 (用於彈出大圖) ---
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)
    
    def __init__(self, img_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_path = img_path
        self.setCursor(Qt.CursorShape.PointingHandCursor) # 滑鼠移過去變手型
        self.setStyleSheet("border: 1px solid #DDDDDD; padding: 2px; background-color: white;")

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.img_path)


# --- 整合版掃描執行緒 ---
class ScannerThread(QThread):
    progress_update = pyqtSignal(str)
    progress_percent = pyqtSignal(int) 
    scan_finished = pyqtSignal(list, list)

    def __init__(self, src_dir, algo_mode, params):
        super().__init__()
        self.src_dir = src_dir
        self.algo_mode = algo_mode  
        self.params = params        

    def run(self):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
        self.all_files = [f for f in os.listdir(self.src_dir) if f.lower().endswith(valid_exts)]
        
        if not self.all_files:
            self.progress_update.emit("找不到支援的圖片檔案。")
            self.progress_percent.emit(0)
            self.scan_finished.emit([], [])
            return

        self.progress_percent.emit(5)

        if self.algo_mode == 'imagehash':
            self._run_imagehash()
        elif self.algo_mode == 'opencv':
            self._run_opencv()
        elif self.algo_mode == 'ai':
            self._run_ai()

    def _run_imagehash(self):
        struct_thresh = self.params.get('struct', 10)
        color_thresh = self.params.get('color', 10)
        
        hashes_structure = {}
        hashes_color = {}
        total = len(self.all_files)
        
        for idx, filename in enumerate(self.all_files):
            if idx % max(1, total // 20) == 0:
                self.progress_update.emit(f"計算指紋中... ({idx}/{total})")
                self.progress_percent.emit(5 + int((idx / total) * 45))
            
            path = os.path.join(self.src_dir, filename)
            try:
                with Image.open(path) as img:
                    hashes_structure[filename] = imagehash.phash(img) 
                    hashes_color[filename] = imagehash.colorhash(img)
            except Exception: pass

        self.progress_update.emit("正在進行雙重交叉比對...")
        grouped_files = set()
        groups = []

        for i, file1 in enumerate(self.all_files):
            if i % max(1, total // 20) == 0:
                self.progress_percent.emit(50 + int((i / total) * 50))

            if file1 in grouped_files or file1 not in hashes_structure: continue
            current_group = [file1]
            for file2 in self.all_files[i+1:]:
                if file2 in grouped_files or file2 not in hashes_structure: continue
                
                if (hashes_structure[file1] - hashes_structure[file2] <= struct_thresh and 
                    hashes_color[file1] - hashes_color[file2] <= color_thresh):
                    current_group.append(file2)
            
            if len(current_group) > 1:
                groups.append(current_group)
                grouped_files.update(current_group)

        single_files = [f for f in self.all_files if f not in grouped_files and f in hashes_structure]
        self.progress_update.emit(f"掃描完成！找到 {len(groups)} 組相似圖片。")
        self.progress_percent.emit(100)
        self.scan_finished.emit(groups, single_files)

    def _run_opencv(self):
        match_thresh = self.params.get('match', 50)
        orb = cv2.ORB_create(nfeatures=500)
        descriptors_dict = {}
        total = len(self.all_files)
        
        for idx, filename in enumerate(self.all_files):
            if idx % max(1, total // 20) == 0:
                self.progress_update.emit(f"提取特徵點... ({idx}/{total})")
                self.progress_percent.emit(5 + int((idx / total) * 45))
            path = os.path.join(self.src_dir, filename)
            try:
                img_data = np.fromfile(path, dtype=np.uint8)
                img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img, (500, 500))
                kp, des = orb.detectAndCompute(img, None)
                if des is not None:
                    descriptors_dict[filename] = des
            except Exception: pass

        self.progress_update.emit("特徵點暴力匹配中...")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        grouped_files = set()
        groups = []
        file_list = list(descriptors_dict.keys())
        total_files = len(file_list)
        
        for i, file1 in enumerate(file_list):
            if i % max(1, total_files // 20) == 0: 
                self.progress_update.emit(f"比對進度: {i}/{total_files}...")
                self.progress_percent.emit(50 + int((i / total_files) * 50))

            if file1 in grouped_files: continue
            current_group = [file1]
            des1 = descriptors_dict[file1]
            
            for file2 in file_list[i+1:]:
                if file2 in grouped_files: continue
                try:
                    matches = bf.match(des1, descriptors_dict[file2])
                    good_matches = [m for m in matches if m.distance < 50]
                    if len(good_matches) >= match_thresh:
                        current_group.append(file2)
                except Exception: pass
            
            if len(current_group) > 1:
                groups.append(current_group)
                grouped_files.update(current_group)

        single_files = [f for f in self.all_files if f not in grouped_files and f in descriptors_dict]
        self.progress_update.emit(f"掃描完成！找到 {len(groups)} 組相似圖片。")
        self.progress_percent.emit(100)
        self.scan_finished.emit(groups, single_files)

    def _run_ai(self):
        sim_thresh = self.params.get('sim', 0.90)
        model_name = self.params.get('ai_model', 'mobilenet_v2')
        self.progress_update.emit(f"載入 {model_name} 模型中...")
        self.progress_percent.emit(10)
        
        try:
            device = torch.device("cpu")
            if model_name == 'mobilenet_v2':
                from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
                model = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
                model.classifier = torch.nn.Identity()
            elif model_name == 'resnet18':
                from torchvision.models import resnet18, ResNet18_Weights
                model = resnet18(weights=ResNet18_Weights.DEFAULT)
                model.fc = torch.nn.Identity() 
            elif model_name == 'efficientnet_b0':
                from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
                model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
                model.classifier = torch.nn.Identity()
            # --- 新增中型模型 ---
            elif model_name == 'resnet50':
                from torchvision.models import resnet50, ResNet50_Weights
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
                model.fc = torch.nn.Identity() 
            elif model_name == 'efficientnet_b2':
                from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
                model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
                model.classifier = torch.nn.Identity()
            # --------------------
                
            model.eval()
            model = model.to(device)
            preprocess = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        except Exception as e:
            self.progress_update.emit(f"AI 模型載入失敗: {e}")
            self.progress_percent.emit(0)
            self.scan_finished.emit([], [])
            return

        features = {}
        total = len(self.all_files)
        
        for idx, filename in enumerate(self.all_files):
            if idx % max(1, total // 20) == 0: 
                self.progress_update.emit(f"提取語義特徵... ({idx}/{total})")
                self.progress_percent.emit(20 + int((idx / total) * 40)) 
            path = os.path.join(self.src_dir, filename)
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = preprocess(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    vec = model(img_tensor).numpy().flatten()
                features[filename] = vec
            except Exception: pass

        self.progress_update.emit("計算高維度餘弦相似度...")
        grouped_files = set()
        groups = []
        file_list = list(features.keys())
        total_files = len(file_list)
        
        for i, file1 in enumerate(file_list):
            if i % max(1, total_files // 20) == 0:
                self.progress_percent.emit(60 + int((i / total_files) * 40))

            if file1 in grouped_files: continue
            current_group = [file1]
            vec1 = features[file1].reshape(1, -1)
            
            for file2 in file_list[i+1:]:
                if file2 in grouped_files: continue
                vec2 = features[file2].reshape(1, -1)
                sim = cosine_similarity(vec1, vec2)[0][0]
                if sim >= sim_thresh:
                    current_group.append(file2)
            
            if len(current_group) > 1:
                groups.append(current_group)
                grouped_files.update(current_group)

        single_files = [f for f in self.all_files if f not in grouped_files and f in features]
        self.progress_update.emit(f"掃描完成！找到 {len(groups)} 組相似圖片。")
        self.progress_percent.emit(100)
        self.scan_finished.emit(groups, single_files)


# --- 主圖形介面 (左右分欄設計) ---
class ImageGrouperApp(QWidget):
    def __init__(self):
        super().__init__()
        self.groups_data = []      
        self.single_files = []     
        self.group_widgets = []    
        self.initUI()
        self.apply_stylesheet()

    def apply_stylesheet(self):
        style = """
        QWidget { font-family: "Segoe UI", "Microsoft JhengHei", sans-serif; font-size: 10pt; color: #333333; }
        QGroupBox { font-weight: bold; border: 1px solid #CCCCCC; border-radius: 6px; margin-top: 10px; background-color: #FAFAFA; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; color: #005A9E; }
        QLineEdit { padding: 4px; border: 1px solid #BDBDBD; border-radius: 4px; background-color: #FFFFFF; }
        QLineEdit:focus { border: 1px solid #0078D7; }
        QPushButton { padding: 5px 10px; border-radius: 4px; background-color: #E1E1E1; border: 1px solid #ADADAD; }
        QPushButton:hover { background-color: #D4D4D4; }
        QPushButton#primaryBtn { background-color: #0078D7; color: white; border: none; font-weight: bold; padding: 8px; }
        QPushButton#primaryBtn:hover { background-color: #005A9E; }
        QPushButton#primaryBtn:disabled { background-color: #A0C5E8; }
        QPushButton#actionBtn { background-color: #107C41; color: white; border: none; font-weight: bold; padding: 8px; }
        QPushButton#actionBtn:hover { background-color: #0B5A2F; }
        QPushButton#actionBtn:disabled { background-color: #8CC2A0; }
        QComboBox, QSpinBox, QDoubleSpinBox { padding: 4px; border: 1px solid #BDBDBD; border-radius: 4px; }
        QProgressBar { border: 1px solid #CCCCCC; border-radius: 4px; text-align: center; color: black; }
        QProgressBar::chunk { background-color: #0078D7; width: 10px; }
        QScrollArea { border: 1px solid #CCCCCC; background-color: #EEEEEE; border-radius: 6px; }
        """
        self.setStyleSheet(style)

    def initUI(self):
        self.setWindowTitle('Image Similarity Pro')
        self.resize(1100, 750)
        
        # 核心佈局：左右分欄
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # ==================== 左側：緊湊功能區 ====================
        left_panel = QWidget()
        left_panel.setMaximumWidth(320) # 限制側邊欄最大寬度
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        dir_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)

        # 1. 路徑設定區
        path_group = QGroupBox("📁 檔案目錄設定")
        path_layout = QVBoxLayout()
        path_layout.addWidget(QLabel("輸入資料夾:"))
        in_layout = QHBoxLayout()
        self.input_entry = QLineEdit()
        in_layout.addWidget(self.input_entry)
        btn_in = QPushButton()
        btn_in.setIcon(dir_icon)
        btn_in.clicked.connect(self.browse_input)
        in_layout.addWidget(btn_in)
        path_layout.addLayout(in_layout)

        path_layout.addWidget(QLabel("輸出資料夾:"))
        out_layout = QHBoxLayout()
        self.output_entry = QLineEdit()
        out_layout.addWidget(self.output_entry)
        btn_out = QPushButton()
        btn_out.setIcon(dir_icon)
        btn_out.clicked.connect(self.browse_output)
        out_layout.addWidget(btn_out)
        path_layout.addLayout(out_layout)
        path_group.setLayout(path_layout)
        left_layout.addWidget(path_group)

        # 2. 演算法與參數區
        algo_group = QGroupBox("⚙️ 引擎與參數")
        algo_layout = QVBoxLayout()
        algo_layout.addWidget(QLabel("分析引擎:"))
        self.combo_algo = QComboBox()
        self.combo_algo.addItem("🧠 AI 語義特徵", 'ai') 
        self.combo_algo.addItem("🧬 OpenCV ORB", 'opencv')
        self.combo_algo.addItem("⚡ 雙重感知哈希", 'imagehash')
        self.combo_algo.currentIndexChanged.connect(self.change_algo_params)
        algo_layout.addWidget(self.combo_algo)

        self.param_stack = QStackedWidget()
        
        # AI 面板
        ai_widget = QWidget()
        ai_layout = QVBoxLayout(ai_widget)
        ai_layout.setContentsMargins(0, 5, 0, 0)
        ai_layout.addWidget(QLabel("AI 模型選擇:"))
        self.combo_ai_model = QComboBox()
        self.combo_ai_model.addItems([
            "mobilenet_v2", 
            "resnet18", 
            "efficientnet_b0", 
            "resnet50",         # 新增中型模型
            "efficientnet_b2"   # 新增中型模型
        ])
        ai_layout.addWidget(self.combo_ai_model)
        
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(QLabel("相似度 (0.5~1.0):"))
        self.ai_sim_spin = QDoubleSpinBox()
        self.ai_sim_spin.setRange(0.50, 1.00)
        self.ai_sim_spin.setSingleStep(0.01)
        self.ai_sim_spin.setValue(0.90)
        sim_layout.addWidget(self.ai_sim_spin)
        ai_layout.addLayout(sim_layout)
        self.param_stack.addWidget(ai_widget)

        # OpenCV 面板
        cv_widget = QWidget()
        cv_layout = QHBoxLayout(cv_widget)
        cv_layout.setContentsMargins(0, 5, 0, 0)
        cv_layout.addWidget(QLabel("最低匹配點:"))
        self.cv_match_spin = QSpinBox()
        self.cv_match_spin.setRange(10, 500)
        self.cv_match_spin.setValue(50)
        cv_layout.addWidget(self.cv_match_spin)
        self.param_stack.addWidget(cv_widget)

        # ImageHash 面板
        ih_widget = QWidget()
        ih_layout = QHBoxLayout(ih_widget)
        ih_layout.setContentsMargins(0, 5, 0, 0)
        ih_layout.addWidget(QLabel("結構:"))
        self.ih_struct_spin = QSpinBox()
        self.ih_struct_spin.setValue(10)
        ih_layout.addWidget(self.ih_struct_spin)
        ih_layout.addWidget(QLabel("色彩:"))
        self.ih_color_spin = QSpinBox()
        self.ih_color_spin.setValue(10)
        ih_layout.addWidget(self.ih_color_spin)
        self.param_stack.addWidget(ih_widget)

        algo_layout.addWidget(self.param_stack)
        algo_group.setLayout(algo_layout)
        left_layout.addWidget(algo_group)

        # 3. 操作與進度區
        exec_group = QGroupBox("🚀 操作與執行")
        exec_layout = QVBoxLayout()
        
        mode_layout = QHBoxLayout()
        self.radio_move = QRadioButton("移動")
        self.radio_copy = QRadioButton("複製")
        self.radio_move.setChecked(True)
        mode_layout.addWidget(self.radio_move)
        mode_layout.addWidget(self.radio_copy)
        exec_layout.addLayout(mode_layout)

        self.btn_scan = QPushButton("🔍 開始掃描與預覽")
        self.btn_scan.setObjectName("primaryBtn")
        self.btn_scan.clicked.connect(self.start_scan)
        exec_layout.addWidget(self.btn_scan)

        # 狀態與進度條移到左側下方
        self.lbl_status = QLabel("就緒。")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #005A9E; font-size: 9pt; margin-top: 5px;")
        exec_layout.addWidget(self.lbl_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        exec_layout.addWidget(self.progress_bar)

        self.btn_execute = QPushButton("✅ 確認執行分組")
        self.btn_execute.setObjectName("actionBtn")
        self.btn_execute.setEnabled(False)
        self.btn_execute.clicked.connect(self.execute_action)
        exec_layout.addWidget(self.btn_execute)

        exec_group.setLayout(exec_layout)
        left_layout.addWidget(exec_group)
        
        # 將左側元件推到上方
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ==================== 右側：巨大預覽顯示區 ====================
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        
        # stretch=1 讓右側佔據所有剩餘空間
        main_layout.addWidget(self.scroll_area, stretch=1) 

        self.setLayout(main_layout)

    def change_algo_params(self, index):
        self.param_stack.setCurrentIndex(index)

    def browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸入")
        if folder: self.input_entry.setText(folder)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸出")
        if folder: self.output_entry.setText(folder)

    def clear_scroll_area(self):
        for i in reversed(range(self.scroll_layout.count())): 
            widget = self.scroll_layout.itemAt(i).widget()
            if widget is not None: widget.deleteLater()
        self.group_widgets.clear()

    def start_scan(self):
        src_dir = self.input_entry.text().strip()
        if not src_dir or not os.path.exists(src_dir):
            QMessageBox.warning(self, "錯誤", "請選擇有效的輸入資料夾！")
            return

        self.clear_scroll_area()
        self.btn_scan.setEnabled(False)
        self.btn_execute.setEnabled(False)
        self.progress_bar.setValue(0)

        algo_mode = self.combo_algo.currentData()
        params = {}
        
        if algo_mode == 'ai':
            params['sim'] = self.ai_sim_spin.value()
            params['ai_model'] = self.combo_ai_model.currentText()
        elif algo_mode == 'opencv':
            params['match'] = self.cv_match_spin.value()
        elif algo_mode == 'imagehash':
            params['struct'] = self.ih_struct_spin.value()
            params['color'] = self.ih_color_spin.value()

        self.thread = ScannerThread(src_dir, algo_mode, params)
        self.thread.progress_update.connect(self.update_status)
        self.thread.progress_percent.connect(self.update_progress)
        self.thread.scan_finished.connect(self.on_scan_finished)
        self.thread.start()

    def update_status(self, text):
        self.lbl_status.setText(text)

    def update_progress(self, val):
        self.progress_bar.setValue(val)

    def on_scan_finished(self, groups, single_files):
        self.groups_data = groups
        self.single_files = single_files
        self.btn_scan.setEnabled(True)

        if not groups and not single_files:
            self.lbl_status.setText("沒有可處理的圖片。")
            return

        self.display_groups()
        self.btn_execute.setEnabled(True)
        self.lbl_status.setText(f"完成！共找到 {len(groups)} 組相似圖，{len(single_files)} 張獨立。")

    def show_full_image(self, img_path):
        """點擊縮圖時彈出大圖查看"""
        dialog = QDialog(self)
        dialog.setWindowTitle(os.path.basename(img_path))
        layout = QVBoxLayout()
        lbl = QLabel()
        pixmap = QPixmap(img_path)
        
        # 限制大圖最大不超過螢幕的 80%
        screen = QApplication.primaryScreen().geometry()
        max_w, max_h = int(screen.width() * 0.8), int(screen.height() * 0.8)
        if pixmap.width() > max_w or pixmap.height() > max_h:
            pixmap = pixmap.scaled(max_w, max_h, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            
        lbl.setPixmap(pixmap)
        layout.addWidget(lbl)
        dialog.setLayout(layout)
        dialog.exec()

    def display_groups(self):
        src_dir = self.input_entry.text().strip()
        
        for idx, group in enumerate(self.groups_data):
            group_box = QGroupBox(f"📂 分組 {idx + 1} (共 {len(group)} 張) - 勾選以保留分組")
            group_box.setCheckable(True)
            group_box.setChecked(True)
            group_box.setStyleSheet("QGroupBox { background-color: #FFFFFF; }")
            
            self.group_widgets.append({'box': group_box, 'files': group})
            
            # 使用自動換行的流式佈局，讓右側空間能塞入更多圖片
            img_layout = QHBoxLayout()
            img_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            
            for filename in group:
                path = os.path.join(src_dir, filename)
                # 使用我們剛剛自訂的可點擊 Label
                lbl_img = ClickableLabel(path) 
                lbl_img.clicked.connect(self.show_full_image) # 綁定點擊事件
                
                try:
                    # 預覽圖放大到 150x150，因為右側空間變大了
                    pixmap = QPixmap(path).scaled(150, 150, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    lbl_img.setPixmap(pixmap)
                except Exception:
                    lbl_img.setText("[預覽失敗]")
                
                img_layout.addWidget(lbl_img)
            
            # 這裡簡單加一個外層 Widget 讓水平圖片可以被良好包含
            scroll_widget = QWidget()
            scroll_widget.setLayout(img_layout)
            
            inner_scroll = QScrollArea()
            inner_scroll.setWidgetResizable(True)
            inner_scroll.setWidget(scroll_widget)
            inner_scroll.setFixedHeight(190) # 固定每組的高度，讓捲動更順暢
            inner_scroll.setStyleSheet("border: none;")

            box_layout = QVBoxLayout()
            box_layout.addWidget(inner_scroll)
            group_box.setLayout(box_layout)
            
            self.scroll_layout.addWidget(group_box)

    def execute_action(self):
        src = self.input_entry.text().strip()
        out = self.output_entry.text().strip()
        
        if not out:
            QMessageBox.warning(self, "錯誤", "請選擇輸出資料夾！")
            return
            
        if not os.path.exists(out):
            os.makedirs(out)

        is_copy_mode = self.radio_copy.isChecked()
        action_name = "複製" if is_copy_mode else "移動"
        file_operation = shutil.copy2 if is_copy_mode else shutil.move

        self.btn_execute.setEnabled(False)
        self.lbl_status.setText(f"正在{action_name}檔案中...")
        self.progress_bar.setValue(0)
        QApplication.processEvents()

        group_count = 0
        final_single_files = list(self.single_files)
        total_groups = len(self.group_widgets)

        for i, widget_data in enumerate(self.group_widgets):
            box = widget_data['box']
            files = widget_data['files']
            
            if box.isChecked():
                group_count += 1
                group_folder = os.path.join(out, f"Group_{group_count}")
                os.makedirs(group_folder, exist_ok=True)
                for f in files:
                    src_path = os.path.join(src, f)
                    if os.path.exists(src_path):
                        file_operation(src_path, os.path.join(group_folder, f))
            else:
                final_single_files.extend(files)
                
            self.progress_bar.setValue(int(((i+1) / max(1, total_groups)) * 50)) 
            QApplication.processEvents()

        if final_single_files:
            single_folder = os.path.join(out, "single")
            os.makedirs(single_folder, exist_ok=True)
            total_singles = len(final_single_files)
            for i, f in enumerate(final_single_files):
                src_path = os.path.join(src, f)
                if os.path.exists(src_path):
                    file_operation(src_path, os.path.join(single_folder, f))
                
                self.progress_bar.setValue(50 + int(((i+1) / total_singles) * 50)) 
                QApplication.processEvents()

        self.progress_bar.setValue(100)
        QMessageBox.information(self, "任務完成", f"✅ 操作完成！\n成功建立 {group_count} 個群組。\n共 {len(final_single_files)} 張獨立圖片。")
        
        self.clear_scroll_area()
        self.groups_data.clear()
        self.single_files.clear()
        self.lbl_status.setText("操作完畢。等待下一次任務。")
        self.progress_bar.setValue(0)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageGrouperApp()
    ex.show()
    sys.exit(app.exec())