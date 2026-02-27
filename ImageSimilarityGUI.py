import os
import shutil
import sys
import numpy as np

# PyQt6 模組
from PyQt6.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QFileDialog, 
                             QSpinBox, QDoubleSpinBox, QScrollArea, QGroupBox, 
                             QMessageBox, QComboBox, QStackedWidget,
                             QProgressBar, QStyle, QDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap, QPalette

# ImageHash
from PIL import Image
import imagehash

# OpenCV
import cv2

# AI / Deep Learning
import torch
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel

# --- 影像美感模型封裝 (自動檢測 GPU) ---
class AestheticScorer:
    def __init__(self):
        # 自動偵測 GPU
        if torch.cuda.is_available():
            self.device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = "mps"
        else:
            self.device = "cpu"
        
        self.model = None
        self.processor = None
        self.is_ready = False  # 新增旗標，確認模型是否加載成功

    def load_model(self):
        """ 顯式加載模型，並確保移動到正確設備 """
        if not self.is_ready:
            try:
                # 指定使用 CLIP 模型
                model_id = "openai/clip-vit-base-patch32"
                self.model = CLIPModel.from_pretrained(model_id).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_id)
                self.labels = ["a high quality professional photo", "a low quality blurry messy photo"]
                self.model.eval() # 設為評估模式，節省資源
                self.is_ready = True
                print(f"美感模型已成功載入設備: {self.device}")
            except Exception as e:
                print(f"模型載入失敗: {e}")
                self.is_ready = False

    def get_score(self, image_path):
        """ 核心評分邏輯 """
        try:
            # A. 讀取影像 (支援中文路徑)
            img_data = np.fromfile(image_path, dtype=np.uint8)
            img_cv = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            if img_cv is None: return 0.0

            # B. 傳統技術分 (CV Score)
            gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
            sharpness = np.log1p(cv2.Laplacian(gray, cv2.CV_64F).var()) * 15
            contrast = gray.std()
            cv_score = sharpness + (contrast * 0.5)

            # C. AI 語義分 (只有在模型準備好時才執行)
            ai_score = 50.0 # 預設中位數
            if self.is_ready:
                pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                inputs = self.processor(text=self.labels, images=pil_img, return_tensors="pt", padding=True).to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = outputs.logits_per_image.softmax(dim=1)
                    ai_score = probs[0][0].item() * 100

            # 混合加權 (AI 70% + CV 30%)
            return float((ai_score * 0.7) + (cv_score * 0.3))
        except:
            return 0.0

# 建立全域單例，避免重複初始化模型
_GLOBAL_SCORER = AestheticScorer()

# --- 影像美感與品質評分演算法 (Aesthetic & Quality Score - 整合版) ---
def calculate_aesthetic_score(image_path):
    """
    綜合評估影像的美感與品質：
    結合傳統指標 (清晰度、對比度、色彩) 與 Deep Learning 語義分析。
    """
    try:
        # 1. 支援中文路徑讀取
        img_data = np.fromfile(image_path, dtype=np.uint8)
        img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        if img is None: return 0.0

        # --- 第一部分：傳統技術指標 (CV Score) ---
        h, w = img.shape[:2]
        target_w = 800
        img_resized = cv2.resize(img, (target_w, int(h * (target_w / w))))
        gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # A. 清晰度 (Sharpness) - 增加抗噪處理
        blurred_gray = cv2.GaussianBlur(gray, (3, 3), 0)
        sharpness = cv2.Laplacian(blurred_gray, cv2.CV_64F).var()
        sharpness_score = np.log1p(sharpness) * 15 

        # B. 對比度 (Contrast)
        contrast = gray.std()

        # C. 色彩豐富度 (Colorfulness)
        (B, G, R) = cv2.split(img_resized.astype("float"))
        rg = np.absolute(R - G)
        yb = np.absolute(0.5 * (R + G) - B)
        colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + (0.3 * np.sqrt(np.mean(rg)**2 + np.mean(yb)**2))

        # D. 曝光平衡懲罰 (Exposure Penalty)
        exposure_penalty = abs(np.mean(gray) - 127.5) / 127.5

        cv_score = (sharpness_score + contrast * 0.4 + colorfulness * 0.4) * (1 - 0.5 * exposure_penalty)

        # --- 第二部分：語義美感指標 (AI Score) ---
        # 轉換為 PIL 給 CLIP 模型使用
        pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ai_score = _GLOBAL_SCORER.get_ai_score(pil_img)

        # --- 第三部分：加權融合 ---
        # AI 分數 (構圖/美感) 佔 70%，CV 分數 (清晰/品質) 佔 30%
        final_score = (ai_score * 0.7) + (cv_score * 0.3)

        return float(final_score)

    except Exception as e:
        print(f"Error scoring {image_path}: {e}")
        return 0.0

# --- 自訂可點擊的圖片標籤 (用於彈出大圖) ---
# 修改 ClickableLabel 的样式以适配深色模式
class ClickableLabel(QLabel):
    clicked = pyqtSignal(str)

    def __init__(self, img_path, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.img_path = img_path
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet(
            """
            QLabel {
                border: 1px solid #444444;
                padding: 2px;
                background-color: #222222;
                color: #FFFFFF;
            }
            QLabel:hover {
                border: 1px solid #888888;
                background-color: #333333;
            }
            """
        )

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
        
        groups = []
        single_files = []

        # 執行分組
        if self.algo_mode == 'imagehash':
            groups, single_files = self._run_imagehash()
        elif self.algo_mode == 'opencv':
            groups, single_files = self._run_opencv()
        elif self.algo_mode == 'ai':
            groups, single_files = self._run_ai()

        # 執行美感評分與組內排序
        self.progress_update.emit("正在進行 AI 美感品質評分與篩選最佳圖片...")
        scored_groups = []
        total_groups = len(groups)
        
        for i, grp in enumerate(groups):
            self.progress_percent.emit(90 + int((i / max(1, total_groups)) * 10))
            grp_scored = []
            for f in grp:
                path = os.path.join(self.src_dir, f)
                score = calculate_aesthetic_score(path)
                grp_scored.append({'file': f, 'score': score})
            
            # 依分數降序排序 (最高分在前面)
            grp_scored.sort(key=lambda x: x['score'], reverse=True)
            scored_groups.append(grp_scored)

        self.progress_update.emit(f"掃描與評分完成！找到 {len(scored_groups)} 組相似圖片。")
        self.progress_percent.emit(100)
        self.scan_finished.emit(scored_groups, single_files)

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
                self.progress_percent.emit(50 + int((i / total) * 40))

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
        return groups, single_files

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
                self.progress_percent.emit(50 + int((i / total_files) * 40))

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
        return groups, single_files

    def _run_ai(self):
        sim_thresh = self.params.get('sim', 0.80)
        model_name = self.params.get('ai_model', 'resnet18')
        self.progress_update.emit(f"載入 {model_name} 模型中...")
        self.progress_percent.emit(10)
        
        try:
            device = "cuda" if torch.cuda.is_available() else \
                      ("mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else "cpu")
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
            elif model_name == 'resnet50':
                from torchvision.models import resnet50, ResNet50_Weights
                model = resnet50(weights=ResNet50_Weights.DEFAULT)
                model.fc = torch.nn.Identity() 
            elif model_name == 'efficientnet_b2':
                from torchvision.models import efficientnet_b2, EfficientNet_B2_Weights
                model = efficientnet_b2(weights=EfficientNet_B2_Weights.DEFAULT)
                model.classifier = torch.nn.Identity()
                
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
            return [], []

        features = {}
        total = len(self.all_files)
        
        for idx, filename in enumerate(self.all_files):
            if idx % max(1, total // 20) == 0: 
                self.progress_update.emit(f"提取語義特徵... ({idx}/{total})")
                self.progress_percent.emit(20 + int((idx / total) * 30)) 
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
                self.progress_percent.emit(50 + int((i / total_files) * 40))

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
        return groups, single_files


# --- 主圖形介面 ---
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
        QWidget { font-family: "Segoe UI", "Microsoft JhengHei", sans-serif; font-size: 10pt; color: #FFFFFF; background-color: #121212; }
        QGroupBox { font-weight: bold; border: 1px solid #444444; border-radius: 6px; margin-top: 10px; background-color: #1E1E1E; }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 3px 0 3px; color: #BB86FC; }
        QLineEdit { padding: 4px; border: 1px solid #555555; border-radius: 4px; background-color: #1E1E1E; color: #FFFFFF; }
        QLineEdit:focus { border: 1px solid #BB86FC; }
        QPushButton { padding: 5px 10px; border-radius: 4px; background-color: #333333; border: 1px solid #555555; color: #FFFFFF; }
        QPushButton:hover { background-color: #444444; }
        QPushButton#primaryBtn { background-color: #BB86FC; color: white; border: none; font-weight: bold; padding: 8px; }
        QPushButton#primaryBtn:hover { background-color: #985EFF; }
        QPushButton#primaryBtn:disabled { background-color: #5A5A5A; }
        QPushButton#actionBtn { background-color: #03DAC6; color: black; border: none; font-weight: bold; padding: 8px; }
        QPushButton#actionBtn:hover { background-color: #018786; }
        QPushButton#actionBtn:disabled { background-color: #5A5A5A; }
        QPushButton#singleActionBtn { background-color: #CF6679; color: white; font-weight: bold; border-radius: 4px; padding: 6px; }
        QPushButton#singleActionBtn:hover { background-color: #B00020; }
        QComboBox, QSpinBox, QDoubleSpinBox { padding: 4px; border: 1px solid #555555; border-radius: 4px; background-color: #1E1E1E; color: #FFFFFF; }
        QProgressBar { border: 1px solid #444444; border-radius: 4px; text-align: center; color: white; background-color: #333333; }
        QProgressBar::chunk { background-color: #BB86FC; width: 10px; }
        QScrollArea { border: 1px solid #444444; background-color: #121212; border-radius: 6px; }
        """
        self.setStyleSheet(style)

    def initUI(self):
        self.setWindowTitle('Image Similarity Pro (美感篩選版)')
        self.resize(1100, 750)
        
        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(15)

        # ==================== 左側：緊湊功能區 ====================
        left_panel = QWidget()
        left_panel.setMaximumWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(10)

        dir_icon = self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon)

        # 1. 路徑設定區 (移除了輸出目錄)
        path_group = QGroupBox("📁 檔案目錄設定")
        path_layout = QVBoxLayout()
        path_layout.addWidget(QLabel("輸入資料夾 (執行後建立 Trash 資料夾):"))
        in_layout = QHBoxLayout()
        self.input_entry = QLineEdit()
        in_layout.addWidget(self.input_entry)
        btn_in = QPushButton()
        btn_in.setIcon(dir_icon)
        btn_in.clicked.connect(self.browse_input)
        in_layout.addWidget(btn_in)
        path_layout.addLayout(in_layout)
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
        self.combo_ai_model.addItems(["resnet18", "mobilenet_v2", "efficientnet_b0", "resnet50", "efficientnet_b2"])
        ai_layout.addWidget(self.combo_ai_model)
        
        sim_layout = QHBoxLayout()
        sim_layout.addWidget(QLabel("相似度 (0.5~1.0):"))
        self.ai_sim_spin = QDoubleSpinBox()
        self.ai_sim_spin.setRange(0.50, 1.00)
        self.ai_sim_spin.setSingleStep(0.01)
        self.ai_sim_spin.setValue(0.80)
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

        self.btn_scan = QPushButton("🔍 開始掃描與美感評估")
        self.btn_scan.setObjectName("primaryBtn")
        self.btn_scan.clicked.connect(self.start_scan)
        exec_layout.addWidget(self.btn_scan)

        self.lbl_status = QLabel("就緒。")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #005A9E; font-size: 9pt; margin-top: 5px;")
        exec_layout.addWidget(self.lbl_status)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        exec_layout.addWidget(self.progress_bar)

        self.btn_execute_all = QPushButton("✅ 批量處理所有組 (移至 Trash)")
        self.btn_execute_all.setObjectName("actionBtn")
        self.btn_execute_all.setEnabled(False)
        self.btn_execute_all.clicked.connect(self.execute_batch_action)
        exec_layout.addWidget(self.btn_execute_all)

        exec_group.setLayout(exec_layout)
        left_layout.addWidget(exec_group)
        
        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ==================== 右側：巨大預覽顯示區 ====================
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.scroll_content)
        
        main_layout.addWidget(self.scroll_area, stretch=1) 
        self.setLayout(main_layout)

    def change_algo_params(self, index):
        self.param_stack.setCurrentIndex(index)

    def browse_input(self):
        folder = QFileDialog.getExistingDirectory(self, "選擇輸入")
        if folder: self.input_entry.setText(folder)

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
        self.btn_execute_all.setEnabled(False)
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

    def on_scan_finished(self, scored_groups, single_files):
        self.groups_data = scored_groups
        self.single_files = single_files
        self.btn_scan.setEnabled(True)

        if not scored_groups and not single_files:
            self.lbl_status.setText("沒有可處理的圖片。")
            return

        self.display_groups()
        if scored_groups:
            self.btn_execute_all.setEnabled(True)
        self.lbl_status.setText(f"完成！共找到 {len(scored_groups)} 組相似圖，{len(single_files)} 張為獨立圖片。")

    def show_full_image(self, img_path):
        dialog = QDialog(self)
        dialog.setWindowTitle(os.path.basename(img_path))
        layout = QVBoxLayout()
        lbl = QLabel()
        pixmap = QPixmap(img_path)
        
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
            group_box = QGroupBox(f"📂 分組 {idx + 1} (共 {len(group)} 張)")
            group_box.setStyleSheet("QGroupBox { background-color: #FFFFFF; }")
            
            img_layout = QHBoxLayout()
            img_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
            
            for i, item in enumerate(group):
                filename = item['file']
                score = item['score']
                path = os.path.join(src_dir, filename)
                
                v_layout = QVBoxLayout()
                v_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
                
                lbl_img = ClickableLabel(path) 
                lbl_img.clicked.connect(self.show_full_image)
                
                try:
                    pixmap = QPixmap(path).scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
                    lbl_img.setPixmap(pixmap)
                except Exception:
                    lbl_img.setText("[預覽失敗]")
                v_layout.addWidget(lbl_img)
                
                # 標籤顯示分數與狀態
                if i == 0:
                    lbl_info = QLabel(f"👑 最佳 (美感分: {score:.1f})")
                    lbl_info.setStyleSheet("color: #D2691E; font-weight: bold; font-size: 11pt;")
                else:
                    lbl_info = QLabel(f"🗑️ 待刪 (美感分: {score:.1f})")
                    lbl_info.setStyleSheet("color: #666666;")
                
                lbl_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
                v_layout.addWidget(lbl_info)
                
                wrapper = QWidget()
                wrapper.setLayout(v_layout)
                img_layout.addWidget(wrapper)
            
            scroll_widget = QWidget()
            scroll_widget.setLayout(img_layout)
            inner_scroll = QScrollArea()
            inner_scroll.setWidgetResizable(True)
            inner_scroll.setWidget(scroll_widget)
            inner_scroll.setFixedHeight(230) 
            inner_scroll.setStyleSheet("border: none;")

            # 處理單組的按鈕
            btn_single_action = QPushButton("🗑️ 處理此組 (保留最佳，其餘移至 Trash)")
            btn_single_action.setObjectName("singleActionBtn")
            btn_single_action.clicked.connect(lambda checked, b=group_box, g=group: self.execute_single_group(b, g))

            box_layout = QVBoxLayout()
            box_layout.addWidget(inner_scroll)
            box_layout.addWidget(btn_single_action)
            group_box.setLayout(box_layout)
            
            self.scroll_layout.addWidget(group_box)
            self.group_widgets.append({'box': group_box, 'data': group})

    def move_files_to_trash(self, src_dir, filenames):
        trash_dir = os.path.join(src_dir, "Trash")
        os.makedirs(trash_dir, exist_ok=True)
        moved_count = 0
        for fname in filenames:
            src_path = os.path.join(src_dir, fname)
            dst_path = os.path.join(trash_dir, fname)
            if os.path.exists(src_path):
                try:
                    shutil.move(src_path, dst_path)
                    moved_count += 1
                except Exception as e:
                    print(f"移動失敗 {fname}: {e}")
        return moved_count

    def execute_single_group(self, box_widget, group_data):
        src_dir = self.input_entry.text().strip()
        # group_data[0] 是最佳圖片，group_data[1:] 移到 Trash
        to_move = [item['file'] for item in group_data[1:]]
        moved = self.move_files_to_trash(src_dir, to_move)
        
        # 將 Widget 從 UI 移除
        box_widget.deleteLater()
        
        # 將資料從清單移除
        if group_data in self.groups_data:
            self.groups_data.remove(group_data)
            
        self.lbl_status.setText(f"成功處理單組，已將 {moved} 張圖片移至 Trash。")
        
        # 檢查是否都處理完了
        if not self.groups_data:
            self.btn_execute_all.setEnabled(False)

    def execute_batch_action(self):
        src_dir = self.input_entry.text().strip()
        self.btn_execute_all.setEnabled(False)
        self.progress_bar.setValue(0)
        
        total_groups = len(self.groups_data)
        total_moved = 0
        
        for i, group_data in enumerate(list(self.groups_data)):
            to_move = [item['file'] for item in group_data[1:]]
            moved = self.move_files_to_trash(src_dir, to_move)
            total_moved += moved
            
            self.progress_bar.setValue(int(((i+1) / max(1, total_groups)) * 100))
            QApplication.processEvents()

        self.clear_scroll_area()
        self.groups_data.clear()
        
        QMessageBox.information(self, "任務完成", f"✅ 批量處理完成！\n已將 {total_moved} 張較低美感評分的圖片移至 Trash。\n（最佳圖片與獨立圖片均留在原位）")
        self.lbl_status.setText("操作完畢。等待下一次任務。")
        self.progress_bar.setValue(0)

# 在主窗口中设置深色模式
class DarkModeApp(QApplication):
    def __init__(self, args):
        super().__init__(args)
        self.set_dark_mode()

    def set_dark_mode(self):
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.WindowText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Base, Qt.GlobalColor.black)
        palette.setColor(QPalette.ColorRole.AlternateBase, Qt.GlobalColor.darkGray)
        palette.setColor(QPalette.ColorRole.ToolTipBase, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.ToolTipText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Text, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.Button, Qt.GlobalColor.darkGray)
        palette.setColor(QPalette.ColorRole.ButtonText, Qt.GlobalColor.white)
        palette.setColor(QPalette.ColorRole.BrightText, Qt.GlobalColor.red)
        palette.setColor(QPalette.ColorRole.Highlight, Qt.GlobalColor.darkBlue)
        palette.setColor(QPalette.ColorRole.HighlightedText, Qt.GlobalColor.white)
        self.setPalette(palette)

# 修改主程序入口以使用 DarkModeApp
if __name__ == "__main__":
    app = DarkModeApp(sys.argv)
    ex = ImageGrouperApp()
    ex.show()
    sys.exit(app.exec())