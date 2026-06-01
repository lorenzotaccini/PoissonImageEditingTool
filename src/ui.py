import sys
import os
import numpy as np
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QFileDialog, QComboBox, 
                             QLabel, QSlider, QStackedWidget, QMessageBox, QGroupBox)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QPixmap, QImage, QPainter, QColor
from PIL import Image
from skimage.draw import polygon as sk_polygon

from canvas import MainCanvas, SourceSelectionCanvas
from utils import load_image, save_image
from solver import (poisson_edit, poisson_edit_flatten, 
                    poisson_edit_illumination, poisson_edit_color)

class PoissonApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Poisson Image Editing Pro")
        self.resize(1100, 800)

        self.source_img = None
        self.dest_img = None
        self.mask = None
        self.last_dir = os.path.expanduser("~")
        self.is_inplace = False

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        layout = QHBoxLayout(main_widget)

        # Left Side: Canvas Area
        self.canvas_stack = QStackedWidget()
        self.main_canvas = MainCanvas()
        self.selection_canvas = SourceSelectionCanvas()
        
        self.canvas_stack.addWidget(self.main_canvas)
        
        selection_container = QWidget()
        selection_layout = QVBoxLayout(selection_container)
        selection_layout.setContentsMargins(0,0,0,0)
        selection_layout.addWidget(self.selection_canvas)
        self.btn_confirm_selection = QPushButton("Confirm Selection (ENTER)")
        self.btn_confirm_selection.setStyleSheet("background-color: #2196F3; color: white; font-weight: bold; height: 35px;")
        self.btn_confirm_selection.clicked.connect(lambda: self.selection_canvas.selectionFinished.emit(self.selection_canvas.points))
        selection_layout.addWidget(self.btn_confirm_selection)
        self.canvas_stack.addWidget(selection_container)
        
        layout.addWidget(self.canvas_stack, 4)

        # Right Side: Control Panel
        self.panel_widget = QWidget()
        controls = QVBoxLayout(self.panel_widget)
        layout.addWidget(self.panel_widget, 1)

        # 1. File Group
        file_group = QGroupBox("1. Load & Select")
        file_layout = QVBoxLayout(file_group)
        btn_load_dest = QPushButton("Load Image (Destination)")
        btn_load_dest.clicked.connect(self.load_destination)
        file_layout.addWidget(btn_load_dest)

        btn_load_src = QPushButton("Import External Source")
        btn_load_src.clicked.connect(self.load_source)
        file_layout.addWidget(btn_load_src)

        btn_inplace = QPushButton("Select on Destination (In-Place)")
        btn_inplace.clicked.connect(self.start_inplace_selection)
        file_layout.addWidget(btn_inplace)
        controls.addWidget(file_group)

        # 2. Mode Group
        mode_group = QGroupBox("2. Editing Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Seamless Cloning", 
            "Mixed Gradients", 
            "Texture Flattening", 
            "Illumination Change",
            "Color Change (Tinting)"
        ])
        self.mode_combo.currentIndexChanged.connect(self.update_param_visibility)
        mode_layout.addWidget(self.mode_combo)
        controls.addWidget(mode_group)

        # 3. Parameters Group
        self.param_group = QGroupBox("3. Mode Parameters")
        self.param_layout = QVBoxLayout(self.param_group)
        
        # Flattening
        self.label_thresh = QLabel("Edge Threshold: 0.10")
        self.slider_thresh = QSlider(Qt.Orientation.Horizontal)
        self.slider_thresh.setRange(1, 100)
        self.slider_thresh.setValue(10)
        self.slider_thresh.valueChanged.connect(lambda v: self.label_thresh.setText(f"Edge Threshold: {v/100:.2f}"))
        self.param_layout.addWidget(self.label_thresh)
        self.param_layout.addWidget(self.slider_thresh)

        # Illumination
        self.label_alpha = QLabel("Brightness (Alpha): 0.20")
        self.slider_alpha = QSlider(Qt.Orientation.Horizontal)
        self.slider_alpha.setRange(1, 100)
        self.slider_alpha.setValue(20)
        self.slider_alpha.valueChanged.connect(lambda v: self.label_alpha.setText(f"Brightness (Alpha): {v/100:.2f}"))
        self.param_layout.addWidget(self.label_alpha)
        self.param_layout.addWidget(self.slider_alpha)

        self.label_beta = QLabel("Compression (Beta): 0.20")
        self.slider_beta = QSlider(Qt.Orientation.Horizontal)
        self.slider_beta.setRange(1, 100)
        self.slider_beta.setValue(20)
        self.slider_beta.valueChanged.connect(lambda v: self.label_beta.setText(f"Compression (Beta): {v/100:.2f}"))
        self.param_layout.addWidget(self.label_beta)
        self.param_layout.addWidget(self.slider_beta)

        # Color Tints
        self.color_ctrls = QWidget()
        color_layout = QVBoxLayout(self.color_ctrls)
        self.label_r = QLabel("Red Scale: 1.0x")
        self.slider_r = QSlider(Qt.Orientation.Horizontal)
        self.slider_r.setRange(0, 200); self.slider_r.setValue(100)
        self.slider_r.valueChanged.connect(lambda v: self.label_r.setText(f"Red Scale: {v/100:.1f}x"))
        color_layout.addWidget(self.label_r); color_layout.addWidget(self.slider_r)

        self.label_g = QLabel("Green Scale: 1.0x")
        self.slider_g = QSlider(Qt.Orientation.Horizontal)
        self.slider_g.setRange(0, 200); self.slider_g.setValue(100)
        self.slider_g.valueChanged.connect(lambda v: self.label_g.setText(f"Green Scale: {v/100:.1f}x"))
        color_layout.addWidget(self.label_g); color_layout.addWidget(self.slider_g)

        self.label_b = QLabel("Blue Scale: 1.0x")
        self.slider_b = QSlider(Qt.Orientation.Horizontal)
        self.slider_b.setRange(0, 200); self.slider_b.setValue(100)
        self.slider_b.valueChanged.connect(lambda v: self.label_b.setText(f"Blue Scale: {v/100:.1f}x"))
        color_layout.addWidget(self.label_b); color_layout.addWidget(self.slider_b)
        
        self.param_layout.addWidget(self.color_ctrls)
        controls.addWidget(self.param_group)
        self.update_param_visibility()

        # 4. Transform Group
        self.trans_group = QGroupBox("4. Position & Scale")
        trans_layout = QVBoxLayout(self.trans_group)
        self.label_scale = QLabel("Scale: 1.0x")
        self.slider_scale = QSlider(Qt.Orientation.Horizontal)
        self.slider_scale.setRange(10, 300) 
        self.slider_scale.setValue(100)
        self.slider_scale.valueChanged.connect(self.on_scale_changed)
        trans_layout.addWidget(self.label_scale)
        trans_layout.addWidget(self.slider_scale)
        controls.addWidget(self.trans_group)

        # 5. Action Group
        self.btn_process = QPushButton("PROCESS")
        self.btn_process.setStyleSheet("""
            QPushButton { background-color: #4CAF50; color: white; font-weight: bold; height: 45px; font-size: 14px; }
            QPushButton:disabled { background-color: #cccccc; color: #666666; }
        """)
        self.btn_process.clicked.connect(self.process_image)
        self.btn_process.setEnabled(False)
        controls.addWidget(self.btn_process)

        self.btn_save = QPushButton("Save Result")
        self.btn_save.clicked.connect(self.save_result)
        controls.addWidget(self.btn_save)

        self.status_label = QLabel("Ready. Load an image to begin.")
        self.status_label.setWordWrap(True)
        controls.addWidget(self.status_label)

        controls.addStretch()

        self.selection_canvas.selectionFinished.connect(self.on_selection_finished)
        self.main_canvas.scaleChanged.connect(self.on_canvas_scale_changed)
        self.processed_img = None

    def update_param_visibility(self):
        mode = self.mode_combo.currentText()
        self.label_thresh.setVisible(mode == "Texture Flattening")
        self.slider_thresh.setVisible(mode == "Texture Flattening")
        self.label_alpha.setVisible(mode == "Illumination Change")
        self.slider_alpha.setVisible(mode == "Illumination Change")
        self.label_beta.setVisible(mode == "Illumination Change")
        self.slider_beta.setVisible(mode == "Illumination Change")
        self.color_ctrls.setVisible(mode == "Color Change (Tinting)")

    def on_canvas_scale_changed(self, scale):
        self.slider_scale.blockSignals(True)
        self.slider_scale.setValue(int(scale * 100))
        self.slider_scale.blockSignals(False)
        self.label_scale.setText(f"Scale: {scale:.2f}x")

    def load_destination(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Image", self.last_dir, "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.last_dir = os.path.dirname(path)
            self.dest_img = load_image(path)
            self.main_canvas.set_background(self.dest_img)
            self.canvas_stack.setCurrentWidget(self.main_canvas)
            self.status_label.setText("Image loaded. Select a region or import another source.")

    def load_source(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import Source", self.last_dir, "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.last_dir = os.path.dirname(path)
            self.source_img = load_image(path)
            self.is_inplace = False
            self.selection_canvas.set_image(self.source_img)
            self.canvas_stack.setCurrentIndex(1) 
            self.panel_widget.setEnabled(False)
            self.status_label.setText("Draw a mask on the source and confirm.")

    def start_inplace_selection(self):
        if self.dest_img is None:
            QMessageBox.warning(self, "In-Place", "Load a destination image first.")
            return
        self.source_img = self.dest_img.copy()
        self.is_inplace = True
        self.selection_canvas.set_image(self.source_img)
        self.canvas_stack.setCurrentIndex(1)
        self.panel_widget.setEnabled(False)
        self.status_label.setText("Select a region in the image to modify.")

    def on_selection_finished(self, points):
        if len(points) < 3:
            QMessageBox.warning(self, "Selection", "Please select at least 3 points.")
            return

        h, w = self.source_img.shape[:2]
        coords = np.array([(p.x(), p.y()) for p in points])
        rr, cc = sk_polygon(coords[:, 1], coords[:, 0], shape=(h, w))
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self.mask[rr, cc] = 255

        cutout = self.source_img.copy()
        qimg = QImage(w, h, QImage.Format_ARGB32)
        qimg.fill(Qt.GlobalColor.transparent)
        for i in range(len(rr)):
            y, x = rr[i], cc[i]
            r, g, b = cutout[y, x]
            qimg.setPixelColor(x, y, QColor(r, g, b, 255))
        
        self.main_canvas.add_layer(QPixmap.fromImage(qimg))
        if self.is_inplace:
            self.main_canvas.layer.setPos(0, 0)
            self.main_canvas.layer.setFlag(self.main_canvas.layer.GraphicsItemFlag.ItemIsMovable, False)
            self.slider_scale.setEnabled(False)
            self.status_label.setText("In-place selection ready. Choose mode and PROCESS.")
        else:
            self.main_canvas.layer.setFlag(self.main_canvas.layer.GraphicsItemFlag.ItemIsMovable, True)
            self.slider_scale.setEnabled(True)
            self.status_label.setText("Position selection and PROCESS.")

        self.canvas_stack.setCurrentWidget(self.main_canvas)
        self.slider_scale.setValue(100)
        self.panel_widget.setEnabled(True)
        self.btn_process.setEnabled(True)

    def on_scale_changed(self, value):
        scale = value / 100.0
        self.label_scale.setText(f"Scale: {scale:.2f}x")
        self.main_canvas.set_layer_scale(scale)

    def process_image(self):
        if not self.main_canvas.layer: return
        
        self.panel_widget.setEnabled(False)
        self.main_canvas.setEnabled(False)
        self.status_label.setText("Solving Poisson equations...")
        QApplication.processEvents()

        x, y, scale = self.main_canvas.get_layer_transform()
        h_s, w_s = self.source_img.shape[:2]
        new_w, new_h = int(w_s * scale), int(h_s * scale)
        
        if new_w <= 0 or new_h <= 0:
            self.panel_widget.setEnabled(True); self.main_canvas.setEnabled(True)
            return

        src_scaled = np.array(Image.fromarray(self.source_img).resize((new_w, new_h), Image.Resampling.LANCZOS))
        mask_scaled = np.array(Image.fromarray(self.mask).resize((new_w, new_h), Image.Resampling.NEAREST))

        mode = self.mode_combo.currentText()
        try:
            if mode == "Seamless Cloning":
                res = poisson_edit(src_scaled, self.dest_img, mask_scaled, offset=(int(y), int(x)), mix_gradients=False)
            elif mode == "Mixed Gradients":
                res = poisson_edit(src_scaled, self.dest_img, mask_scaled, offset=(int(y), int(x)), mix_gradients=True)
            elif mode == "Texture Flattening":
                thresh = self.slider_thresh.value() / 100.0
                res = poisson_edit_flatten(src_scaled, self.dest_img, mask_scaled, offset=(int(y), int(x)), edge_threshold=thresh)
            elif mode == "Illumination Change":
                alpha = self.slider_alpha.value() / 100.0
                beta = self.slider_beta.value() / 100.0
                res = poisson_edit_illumination(src_scaled, self.dest_img, mask_scaled, offset=(int(y), int(x)), alpha_factor=alpha, beta=beta)
            elif mode == "Color Change (Tinting)":
                r_s, g_s, b_s = self.slider_r.value()/100.0, self.slider_g.value()/100.0, self.slider_b.value()/100.0
                res = poisson_edit_color(src_scaled, self.dest_img, mask_scaled, offset=(int(y), int(x)), red_scale=r_s, green_scale=g_s, blue_scale=b_s)
            
            self.processed_img = res
            self.dest_img = self.processed_img.copy()
            self.main_canvas.set_background(self.dest_img)
            
            if self.main_canvas.layer:
                self.main_canvas.scene.removeItem(self.main_canvas.layer)
                self.main_canvas.layer = None
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Processing failed: {str(e)}")
        finally:
            self.panel_widget.setEnabled(True)
            self.main_canvas.setEnabled(True)
            self.status_label.setText("Finished. Edit again or save.")
            self.btn_process.setEnabled(False)

    def save_result(self):
        if self.processed_img is None: return
        path, _ = QFileDialog.getSaveFileName(self, "Save Result", self.last_dir, "Images (*.png *.jpg *.jpeg)")
        if path:
            self.last_dir = os.path.dirname(path); save_image(self.processed_img, path)
            QMessageBox.showinfo("Success", "Saved successfully!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PoissonApp(); window.show()
    sys.exit(app.exec())
