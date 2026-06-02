import numpy as np
from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsPolygonItem
from PySide6.QtCore import Qt, QPointF, Signal, QRectF
from PySide6.QtGui import QPixmap, QImage, QPolygonF, QPen, QColor, QBrush, QTransform, QPainter

def ndarray_to_qpixmap(arr):
    h, w, c = arr.shape
    bytes_per_line = c * w
    qimg = QImage(arr.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(qimg)

class SourceSelectionCanvas(QGraphicsView):
    selectionFinished = Signal(object) 

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.points = []
        self.polygon_item = None
        self.pixmap_item = None

    def zoom_view(self, factor):
        self.scale(factor, factor)

    def reset_zoom(self):
        self.resetTransform()

    def set_image(self, img_array):
        self.points = []
        self.scene.clear()
        self.polygon_item = None
        pixmap = ndarray_to_qpixmap(img_array)
        self.pixmap_item = self.scene.addPixmap(pixmap)
        self.setSceneRect(pixmap.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = self.mapToScene(event.position().toPoint())
            self.points.append(pos)
            self.update_polygon()
        super().mousePressEvent(event)

    def update_polygon(self):
        if self.polygon_item:
            self.scene.removeItem(self.polygon_item)
        
        if len(self.points) > 1:
            poly = QPolygonF(self.points)
            self.polygon_item = QGraphicsPolygonItem(poly)
            self.polygon_item.setPen(QPen(QColor(255, 255, 0), 2))
            self.polygon_item.setBrush(QBrush(QColor(255, 255, 0, 50)))
            self.scene.addItem(self.polygon_item)

    def keyPressEvent(self, event):
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if len(self.points) > 2:
                self.selectionFinished.emit(self.points)
        elif event.key() == Qt.Key.Key_Escape:
            self.points = []
            self.update_polygon()
        super().keyPressEvent(event)

class DraggableLayer(QGraphicsPixmapItem):
    def __init__(self, pixmap):
        super().__init__(pixmap)
        self.setFlags(QGraphicsPixmapItem.GraphicsItemFlag.ItemIsMovable | 
                      QGraphicsPixmapItem.GraphicsItemFlag.ItemIsSelectable |
                      QGraphicsPixmapItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setTransformationMode(Qt.TransformationMode.SmoothTransformation)

class MainCanvas(QGraphicsView):
    scaleChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.bg_item = None
        self.layer = None

    def zoom_view(self, factor):
        self.scale(factor, factor)

    def reset_zoom(self):
        self.resetTransform()

    def wheelEvent(self, event):
        # Only scale the LAYER if it's selected
        if self.layer and self.layer.isSelected():
            delta = event.angleDelta().y()
            current_scale = self.layer.scale()
            factor = 1.05 if delta > 0 else 0.95
            new_scale = max(0.1, min(3.0, current_scale * factor))
            self.set_layer_scale(new_scale)
            self.scaleChanged.emit(new_scale)
            event.accept()
        else:
            # Otherwise, use standard scroll behavior
            super().wheelEvent(event)

    def set_background(self, img_array):
        if self.bg_item:
            self.scene.removeItem(self.bg_item)
        pixmap = ndarray_to_qpixmap(img_array)
        self.bg_item = self.scene.addPixmap(pixmap)
        self.bg_item.setZValue(-1)
        self.setSceneRect(pixmap.rect())

    def add_layer(self, pixmap):
        if self.layer:
            self.scene.removeItem(self.layer)
        self.layer = DraggableLayer(pixmap)
        self.scene.addItem(self.layer)
        self.layer.setPos(50, 50)

    def get_layer_transform(self):
        if not self.layer:
            return None
        pos = self.layer.pos()
        scale = self.layer.scale()
        return pos.x(), pos.y(), scale

    def set_layer_scale(self, scale):
        if self.layer:
            self.layer.setScale(scale)
