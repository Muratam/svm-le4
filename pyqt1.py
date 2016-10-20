from PyQt5.QtCore import QDir, QPoint, QRect, QSize, Qt
from PyQt5.QtGui import QImage, QImageWriter, QPainter, QPen, qRgb
from PyQt5.QtWidgets import (QAction, QApplication, QColorDialog, QFileDialog,
                             QInputDialog, QMainWindow, QMenu, QMessageBox,
                             QWidget)
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter

# GUI は mousePressEvent / mouseMoveEvent / mouseReleaseEvent


class ScribbleArea(QWidget):
    def __init__(self, parent=None):
        super(ScribbleArea, self).__init__(parent)
        self.image = QImage()
        self.lastPoint = QPoint()

    def mousePressEvent(self, event):
        self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        self.drawLineTo(event.pos())

    def mouseReleaseEvent(self, event):
        self.drawLineTo(event.pos())

    def paintEvent(self, event):
        QPainter(self).drawImage(event.rect(), self.image, event.rect())

    def drawLineTo(self, endPoint):
        painter = QPainter(self.image)
        painter.setPen(
            QPen(Qt.blue, 5, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawLine(self.lastPoint, endPoint)
        rad = 10
        self.update(
            QRect(self.lastPoint, endPoint).normalized().adjusted(-rad, -rad,
                                                                  +rad, +rad))
        self.lastPoint = QPoint(endPoint)

    def resizeEvent(self, event):
        self.resizeImage(self.image,
                         QSize(self.width() + 128, self.height() + 128))
        self.update()

    def resizeImage(self, image, newSize):
        self.image = QImage(newSize, QImage.Format_RGB32)
        self.image.fill(qRgb(0, 0, 0))


class MainWindow(QMainWindow):
    def __init__(self, w, h):
        super(MainWindow, self).__init__()
        self.setCentralWidget(ScribbleArea())
        self.resize(w, h)


if __name__ == '__main__':
    import sys
    app = QApplication(sys.argv)
    window = MainWindow(500, 500)
    window.show()
    sys.exit(app.exec_())
