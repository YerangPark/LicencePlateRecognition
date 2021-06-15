import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap, QIcon


# 포토 뷰어
class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.initImageLabel()

    def initImageLabel(self):
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
                    QLabel{
                    border : 2px dashed #aaa
                    }
                ''')

    def setPixmap(self, image):
        super().setPixmap(image)


# 메인 클래스
class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 600, 400)
        self.setFixedSize(600, 400)
        self.setAcceptDrops(True)

        # 위젯 선언
        self.label1 = QLabel("인식 결과 : ")
        self.label2 = QLabel("번호판 좌표 : ")
        self.label2_1 = QLabel("")
        self.lineEdit1 = QLineEdit()
        self.pushButton1 = QPushButton("검출")
        self.quitButton = QPushButton('종료')
        self.quitButton.clicked.connect(QCoreApplication.instance().quit)
        self.clearButton = QPushButton('초기화')
        self.clearButton.clicked.connect(self.clearBtn)

        # GridLayout
        mainLayout = QGridLayout()
        self.photoViewer = ImageLabel()

        # Widget layout
        mainLayout.addWidget(self.photoViewer, 0, 0, 4, 3)

        mainLayout.addWidget(self.label1, 5, 0)
        mainLayout.addWidget(self.lineEdit1, 5, 1)
        mainLayout.addWidget(self.pushButton1, 5, 2)

        mainLayout.addWidget(self.label2, 6, 0)
        mainLayout.addWidget(self.label2_1, 6, 1)
        mainLayout.addWidget(self.clearButton, 6, 2)

        mainLayout.addWidget(self.quitButton, 7, 2)

        # Window Setting
        self.setWindowTitle('자동차 번호판 인식 프로그램')
        self.setWindowIcon(QIcon('car_icon.ico'))
        self.setLayout(mainLayout)

    def clearBtn(self):
        self.lineEdit1.clear()
        self.label2_1.clear()
        self.photoViewer.clear()
        self.photoViewer.initImageLabel()

    def dragEnterEvent(self, event):
        if  event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dragMoveEvent(self, event):
        if  event.mimeData().hasImage:
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasImage:
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)

            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))

app = QApplication(sys.argv)

demo = AppDemo()
demo.show()

sys.exit(app.exec_())