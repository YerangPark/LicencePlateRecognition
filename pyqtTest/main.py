import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap, QIcon

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
        image = image.scaledToHeight(350)
        super().setPixmap(image)


# 메인 클래스
class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.setTableWidgetData()

    def initUI(self):
        self.setGeometry(100, 100, 600, 600)
        self.setFixedSize(600, 600)
        self.setAcceptDrops(True)

        # 위젯 선언
        self.photoViewer = ImageLabel()
        self.photoViewer.setFixedHeight(370)
        self.quitButton = QPushButton('종료')
        self.clearButton = QPushButton('초기화')

        # 버튼 기능 연동
        self.quitButton.clicked.connect(QCoreApplication.instance().quit)
        self.clearButton.clicked.connect(self.clearBtn)

        tableRow=5
        tableCol=4
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(tableRow)
        self.tableWidget.setColumnCount(tableCol)
        self.tableWidget.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tableWidget.setFixedHeight(170)

        # Add Widget on Grid Layout
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.photoViewer, 0, 0, 3, 4)
        mainLayout.addWidget(self.tableWidget, 3, 0, 2, 4)
        mainLayout.addWidget(self.clearButton, 5, 1)
        mainLayout.addWidget(self.quitButton, 5, 2)


        # Window Setting
        self.setWindowTitle('자동차 번호판 인식 프로그램')
        self.setWindowIcon(QIcon('car_icon.ico'))
        self.setLayout(mainLayout)

    def setTableWidgetData(self):
        column_headers = ['파일명', '번호판', '좌표', '']
        self.tableWidget.setHorizontalHeaderLabels(column_headers)
        """ 데이터 삽입용
        for i in range(tableRow):
            for j in range(tableCol):
                self.tableWidget.setItem(i, j, QTableWidgetItem(str(i + j)))
        """

    def clearBtn(self):
        self.tableWidget.clear()
        self.setTableWidgetData()
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