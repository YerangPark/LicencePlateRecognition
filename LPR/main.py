import cv2
import os

try:
	from PIL import Image
except ImportError:
	import Image
import pytesseract
from matplotlib import pyplot as plt
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack, threshold_sauvola)
import numpy as np

import sys, os
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap, QIcon


################################################################################################
################################################################################################
################################################################################################
################################################################################################
################################################################################################
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


################################################################################################
################################################################################################################################################################################################
################################################################################################
################################################################################################
############### Image Processing

class ImageProcessing(Image):
	def __init__(self):
		gray = cv2.cvtColor(self, cv2.COLOR_BGR2GRAY)
		imgRGB = cv2.cvtColor(self, cv2.COLOR_BGR2RGB)
		rgbGray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)

		img_gau_blurred = cv2.GaussianBlur(rgbGray, ksize=(5, 5), sigmaX=0)
		img_bil_blurred = cv2.bilateralFilter(gray, -1, 10, 5)

		thresh_sauvola = threshold_sauvola(rgbGray, 25)
		binary_sauvola = rgbGray > thresh_sauvola
		binary_sauvola = (binary_sauvola).astype('uint8')
		binary_sauvola = binary_sauvola * 255
		th = binary_sauvola
		th = (th).astype('uint8')


		contours_dict, pos_cnt = self.findContour(th)
		self.pickContour(th, contours_dict, pos_cnt)

	def findContour(img):
		contours, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		contours_dict = []
		pos_cnt = list()

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			# boundingRect(): 인자로 받은 contour에 외접하고 똑바로 세워진 직사각형의 좌상단 꼭지점 좌표(x, y)와 가로 세로 폭을 리턴함.
			cv2.rectangle(img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)
			# cv2.rectangle(이미지, 중심 좌표, 반지름, 색상, 두께) : 사각형 그리기

			# insert to dict
			contours_dict.append({
				'contour': contour,
				'x': x,
				'y': y,
				'w': w,
				'h': h,
				'cx': x + (w / 2),  # 중앙 x
				'cy': y + (h / 2)  # 중앙 y
			})
		return contours_dict, pos_cnt

	def pickContour(img, contours_dict, pos_cnt):
		## 1st
		count = 0
		for d in contours_dict:
			rect_area = d['w'] * d['h']  # 영역 크기
			aspect_ratio = d['w'] / d['h']

			# 이 부분을 내가 원하는 Contour 사각형의 비율, 넓이로 변경해줘야 함.
			# 이미지에 따라 값이 바뀔 수 있으므로 이미지 환경을 통일시켜 주는 것이 좋을 것 같음.(어짜피 환경에서는 동일한 설정의 이미지가 입력될테니까..)
			# 그리고 넓이 때문에 예전 번호판의 경우 윗줄 인식이 안된다는 점 인식하기...
			if (aspect_ratio >= 0.3) and (aspect_ratio <= 1.0) and (rect_area >= 800) and (rect_area <= 2000):
				d['idx'] = count
				count += 1
				pos_cnt.append(d)

		## 2nd
		result_idx = img.find_number(pos_cnt)
		matched_result = []
		for idx_list in result_idx:
			matched_result.append(np.take(pos_cnt, idx_list))
		return matched_result

	def find_number(contour_list):
		MAX_DIAG_MULTIPLYER = 4  # contourArea의 대각선 x7 안에 다음 contour가 있어야함
		MAX_ANGLE_DIFF = 15.0  # contour와 contour 중심을 기준으로 한 각도가 설정각 이내여야함 --> 카메라 각도가 너무 틀어져있으면 이 각도로 측정되지 않을 수 있음에 주의...
		MAX_AREA_DIFF = 0.5  # contour간에 면적 차이가 설정값보다 크면 인정하지 x
		MAX_WIDTH_DIFF = 0.8  # contour간에 너비 차이가 설정값보다 크면 인정 x
		MAX_HEIGHT_DIFF = 0.2  # contour간에 높이 차이가 크면 인정 x
		MIN_N_MATCHED = 3  # 위의 조건을 따르는 contour가 최소 3개 이상이어야 번호판으로 인정
		MAX_N_MATCHED = 8

		if len(contour_list) < 3:
			return

		matched_result_idx = []

		# contour_list[n]의 keys = dict_keys(['contour', 'x', 'y', 'w', 'h', 'cx', 'cy', 'idx'])
		for d1 in contour_list:
			matched_contour_idx = []
			for d2 in contour_list:  # for문을 2번 돌면서 d1과 d2를 비교할 것임
				if d1['idx'] == d2['idx']:
					continue

				# 피타고라스로 대각 길이를 구하기 위해 dy와 dx를 구함.
				dx = abs(d1['cx'] - d2['cx'])  # 각각 중앙점 기준으로 가로 거리
				dy = abs(d1['cy'] - d2['cy'])  # 각각 중앙점 기준으로 세로 거리

				# d1 사각형의 대각선 거리
				diag_len = np.sqrt(d1['w'] ** 2 + d1['w'] ** 2)

				# contour 중심간의 거리 (L2 norm으로 계산한 거리)
				distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))

				# 각도 구하기
				# tan세타 = dy / dx
				# 세타 = arctan(dy/dx) (using 역함수)
				if dx == 0:
					angle_diff = 90  # x축의 차이가 없다는 것은 contour가 서로 위/아래에 위치한다는 것
				else:
					angle_diff = np.degrees(np.arctan(dy / dx))  # 라디안 값을 도로 바꾼다.

				# 면적의 비율 (기준 contour 대비)
				area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
				# 너비의 비율
				width_diff = abs(d1['w'] - d2['w']) / d1['w']
				# 높이의 비율
				height_diff = abs(d1['h'] - d2['h']) / d2['h']

				# 조건에 맞는 idx만을 matched_contours_idx에 append할 것이다.
				if distance < diag_len * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF \
						and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF \
						and height_diff < MAX_HEIGHT_DIFF:
					matched_contour_idx.append(d2['idx'])

			# d2 다 돌고 기준이었던 d1을 append
			matched_contour_idx.append(d1['idx'])

			# 앞서 정한 후보군의 갯수보다 적으면 탈락
			if len(matched_contour_idx) < MIN_N_MATCHED:
				continue
			elif len(matched_contour_idx) >= MAX_N_MATCHED:
				continue

			# 최종 contour 묶음을 입력
			matched_result_idx.append(matched_contour_idx)

			# 최종 묶음에 들지 못한 애들은 따로 구분.
			unmatched_contour_idx = []
			for d4 in contour_list:
				if d4['idx'] not in matched_contour_idx:
					unmatched_contour_idx.append(d4['idx'])

			# 묶음이 안된 애 전체 정보를 unmatched_contour에 대입.
			# np.take(a,idx)   a 배열에서 idx위치에 해당하는 아이만 뽑음.
			unmatched_contour = np.take(pos_cnt, unmatched_contour_idx)

			# 묶음 안된 애들에 대해 재귀로 돈다.
			recursive_contour_list = find_number(unmatched_contour)

			# 최종 리스트에 추가
			for idx in recursive_contour_list:
				matched_result_idx.append(idx)
			break
		return matched_result_idx

	def perspectiveTransform(img):
		for i, matched_chars in enumerate(matched_result):
			orig_img = image.copy()

			# lambda 함수로 소팅. 'cx'의 키값을 오름차순으로 정렬한다. (contours들이 좌측부터 차례대로 정렬됨)
			sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

			# 0번째 중앙 x좌표에서 마지막 중앙 x좌표까지의 길이
			plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
			plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

			# 번호판 영역의 네 모서리 좌표를 저장한다.
			leftUp = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y']}
			leftDown = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y'] + sorted_chars[0]['h']}
			rightUp = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'], 'y': sorted_chars[-1]['y']}
			rightDown = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'],
						 'y': sorted_chars[-1]['y'] + sorted_chars[-1]['h']}

			# 원근 변환을 위해 input 좌표와 output 좌표를 기록 (좌상->좌하->우상->우하) (번호판 크기에 따라서 pts2는 달라질 수 있음에 주의)
			pts1 = np.float32([[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightUp['x'], rightUp['y']],
							   [rightDown['x'], rightDown['y']]])
			pts2 = np.float32([[0, 0], [0, 110], [520, 0], [520, 110]])
			# 다각형 선을 그리기 위해서 (좌상->좌하->우하->우상)
			ptsPoly = np.array(
				[[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightDown['x'], rightDown['y']],
				 [rightUp['x'], rightUp['y']]])

			M = cv2.getPerspectiveTransform(pts1, pts2)
			dst = cv2.warpPerspective(img, M, (520, 110))
			numPlate = dst.copy()

			# 점과 선 그리기
			orig_img = cv2.polylines(img, [ptsPoly], True, (0, 255, 255), 2)
			cv2.circle(img, (leftUp['x'], leftUp['y']), 10, (255, 0, 0), -1)
			cv2.circle(img, (leftDown['x'], leftDown['y']), 10, (0, 255, 0), -1)
			cv2.circle(img, (rightUp['x'], rightUp['y']), 10, (0, 0, 255), -1)
			cv2.circle(img, (rightDown['x'], rightDown['y']), 10, (0, 0, 0), -1)

			plt.subplot(121), plt.imshow(orig_img), plt.title('image')
			plt.subplot(122), plt.imshow(dst, 'gray'), plt.title('Perspective')
			plt.show()

			# crop하고 패딩 주기 전 Test로 OCR 인식
			print('------------------------\nBefore Padding : ')
			text = pytesseract.image_to_string(dst, lang='kor', config='--psm 7')
			print(text)


sys.exit(app.exec_())