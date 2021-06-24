import cv2
try:
	from PIL import Image
except ImportError:
	import Image
import pytesseract
from skimage.filters import threshold_sauvola
import numpy as np

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QCoreApplication
from PyQt5.QtGui import QPixmap, QIcon
import time

class ImageLabel(QLabel):
    def __init__(self):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.initImageLabel()

    def initImageLabel(self):
        self.setFixedHeight(370)
        self.setText('\n\n Drop Image Here \n\n')
        self.setStyleSheet('''
                    QLabel{
                    border : 2px dashed #aaa
                    }
                ''')

    def setPixmap(self, image):
        image = image.scaledToHeight(370)
        super().setPixmap(image)


class Table(QTableWidget):
    def __init__(self, row, col):
        super().__init__()
        self.clear()
        self.limitRow=row
        self.limitCol=col
        self.nowRow=0
        self.initTable()

    def initTable(self):
        self.setFixedHeight(170)
        self.setRowCount(self.limitRow)
        self.setColumnCount(self.limitCol)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setTableWidgetData()
        self.resizeColumnsToContents()

    def setTableWidgetData(self):
        column_headers = ['파일명', '번호판', '감지 영역\n(좌상단,우하단)', '파일 위치']
        self.setHorizontalHeaderLabels(column_headers)

    def set_data(self, cnt, recoInfo):
        self.setItem(cnt, 0, QTableWidgetItem(recoInfo['filename']))
        self.setItem(cnt, 1, QTableWidgetItem(recoInfo['text']))
        self.setItem(cnt, 2, QTableWidgetItem("({0}, {1}), ({2}, {3})".format(recoInfo['axis'][0][0], recoInfo['axis'][0][1], recoInfo['axis'][3][0],recoInfo['axis'][3][1])))
        self.setItem(cnt, 3, QTableWidgetItem(recoInfo['filepath']))


class AppDemo(QWidget):
    def __init__(self):
        super().__init__()
        self.cnt=-1
        self.nowRow=self.nowCol=0
        self.initUI()

    def initUI(self):
        self.setGeometry(100, 100, 600, 600)
        self.setFixedSize(600, 600)
        self.setAcceptDrops(True)

        # 위젯 선언
        self.photoViewer = ImageLabel()
        self.quitButton = QPushButton('종료')
        self.clearButton = QPushButton('초기화')
        self.tableViewer = Table(20,4)
        self.showContourBoxButton = QPushButton('영역 감지 결과 보기')
        self.showPlateButton = QPushButton('번호판 이미지 보기')

        # 버튼 기능 연동
        self.quitButton.clicked.connect(QCoreApplication.instance().quit)
        self.clearButton.clicked.connect(self.clearBtn)
        self.showContourBoxButton.clicked.connect(self.contourBtn)
        self.showPlateButton.clicked.connect(self.plateBtn)

        # Table Event
        self.tableViewer.cellDoubleClicked.connect(self.selectCell)

        # Add Widget on Grid Layout
        photoHeight=6
        mainLayout = QGridLayout()
        mainLayout.addWidget(self.photoViewer, 0, 0, photoHeight-3, 4)
        mainLayout.addWidget(self.tableViewer, photoHeight-3, 0, 2, 4)
        mainLayout.addWidget(self.showContourBoxButton, 5, 0)
        mainLayout.addWidget(self.showPlateButton, 5, 1)
        mainLayout.addWidget(self.clearButton, 5, 2)
        mainLayout.addWidget(self.quitButton, 5, 3)


        # Window Setting
        self.setWindowTitle('자동차 번호판 인식 프로그램')
        self.setWindowIcon(QIcon('car_icon.ico'))
        self.setLayout(mainLayout)

    def clearBtn(self):
        self.cnt = -1
        self.nowRow = self.nowCol = 0
        self.tableViewer.clear()
        self.tableViewer.setTableWidgetData()
        self.photoViewer.clear()
        self.photoViewer.initImageLabel()

    def contourBtn(self):
        if self.tableViewer.item(self.nowRow, self.nowCol):
            cv2.destroyAllWindows()
            img = cv2.imread('images/contourBox(%i).jpg' % self.nowRow)
            dst=cv2.resize(img, dsize=(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Licence Plate Detection Result", dst)
            cv2.waitKey(0)

    def plateBtn(self):
        if self.tableViewer.item(self.nowRow, self.nowCol):
            cv2.destroyAllWindows()
            img=cv2.imread('images/lastImage(%i).jpg' % self.nowRow)
            cv2.imshow("NumPlate", img)
            cv2.waitKey(0)

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
            self.cnt=self.cnt+1
            self.nowRow+=1
            self.nowCol+=1
            event.setDropAction(Qt.CopyAction)
            file_path = event.mimeData().urls()[0].toLocalFile()
            self.set_image(file_path)
            cv_image=cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            imp=ImageProcessing(cv_image, self.cnt)
            fileText=file_path.split('/')
            recoInfo = {'axis':imp.axis,'text':imp.lastString, 'filename':fileText[-1], 'filepath':file_path}
            self.tableViewer.set_data(self.cnt, recoInfo)
            event.accept()
        else:
            event.ignore()

    def set_image(self, file_path):
        self.photoViewer.setPixmap(QPixmap(file_path))

    def selectCell(self, row, column):
        if self.tableViewer.item(row,column):
            self.nowRow=row
            self.nowCol=column
            self.tableViewer.nowRow=row
            self.photoViewer.setPixmap(QPixmap(self.tableViewer.item(row, 3).text()))


class ImageProcessing(AppDemo):
	def __init__(self, image, cnt):
		super(ImageProcessing, self).__init__()
		self.image = image
		self.lastString=""
		self.cnt = cnt
		self.start = time.time()
		self.axis = []
		self.isTwoLine = True
		self.mask = cv2.imread('./images/plateMask.jpg', 0)
		self.beforeProcessing(False, 0)
		self.startProcessing()

	def startProcessing(self):
		# 변수 선언
		self.minNumCnt = 3
		self.contours_dict = []
		self.pos_cnt = list()
		self.doubleLineBoxes = list()
		self.doubleLineBoxInfo = []
		self.innerBoxInDouble = list()
		self.toDelNum = []
		self.last_sorted_chars=[]
		self.X_IQR=1.5
		self.Y_IQR=1.5



		#### ADD || EDIT
		self.findContour()
		self.pickNumContour()
		self.pickContourGroup()
		#print("lastGroup!!!! ", self.lastGroup)
		self.check2LinePlateSize()

		if self.isTwoLine : # 2줄 번호판이면
			self.perspectiveTransformTwoLine()
			self.addBorderTwoLine()
			self.beforeProcessing(True, 1)
			#1005+200 = 1205 / 510+200 = 710 -> X:(100, 1105) / Y:(100, 610)
			upImg=self.th[100:270+12, 100:1105]
			self.textReco(self.th)
			#cv2.imshow("Up Image", self.th)
			#cv2.waitKey(0)
			self.text+=' '
			self.lastString+=self.text

			self.beforeProcessing(True, 2)
			downImg = self.th[100:440, 100:1105]
			self.textReco(self.th)
			#cv2.imshow("Down Image", self.th)
			#cv2.waitKey(0)
			self.lastString += self.text
			print("last string : ", self.lastString)

			img_cat = cv2.vconcat([upImg, downImg])
			img_cat = cv2.resize(img_cat,(335,170+4))
			cv2.imwrite('images/lastImage(%i).jpg' % self.cnt, img_cat)


			# last Image 정제해서 줘야 함!!!!!!!!!!!!!!!!!!!!!!!!

		else : # 1줄 번호판이면
			self.perspectiveTransformOneLine()
			self.addBorderOneLine()
			self.beforeProcessing(True, 0)
			self.textReco(self.th)
			self.lastString = self.text
			cv2.imwrite('images/lastImage(%i).jpg' % self.cnt, self.th)


	def check2LinePlateSize(self):
		#입력받은 Num Contour 묶음(lastGroup)의 가로, 세로 길이로 2줄인지 1줄인지 판단한다.
		self.last_sorted_chars = sorted(self.lastGroup, key=lambda x: x['cx'])
		width = self.last_sorted_chars[-1]['x']+self.last_sorted_chars[-1]['w'] - self.last_sorted_chars[0]['x']
		height = self.last_sorted_chars[0]['h']
		ratio = width/height

		# 2줄 번호판 Num 부분 비율 2.96
		if ratio > 2.65 and ratio < 3.25 :
			self.isTwoLine = True
		# 1줄 번호판 Num 부분 비율 : 5.5
		elif ratio > 4.1 and ratio < 5.8 :
			self.isTwoLine = False
		else :
			print("Out of Plate Ratio!!!!!!! - Its ratio is ", ratio)

		'''##### 아래는 구 코드임(2줄 번호판 외곽의 비율을 판단하는 코드)
		# d에 두 줄 번호판의 비율, 넓이에 해당하는 박스 정보들을 담는다.
		ratio = [1.6, 1.9]
		area = [10000, 25000]

		# contour 추리기 1st
		count = 0
		for d in self.contours_dict:
			rect_area = d['w'] * d['h']
			aspect_ratio = d['w'] / d['h']

			if (aspect_ratio >= ratio[0]) and (aspect_ratio <= ratio[1]) and (rect_area >= area[0]) and (rect_area <= area[1]):
				d['idx'] = count
				count += 1
				self.doubleLineBoxes.append(d)

		# 1차 추린 결과 저장
		orig_img = self.image.copy()
		for d in self.doubleLineBoxes:
			cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)'''


	def beforeProcessing(self, is2nd, i):
		if(is2nd==True):
			if i==1:
				img=self.borderUp
			elif i==2:
				img=self.borderDown
			else :
				img=self.border
		else :
			img=self.image
		# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		rgbGray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)


		if is2nd :
			img_blurred = cv2.medianBlur(rgbGray, 5)
		else :
			img_blurred = cv2.bilateralFilter(rgbGray, -1, 3, 3)

		thresh_sauvola = threshold_sauvola(img_blurred, 31)
		binary_sauvola = img_blurred > thresh_sauvola
		binary_sauvola = (binary_sauvola).astype('uint8')
		binary_sauvola = binary_sauvola * 255
		self.th = binary_sauvola
		self.th = (self.th).astype('uint8')

		#cv2.imshow("Licence Plate Second Processing", self.th)
		#cv2.waitKey(0)
		"""
		if self.isTwoLine :
			canny = cv2.Canny(img, 70, 30)
			self.th = canny
		else:
		"""

		if is2nd:
			if i==1:
				cv2.imwrite('images/secondNumPlateUp(%i).jpg' % self.cnt, self.th)
			elif i==2:
				cv2.imwrite('images/secondNumPlateDown(%i).jpg' % self.cnt, self.th)
			else :
				cv2.imwrite('images/secondNumPlate(%i).jpg' % self.cnt, self.th)
		else:
			cv2.imwrite('images/binaryzation(%i).jpg' % self.cnt, self.th)

	def findContour(self):
		contours, _ = cv2.findContours(self.th, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
		orig_img = self.image.copy()

		for contour in contours:
			x, y, w, h = cv2.boundingRect(contour)
			cv2.rectangle(orig_img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)

			self.contours_dict.append({
				'contour': contour,
				'x': x,
				'y': y,
				'w': w,
				'h': h,
				'cx': x + (w / 2),
				'cy': y + (h / 2)
			})
		cv2.imwrite('images/probBoxes(%i).jpg' % self.cnt, orig_img)

	'''def pickNumIn2LinePlate(self):
		is2=False
		for i, r in enumerate(self.doubleLineBoxes):
			maxY = 0
			tmpList = list()
			for d in self.contours_dict:
				if (r['contour'].all == d['contour'].all):
					continue
				rect_area = d['w'] * d['h']  # 영역 크기
				aspect_ratio = d['w'] / d['h']

				if (r['x'] < d['x']) and (r['y'] < d['y']) and (r['x'] + r['w'] > d['x'] + d['w']) and (
						r['y'] + r['h'] > d['y'] + d['h']):
					if (aspect_ratio >= 0.25) and (aspect_ratio <= 0.8) and (rect_area >= 600) and (rect_area <= 2000):
						tmpList.append(d)
						#self.pos_cnt.append(d)
						maxY = max(maxY, d['y'])

			if len(tmpList) >= self.minNumCnt:
				self.doubleLineBoxInfo.append({
					'x': r['x'] + 3,
					'y': r['y'] + 3,
					'w': r['w'] - 6,
					'h': r['h'] - 6,
					'divY': maxY - 6,
					'contour': d['contour']
				})
				for d in tmpList:
					self.innerBoxInDouble.append(d)
					# 근데 이건 그룹으로 만든게 아니라, 따로따로 저장한거다 !!!

				### EDIT!!!!!!!!!
				is2=True
				break

		return is2
		#pickNumContour2nd Start!!
'''

	def pickNumContour(self):
		##  Num 컨투어 추리기 1st
		count = 0
		for d in self.contours_dict:
			rect_area = d['w'] * d['h']
			aspect_ratio = d['w'] / d['h']

			if (aspect_ratio >= 0.2) and (aspect_ratio <= 0.8) and (rect_area >= 600) and (rect_area <= 2000):
				d['idx'] = count
				count += 1
				self.pos_cnt.append(d)

		# 1차 추린 결과 저장
		orig_img = self.image.copy()
		for d in self.pos_cnt:
			cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
		cv2.imwrite('images/contourBox1st(%i).jpg' % self.cnt, orig_img)

		## 컨투어 추리기 2nd
		self.result_idx = self.find_number(self.pos_cnt)
		matched_result = []
		for idx_list in self.result_idx:
			matched_result.append(np.take(self.pos_cnt, idx_list))

		# 2차 추린 결과 저장
		orig_img = self.image.copy()
		for r in matched_result:
			for d in r:
				cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
		cv2.imwrite('images/contourBox2nd(%i).jpg' % self.cnt, orig_img)

		return matched_result

	def find_number(self, contour_list):
		MAX_DIAG_MULTIPLYER = 8  # contourArea의 대각선 x7 안에 다음 contour가 있어야함
		MAX_ANGLE_DIFF = 10.0  # contour와 contour 중심을 기준으로 한 각도가 설정각 이내여야함 --> 카메라 각도가 너무 틀어져있으면 이 각도로 측정되지 않을 수 있음에 주의...
		MAX_AREA_DIFF = 0.8  # contour 사각형의 면적 차이가 설정값보다 크면 인정하지 x
		MAX_WIDTH_DIFF = 0.8  # contour 사각형의 너비 차이가 설정값보다 크면 인정 x
		MAX_HEIGHT_DIFF = 0.2  # contour 사각형의 높이 차이가 설정값보다 크면 인정 x
		MIN_N_MATCHED = 4  # 위의 조건을 따르는 contour가 최소 3개 이상이어야 번호판으로 인정

		if (len(contour_list) < 3):
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

				# 면적/너비/높이의 비율 (기준 contour 대비)
				area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
				width_diff = abs(d1['w'] - d2['w']) / d1['w']
				height_diff = abs(d1['h'] - d2['h']) / d2['h']

				# 조건에 맞는 idx만을 matched_contours_idx에 append할 것이다.
				if distance < diag_len * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF \
						and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF \
						and height_diff < MAX_HEIGHT_DIFF :
					matched_contour_idx.append(d2['idx'])
			matched_contour_idx.append(d1['idx'])

			# 앞서 정한 후보군의 갯수보다 적으면 탈락
			if len(matched_contour_idx) < MIN_N_MATCHED:
				continue
			#elif len(matched_contour_idx) >= MAX_N_MATCHED:
			#	continue

			# 최종 contour 묶음을 입력
			matched_result_idx.append(matched_contour_idx)

			# 최종 묶음에 들지 못한 애들은 따로 구분.
			unmatched_contour_idx = []
			for d4 in contour_list:
				if d4['idx'] not in matched_contour_idx:
					unmatched_contour_idx.append(d4['idx'])

			if unmatched_contour_idx is None:
				break

			# 묶음이 안된 애 전체 정보를 unmatched_contour에 대입.
			unmatched_contour = np.take(self.pos_cnt, unmatched_contour_idx)

			# 묶음 안된 애들에 대해 재귀로 돈다.
			recursive_contour_list = self.find_number(unmatched_contour)

			if recursive_contour_list is None:
				break

			# 최종 리스트에 추가
			for idx in recursive_contour_list:
				matched_result_idx.append(idx)
			break
		return matched_result_idx

	def pickContourGroup(self):
		matched_result = []
		if self.result_idx is None :
			print('result_idx is None! in pickContourGroup()')
		for idx_list in self.result_idx:
			matched_result.append(np.take(self.pos_cnt, idx_list))
		matched_axis, resultGroupDict = self.checkPlateRatio(matched_result)
		self.lastGroup = self.deleteOutlier(matched_axis, resultGroupDict)


	def checkOutlier(self, arr, idx, x):
		# 좌표 값 정규분포에서 이상치 뽑아내기
		if len(arr)%2==0 :
			midIdx=(len(arr)-1)/2

		else :
			midIdx = (len(arr)/2 + (len(arr)/2)-1)/2

		if midIdx%2==1 :
			q1 = arr[int((midIdx - 1) / 2)][idx]
			q3 = arr[int(midIdx + (midIdx - 1) / 2 + 1)][idx]
			iqr = q3 - q1
		else :
			q1 = (arr[int((midIdx) / 2)][idx] + arr[int((midIdx) / 2 - 1)][idx])/2
			q3 = (arr[int(midIdx + midIdx/2)][idx]+arr[int(midIdx + midIdx/2 + 1)][idx])/2
			iqr = q3 - q1

		downOutlier = q1-x*iqr
		upOutlier = q3+x*iqr
		arrLen=len(arr)
		isPass=True
		toDelIdx=[]


		for i in range(arrLen):
			if arr[i][idx] > upOutlier:
				if arr[i][idx] not in self.toDelNum :
					toDelIdx.append(i)
					self.toDelNum.append(arr[i][2])
				isPass=False

			elif arr[i][idx] < downOutlier :
				if arr[i][idx] not in self.toDelNum:
					toDelIdx.append(i)
					self.toDelNum.append(arr[i][2])
				isPass=False

		toDelIdx.sort(reverse=True)
		for i in range(len(toDelIdx)) :
			del arr[toDelIdx[i]]

		if isPass :
			return arr
		else :
			arr=self.checkOutlier(arr, idx, x)
			return arr

	def deleteOutlier(self,matched_axis, resultGroupDict):
		# 이상치 제거하기
		#print(matched_axis)
		sorted_contour = sorted(matched_axis, key=lambda x: x[0])
		afterX = self.checkOutlier(sorted_contour, 0, self.X_IQR)
		sorted_contour = sorted(afterX, key=lambda x: x[1])
		afterXY = self.checkOutlier(sorted_contour, 1, self.Y_IQR)
		# 제거할 애 거르기
		lastGroup = []
		for i, matched_chars in enumerate(resultGroupDict):
			if matched_chars['idx'] not in self.toDelNum:
				lastGroup.append(matched_chars)
		return lastGroup

	def checkPlateRatio(self, matched_result):
		# 번호판 영역인지 아닌지, 2줄짜리 번호판인지 아닌지 체크하는 함수. ROI의 설정 하려면 전처리 전에 중앙위주로 잘라주던가 하면 될 듯 싶음.
		#print("matched_result LEN : ", len(matched_result))
		matched_axis = []
		resultGroupDict = []

		for i, matched_chars in enumerate(matched_result):
			#print("before matched_chars LEN : ", len(matched_chars))
			# lambda 함수로 소팅. 'cx'의 키값을 오름차순으로 정렬한다.
			sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
			tempSum=0
			isPlate=True

			for i in range(len(sorted_chars) - 1):
				# 앞 글자의 우측 X좌표와 뒷 글자의 좌측 X 좌표의 비교를 통해 간격 계산
				firEndX = sorted_chars[i]['x'] + sorted_chars[i]['w']
				secStartX = sorted_chars[i + 1]['x']
				distanceX = secStartX - firEndX
				tempSum += distanceX
				if distanceX < -5:
					isPlate = False
					break
			if tempSum < 5:
				isPlate = False

			if not isPlate:  # 번호판이 아니라고 판정되면
				print('it is not a Plate!!!!!!!!')
				continue

			else:
				resultGroupDict = sorted_chars
				for i in range(len(sorted_chars)):
					matched_axis.append([sorted_chars[i]['x'], sorted_chars[i]['y'], sorted_chars[i]['idx']])
				break
		#print("result Group Dict : ", resultGroupDict)
		return matched_axis, resultGroupDict

	def perspectiveTransformOneLine(self):
		orig_img = self.image.copy()

		# lambda 함수로 소팅. 'cx'의 키값을 오름차순으로 정렬한다. (contours들이 좌측부터 차례대로 정렬됨)
		sorted_chars = self.last_sorted_chars

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
		self.axis = pts1

		# 다각형 선을 그리기 위해서 (좌상->좌하->우하->우상)
		ptsPoly = np.array(
			[[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightDown['x'], rightDown['y']],
			 [rightUp['x'], rightUp['y']]])

		M = cv2.getPerspectiveTransform(pts1, pts2)
		dst = cv2.warpPerspective(self.th, M, (520, 110))
		self.numPlate = dst.copy()

		# 점과 선 그리기
		orig_img = cv2.polylines(orig_img, [ptsPoly], True, (0, 255, 255), 2)
		cv2.circle(orig_img, (leftUp['x'], leftUp['y']), 10, (255, 0, 0), -1)
		cv2.circle(orig_img, (leftDown['x'], leftDown['y']), 10, (0, 255, 0), -1)
		cv2.circle(orig_img, (rightUp['x'], rightUp['y']), 10, (0, 0, 255), -1)
		cv2.circle(orig_img, (rightDown['x'], rightDown['y']), 10, (0, 0, 0), -1)

		cv2.imwrite('images/numPlate(%i).jpg' % self.cnt, self.numPlate)
		cv2.imwrite('images/contourBox(%i).jpg' % self.cnt, orig_img)

	def perspectiveTransformTwoLine(self):
		# 영역 다각형 그리기
		# 영역 뽑아서 원근변환
		orig_img = self.image.copy()
		sorted_chars = self.last_sorted_chars

		dy=0
		cnt=1

		# lambda 함수로 소팅. 'cx'의 키값을 오름차순으로 정렬한다. (contours들이 좌측부터 차례대로 정렬됨)
		#print("sorted_chars LEN : ", len(sorted_chars))
		plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
		plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

		while cnt < 3 :
			if cnt == 1:
				# 번호판 영역의 네 모서리 좌표를 저장한다.
				leftUp = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y']}
				leftDown = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y'] + sorted_chars[0]['h']}
				rightUp = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'], 'y': sorted_chars[-1]['y']}
				rightDown = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'],
							 'y': sorted_chars[-1]['y'] + sorted_chars[-1]['h']}
				dy = int((leftDown['y'] - leftUp['y']) / 2)+2 ############### EDIT!! (상수멈춰! 비율에 맞게 바꾸는 작업 하기)
				downLineSize=[1005, 340]

				# 원근 변환을 위해 input 좌표와 output 좌표를 기록 (좌상->좌하->우상->우하) (번호판 크기에 따라서 pts2는 달라질 수 있음에 주의)
				pts1 = np.float32(
					[[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightUp['x'], rightUp['y']],
					 [rightDown['x'], rightDown['y']]])
				pts2 = np.float32([[0, 0], [0, 340], [1005, 0], [1005, 340]])
				self.axis.append([leftUp['x'], leftUp['y']])
				self.axis.append([leftDown['x'], leftDown['y']])

				# 다각형 선을 그리기 위해서 (좌상->좌하->우하->우상)
				ptsPoly = np.array(
					[[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightDown['x'], rightDown['y']],
					 [rightUp['x'], rightUp['y']]])
				M = cv2.getPerspectiveTransform(pts1, pts2)
				dst = cv2.warpPerspective(self.th, M, (1005, 340))
				#self.numPlate = dst.copy()
				# 점과 선 그리기
				orig_img = cv2.polylines(orig_img, [ptsPoly], True, (0, 255, 255), 2)
				cv2.circle(orig_img, (leftUp['x'], leftUp['y']), 3, (255, 0, 0), -1)
				cv2.circle(orig_img, (leftDown['x'], leftDown['y']), 3, (0, 255, 0), -1)
				cv2.circle(orig_img, (rightUp['x'], rightUp['y']), 3, (0, 0, 255), -1)
				cv2.circle(orig_img, (rightDown['x'], rightDown['y']), 3, (0, 0, 0), -1)

				self.numPlateDown = dst.copy()

			elif cnt == 2 : # 윗줄
				leftUp = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y'] - dy}
				rightUp = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'], 'y': sorted_chars[-1]['y'] - dy}
				leftDown = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y']-5}
				rightDown = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'],
							 'y': sorted_chars[-1]['y']-5}
				upLineSize = [1005, 170]

				# 원근 변환을 위해 input 좌표와 output 좌표를 기록 (좌상->좌하->우상->우하) (번호판 크기에 따라서 pts2는 달라질 수 있음에 주의)
				pts1 = np.float32(
					[[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightUp['x'], rightUp['y']],
					 [rightDown['x'], rightDown['y']]])
				pts2 = np.float32([[0, 0], [0, 170], [1005, 0], [1005, 170]])
				self.axis.append([rightUp['x'], rightUp['y']])
				self.axis.append([rightDown['x'], rightDown['y']])


				# 다각형 선을 그리기 위해서 (좌상->좌하->우하->우상)
				ptsPoly = np.array(
					[[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightDown['x'], rightDown['y']],
					 [rightUp['x'], rightUp['y']]])
				M = cv2.getPerspectiveTransform(pts1, pts2)
				dst = cv2.warpPerspective(self.th, M, (1005, 170))


				# 마스킹
				resizedMask = cv2.resize(self.mask, dsize=(1005, 170), interpolation=cv2.INTER_AREA).astype('uint8')
				#whiteImg = np.ones((1005, 170, 3), np.uint8) * 255
				dst = cv2.bitwise_or(dst, dst, mask=resizedMask)

				#self.numPlate = dst.copy()
				# 점과 선 그리기
				orig_img = cv2.polylines(orig_img, [ptsPoly], True, (0, 255, 255), 2)
				cv2.circle(orig_img, (leftUp['x'], leftUp['y']), 3, (255, 0, 0), -1)
				cv2.circle(orig_img, (leftDown['x'], leftDown['y']), 3, (0, 255, 0), -1)
				cv2.circle(orig_img, (rightUp['x'], rightUp['y']), 3, (0, 0, 255), -1)
				cv2.circle(orig_img, (rightDown['x'], rightDown['y']), 3, (0, 0, 0), -1)

				self.numPlateUp = dst.copy()
			cnt+=1

		#cv2.imwrite('images/numPlate/numPlate(%i).jpg' % self.cnt, self.numPlate)
		cv2.imwrite('images/contourBox(%i).jpg' % self.cnt, orig_img)
		#cv2.imshow("contourBox!!!", orig_img)
		#cv2.waitKey(0)

	def addBorderOneLine(self):
		bordersize = 100

		self.border = cv2.copyMakeBorder(
			self.numPlate,
			top=bordersize,
			bottom=bordersize,
			left=bordersize,
			right=bordersize,
			borderType=cv2.BORDER_CONSTANT,
			value=[255, 255, 255]
		)
		cv2.imwrite('images/Border(%i).jpg' % self.cnt, self.border)

	def addBorderTwoLine(self):
		self.numPlateUp = 255 - self.numPlateUp
		self.numPlateDown = 255 - self.numPlateDown
		bordersize = 100

		self.borderUp = cv2.copyMakeBorder(
			self.numPlateUp,
			top=bordersize,
			bottom=bordersize,
			left=bordersize,
			right=bordersize,
			borderType=cv2.BORDER_CONSTANT,
			value=[255, 255, 255]
		)
		self.borderDown = cv2.copyMakeBorder(
			self.numPlateDown,
			top=bordersize,
			bottom=bordersize,
			left=bordersize,
			right=bordersize,
			borderType=cv2.BORDER_CONSTANT,
			value=[255, 255, 255]
		)
		cv2.imwrite('images/Border/BorderUp(%i).jpg' % self.cnt, self.borderUp)
		cv2.imwrite('images/Border/BorderDown(%i).jpg' % self.cnt, self.borderDown)

	def textReco(self, image):
		self.text = pytesseract.image_to_string(image, lang='kor', config='--psm 7')
		self.text = self.text.split('\x0c')[0]
		self.text = self.text.split('\n')[0]
		print('text = ',self.text)
		if not self.text :
			self.text='인식 실패'
		print("처리 경과 시간 :", time.time() - self.start)
		print('-----------------------------------------')


if __name__ == '__main__':
	app = QApplication(sys.argv)
	demo = AppDemo()
	demo.show()
	sys.exit(app.exec_())