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

# 설치한 tesseract 프로그램 경로 (64비트)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'


#########################################################
# 1. 이미지 불러오기, Gray 프로세싱
#########################################################

####### 파일명 for testing #########
image = cv2.imread('./image/car (3).jpg')
#19
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
rgbGray = cv2.cvtColor(imgRGB, cv2.COLOR_RGB2GRAY)



#########################################################
# 2. 블러 :
#########################################################

# 가우시안 블러/중앙값 블러/Bilateral필터 : 노이즈 줄이기(Bilateral은 경계선 유지)
#BilteralFilter Parameter : (src, 픽셀 지름, 컬러 고려 공간, 멀리있는 픽셀까지 고려할지)
img_gau_blurred = cv2.GaussianBlur(rgbGray, ksize=(5,5), sigmaX=0)
#img_blurred = cv2.medianBlur(img_blurred, 3)
img_bil_blurred = cv2.bilateralFilter(gray,-1,10,5)



#########################################################
# 3. 스레시 홀드
#########################################################

# 3-1. OpenCV
# 옵션에 대한 설명은 링크 참고 : https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html

# 기본 스레시홀드(OTSU를 이용하는건 히스토그램이 2개의 고점을 가질 때 효율적)
ret, th1 = cv2.threshold(img_bil_blurred,127,255,cv2.THRESH_BINARY)
th2 = cv2.threshold(img_bil_blurred, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Adaptive Threshold : 영역별로 스레시홀드
th3 = cv2.adaptiveThreshold(img_bil_blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,11,2)
th4 = cv2.adaptiveThreshold(img_bil_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,13,2)

# Canny()
th5 = cv2.Canny(gray, 30, 50)

# 3-2. Skimage
# Niblack()
thresh_niblack = threshold_niblack(rgbGray,25, k=1.0)
binary_niblack = rgbGray > thresh_niblack
# 비교의 결과가 bool이라서 uint8 형식으로 바꿔줘야 나중에 cv2관련 함수를 사용할 때 문제가 안생김(cv2와 plt의 차이땜에)
binary_niblack=(binary_niblack).astype('uint8')
binary_niblack = binary_niblack*255
th6 = binary_niblack

# Sauvola()
thresh_sauvola = threshold_sauvola(rgbGray, 25)
binary_sauvola = rgbGray > thresh_sauvola
# 비교의 결과가 bool이라서 uint8 형식으로 바꿔줘야 나중에 cv2관련 함수를 사용할 때 문제가 안생김(cv2와 plt의 차이땜에)
binary_sauvola=(binary_sauvola).astype('uint8')
binary_sauvola = binary_sauvola*255
th7 = binary_sauvola
# 결과는 다른 것들과 다르게 bool 형태로 나옴.. -> unit8


# 한 창에 띄우기
titles = ['Original','Basic', 'Basic: Otsu','Adaptive: Mean','Adaptive: Gaussian', 'Canny', 'Niblack', 'Sauvola', 'Histogram']
images = [gray, th1, th2, th3, th4, th5, th6, th7, gray]

for i in range(9):
	plt.subplot(3,3,i+1)
	if i+1<7 : # OpenCV에서 제공하는 threshold
		plt.imshow(images[i], cmap='gray')
	elif i+1==9: # histogram (도수분포표)
		plt.hist(image.ravel(), 256, [0,256]);
	else : # skimage에서 제공하는 threshold
		plt.imshow(images[i], cmap=plt.cm.gray)

	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])
plt.show()



#########################################################
# 4. Contours 경계선 찾기
#########################################################

# 이미지를 읽을 때 OpenCV를 사용하면 BGR형식으로 받는데, plt의 경우 RGB로 보여준다. 그래서 plt로 show 할 때는 변환해주는 과정을 거쳐야 한다.
## 그리고 skimage를 통해 전처리를 한 경우 결과값이 8비트가 아니라서 OpenCV랑 병행해서 사용할 때 자꾸 오류가 난다.
## astype('unit8')로 변환해줘야 한다.
th7=(th7).astype('uint8')
th6=(th6).astype('uint8')

orig_img=image.copy()
thr=th7

# 언더스코어(_) : 특정 위치의 값을 무시하기 위함.
# findContours() 함수의 첫 번째 리턴값만 필요하므로 언더스코어로 생략한 것임.
contours,_ = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_dict = []		# 윤곽 전체 정보
pos_cnt = list()

for contour in contours:
	x, y, w, h = cv2.boundingRect(contour)
	# boundingRect(): 인자로 받은 contour에 외접하고 똑바로 세워진 직사각형의 좌상단 꼭지점 좌표(x, y)와 가로 세로 폭을 리턴함.
	cv2.rectangle(orig_img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 255, 0), thickness=1)
	# cv2.rectangle(이미지, 중심 좌표, 반지름, 색상, 두께) : 사각형 그리기

	# insert to dict
	contours_dict.append({
		'contour': contour,
		'x': x,
		'y': y,
		'w': w,
		'h': h,
		'cx': x + (w / 2), # 중앙 x
		'cy': y + (h / 2)  # 중앙 y
	})
# figure() : 그림의 크기. (피겨는 그림을 그리는 캔버스를 의미한다.)
plt.figure(figsize=(12, 8))
plt.imshow(orig_img[:, :,::-1])
plt.show()



#########################################################
# 5. Contours 1차 추리기 - 크기, 비율
#########################################################

orig_img = image.copy()
count = 0

for d in contours_dict:
	rect_area = d['w'] * d['h']  # 영역 크기
	aspect_ratio = d['w'] / d['h']

	# 이 부분을 내가 원하는 Contour 사각형의 비율, 넓이로 변경해줘야 함.
	# 이미지에 따라 값이 바뀔 수 있으므로 이미지 환경을 통일시켜 주는 것이 좋을 것 같음.(어짜피 환경에서는 동일한 설정의 이미지가 입력될테니까..)
	# 그리고 넓이 때문에 예전 번호판의 경우 윗줄 인식이 안된다는 점 인식하기...
	if (aspect_ratio >= 0.25) and (aspect_ratio <= 1.0) and (rect_area >= 600) and (rect_area <= 2000):
		cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
		d['idx'] = count
		count += 1
		pos_cnt.append(d)

plt.figure(figsize=(12, 8))
plt.imshow(orig_img[:, :,::-1])
plt.show()



#########################################################
# 6. Contour 2차 추리기 (배열)
#########################################################
MAX_DIAG_MULTIPLYER = 8  # contourArea의 대각선 x7 안에 다음 contour가 있어야함
MAX_ANGLE_DIFF = 10.0  # contour와 contour 중심을 기준으로 한 각도가 설정각 이내여야함 --> 카메라 각도가 너무 틀어져있으면 이 각도로 측정되지 않을 수 있음에 주의...
MAX_AREA_DIFF = 0.8  # contour간에 면적 차이가 설정값보다 크면 인정하지 x
MAX_WIDTH_DIFF = 0.8  # contour간에 너비 차이가 설정값보다 크면 인정 x
MAX_HEIGHT_DIFF = 0.3  # contour간에 높이 차이가 크면 인정 x
MIN_N_MATCHED = 4  # 위의 조건을 따르는 contour가 최소 3개 이상이어야 번호판으로 인정
#MAX_N_MATCHED = 8
orig_img = image.copy()


def find_number(contour_list):
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
		#elif len(matched_contour_idx) >= MAX_N_MATCHED:
		#	continue

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


result_idx = find_number(pos_cnt)


matched_result = []
for idx_list in result_idx:
	matched_result.append(np.take(pos_cnt, idx_list))

# pos_cnt 시각화
for r in matched_result:
	for d in r:
		cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)

plt.figure(figsize=(20, 20))
plt.imshow(orig_img[:, :, ::-1])
plt.show()


#########################################################
# 여기서 도출된 묶음들이 번호판인지 추가로 검사하는 알고리즘 필요  #
#########################################################


#########################################################
# 7. 회전(변환) 시키기
#########################################################
numPlate = image.copy()

for i, matched_chars in enumerate(matched_result):
	orig_img = image.copy()

	# lambda 함수로 소팅. 'cx'의 키값을 오름차순으로 정렬한다. (contours들이 좌측부터 차례대로 정렬됨)
	sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

	########################## 동떨어진 contour가 있는지 추세 확인

	##########################

	# 0번째 중앙 x좌표에서 마지막 중앙 x좌표까지의 길이
	plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
	plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2


	# 번호판 영역의 네 모서리 좌표를 저장한다.
	leftUp = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y']}
	leftDown = {'x': sorted_chars[0]['x'], 'y': sorted_chars[0]['y'] + sorted_chars[0]['h']}
	rightUp = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'], 'y': sorted_chars[-1]['y']}
	rightDown = {'x': sorted_chars[-1]['x'] + sorted_chars[-1]['w'], 'y': sorted_chars[-1]['y'] + sorted_chars[-1]['h']}

	# 원근 변환을 위해 input 좌표와 output 좌표를 기록 (좌상->좌하->우상->우하) (번호판 크기에 따라서 pts2는 달라질 수 있음에 주의)
	pts1 = np.float32([[leftUp['x'], leftUp['y']], [leftDown['x'], leftDown['y']], [rightUp['x'], rightUp['y']], [rightDown['x'], rightDown['y']]])
	pts2 = np.float32([[0, 0], [0, 110], [520, 0], [520, 110]])
	# 다각형 선을 그리기 위해서 (좌상->좌하->우하->우상)
	ptsPoly = np.array([[leftUp['x'],leftUp['y']], [leftDown['x'], leftDown['y']], [rightDown['x'], rightDown['y']], [rightUp['x'], rightUp['y']]])

	##########################
	isPlate = True
	is2Line = False

	tempSum=0
	# 글자영역 컨투어 소팅 후 두 컨투어씩 비교하여 서로 간격이 너무 좁으면(거의 붙어있으면) 번호판 아님.
	for i in range(len(sorted_chars) - 1):
		firEndX = sorted_chars[i]['x'] + sorted_chars[i]['w']
		secStartX = sorted_chars[i + 1]['x']
		distanceX = secStartX - firEndX
		tempSum += distanceX
		if distanceX < -5:
			isPlate = False
			break
	if tempSum<10 :
		isPlate=False

	# 글자 영역 소팅 후 받아와서 좌상단 우하단 좌표까지 알아낼 수 있는 상태가 되면
	# 비율로 2줄짜리인지 1줄 짜리인지 검사를 해야 함.
	width = pts1[2][0] - pts1[0][0]
	height = pts1[1][1] - pts1[0][1]
	if width / height < 3:
		is2Line = True

	if not isPlate :
		print('it is not a Plate!!!!!!!!')
		continue

	if is2Line:
		# 두 줄짜리 번호판일 때 해줘야 할 일...
		print('it is 2 Line Num Plate@@@@@@@@')
	####################################
	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(thr, M, (520, 110))
	numPlate = dst.copy()

	# 점과 선 그리기
	orig_img = cv2.polylines(orig_img, [ptsPoly], True, (0, 255, 255),2)
	cv2.circle(orig_img, (leftUp['x'], leftUp['y']), 10, (255, 0, 0), -1)
	cv2.circle(orig_img, (leftDown['x'], leftDown['y']), 10, (0, 255, 0), -1)
	cv2.circle(orig_img, (rightUp['x'], rightUp['y']), 10, (0, 0, 255), -1)
	cv2.circle(orig_img, (rightDown['x'], rightDown['y']), 10, (0, 0, 0), -1)

	plt.subplot(121), plt.imshow(orig_img), plt.title('image')
	plt.subplot(122), plt.imshow(dst,'gray'), plt.title('Perspective')
	plt.show()

	# crop하고 패딩 주기 전 Test로 OCR 인식
	print('------------------------\nBefore Padding : ')
	text = pytesseract.image_to_string(dst, lang='kor', config='--psm 7')
	print(text)

	if isPlate:
		break


#########################################################
# 8. 이미지에 패딩 주기 (테두리 삽입)                        #
#########################################################
bordersize = 100

#numPlate = cv2.cvtColor(numPlate, cv2.COLOR_RGB2BGR)

border = cv2.copyMakeBorder(
    numPlate,
    top=bordersize,
    bottom=bordersize,
    left=bordersize,
    right=bordersize,
    borderType=cv2.BORDER_CONSTANT,
    value=[255, 255, 255]
)

plt.subplot(121), plt.imshow(numPlate,'gray'), plt.title('origin')
plt.subplot(122), plt.imshow(border,'gray'), plt.title('border')
plt.show()

# Test로 OCR 인식
print('------------------------\nAfter Padding : ')
text = pytesseract.image_to_string(border, lang='kor', config='--psm 7')
print(text)



#########################################################
# 후처리....
#########################################################


# 블러
img_blurred = cv2.medianBlur(border, 5)
#gau_blurred = cv2.GaussianBlur(border, ksize=(0,0), sigmaX=1)
#img_blurred = cv2.bilateralFilter(border,20,20,250)

ret, th1 = cv2.threshold(img_blurred,127,255,cv2.THRESH_BINARY)
th2 = cv2.threshold(img_blurred, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Adaptive Threshold : 영역별로 스레시홀드
th3 = cv2.adaptiveThreshold(img_blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,11,2)
th4 = cv2.adaptiveThreshold(img_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,13,2)

# Canny()
th5 = cv2.Canny(border, 30, 50)

# 3-2. Skimage
# Niblack()
thresh_niblack = threshold_niblack(img_blurred,25, k=0.8)
binary_niblack = img_blurred > thresh_niblack
# 비교의 결과가 bool이라서 uint8 형식으로 바꿔줘야 나중에 cv2관련 함수를 사용할 때 문제가 안생김(cv2와 plt의 차이땜에)
binary_niblack=(binary_niblack).astype('uint8')
binary_niblack = binary_niblack*255
th6 = binary_niblack


# Sauvola()
thresh_sauvola = threshold_sauvola(img_blurred, 25)
binary_sauvola = img_blurred > thresh_sauvola
# 비교의 결과가 bool이라서 uint8 형식으로 바꿔줘야 나중에 cv2관련 함수를 사용할 때 문제가 안생김(cv2와 plt의 차이땜에)
binary_sauvola=(binary_sauvola).astype('uint8')
binary_sauvola = binary_sauvola*255
th7 = binary_sauvola



titles = ['Original','Basic', 'Basic: Otsu','Adaptive: Mean','Adaptive: Gaussian', 'Canny', 'Niblack', 'Sauvola']
images = [border, th1, th2, th3, th4, th5, th6, th7]

print('=================================================')
# Test로 OCR 인식
for i in range(8):
	print('------------------------\n',titles[i],' : ')
	text = pytesseract.image_to_string(images[i], lang='kor', config='--psm 7')
	print(text)




# 한 창에 띄우기
for i in range(8):
	plt.subplot(3,3,i+1)
	if i+1<7 : # OpenCV에서 제공하는 threshold
		plt.imshow(images[i], cmap='gray')
	else : # skimage에서 제공하는 threshold
		plt.imshow(images[i], cmap=plt.cm.gray)

	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])
plt.show()