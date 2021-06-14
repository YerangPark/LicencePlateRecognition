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
# 으ㅏㅏㅏㅏ OJTSample2
# 전기차 저대비 NumTest2
# 옛날 번호판(눈) NumTest3
# 전기경차 OJTSample3
# 옛날 번호판 Before1
# 국지적으로 빛나는 Light3
# 밝은 Light2
image = cv2.imread('OJTSample2.jpg')
height, width, channel = image.shape
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# imgRGB = np.array(imgRGB)

# 원본 이미지 보여주기
#cv2.imshow("Image", image)
#cv2.waitKey(0)



#########################################################
# 2. 블러
#########################################################

# 가우시안 블러/중앙값 블러/Bilateral필터 : 노이즈 줄이기(Bilateral은 경계선 유지)
#BilteralFilter Parameter : (src, 픽셀 지름, 컬러 고려 공간, 멀리있는 픽셀까지 고려할지)
#img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
#img_blurred = cv2.medianBlur(img_blurred, 3)
img_blurred = cv2.bilateralFilter(gray,20,20,250)



#########################################################
# 3. 스레시 홀드
#########################################################

# 3-1. OpenCV
# 옵션에 대한 설명은 링크 참고 : https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html

# 기본 스레시홀드(OTSU를 이용하는건 히스토그램이 2개의 고점을 가질 때 효율적)
ret, th1 = cv2.threshold(img_blurred,127,255,cv2.THRESH_BINARY)
th2 = cv2.threshold(img_blurred, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Adaptive Threshold : 영역별로 스레시홀드
th3 = cv2.adaptiveThreshold(img_blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,15,4)
th4 = cv2.adaptiveThreshold(img_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,19,3)

# Canny()
th5 = cv2.Canny(gray, 30, 50)

# 3-2. Skimage
# Niblack()
thresh_niblack = threshold_niblack(gray,25, k=0.8)
binary_niblack = gray > thresh_niblack
th6 = binary_niblack

# Saubola()
thresh_sauvola = threshold_sauvola(gray, 25)
binary_sauvola = gray > thresh_sauvola
th7 = binary_sauvola


# 한 창에 띄우기
titles = ['Original','Basic', 'Basic: Otsu','Adaptive: Mean','Adaptive: Gaussian', 'Canny', 'Niblack', 'Saubola', 'Histogram']
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

# 이미지를 읽을 때 OpenCV를 사용하면 BGR형식으로 이미지를 표시하기 때문에 문제가 발생한다. 변환해주는 과정을 거친다.
## 그리고 skimage를 통해 전처리를 한 경우 결과값이 8비트가 아니라서 OpenCV랑 병행해서 사용할 때 자꾸 오류가 난다.
## astype('unit8')로 변환해줘야 한다.
th7=(th7).astype('uint8')
orig_img=image.copy()
thr=th4

contours,_ = cv2.findContours(thr,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours_dict = []
pos_cnt = list()
box1 = list()

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
# 5. Contours 추리기
#########################################################

orig_img = image.copy()
count = 0

for d in contours_dict:
	rect_area = d['w'] * d['h']  # 영역 크기
	aspect_ratio = d['w'] / d['h']

	# 이 부분을 내가 원하는 Contour 사각형의 비율, 넓이로 변경해줘야 함.
	# 이미지에 따라 값이 바뀔 수 있으므로 이미지 환경을 통일시켜 주는 것이 좋을 것 같음.(어짜피 환경에서는 동일한 설정의 이미지가 입력될테니까..)
	# 그리고 넓이 때문에 예전 번호판의 경우 윗줄 인식이 안된다는 점 인식하기...
	if (aspect_ratio >= 0.3) and (aspect_ratio <= 1.0) and (rect_area >= 800) and (rect_area <= 2000):
		cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
		d['idx'] = count
		count += 1
		pos_cnt.append(d)

plt.figure(figsize=(12, 8))
plt.imshow(orig_img[:, :,::-1])
plt.show()



#########################################################
# 6. 올바른 사이즈의 Contour가 자동차 번호판에 해당하는지 검사하기
#########################################################

MAX_DIAG_MULTIPLYER = 10  # contourArea의 대각선 x10 안에 다음 contour가 있어야함
MAX_ANGLE_DIFF = 12.0  # contour와 contour 중심을 기준으로 한 각도가 12.0 이내여야함
MAX_AREA_DIFF = 0.5  # contour간에 면적 차이가 0.5보다 크면 인정하지 x
MAX_WIDTH_DIFF = 0.8  # contour간에 너비 차이가 0.8보다 크면 인정 x
MAX_HEIGHT_DIFF = 0.2  # contour간에 높이 차이가 크면 인정 x
MIN_N_MATCHED = 3  # 위의 조건을 따르는 contour가 최소 3개 이상이어야 번호판으로 인정

orig_img = image.copy()

def find_number(contour_list):
	matched_result_idx = []

	# contour_list[n]의 keys = dict_keys(['contour', 'x', 'y', 'w', 'h', 'cx', 'cy', 'idx'])
	for d1 in contour_list:
		matched_contour_idx = []
		for d2 in contour_list:  # for문을 2번 돌면서 contour끼리 비교해줄 것임
			if d1['idx'] == d2['idx']:
				continue

			# 피타고라스로 대각 길이를 구하기 위해 dy와 dx를 구함.
			dx = abs(d1['cx'] - d2['cx'])  # 각각 중앙점 기준으로 가로 거리
			dy = abs(d1['cy'] - d2['cy'])  # 각각 중앙점 기준으로 세로 거리
			diag_len = np.sqrt(d1['w'] ** 2 + d1['w'] ** 2)

			# contour 중심간의 대각 거리 (유클리디언 거리(유사도))
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

			# 이제 조건에 맞는 idx만을 matched_contours_idx에 append할 것이다.
			if distance < diag_len * MAX_DIAG_MULTIPLYER and angle_diff < MAX_ANGLE_DIFF \
					and area_diff < MAX_AREA_DIFF and width_diff < MAX_WIDTH_DIFF \
					and height_diff < MAX_HEIGHT_DIFF:
				# 계속 d2를 번갈아 가며 비교했기에 지금 d2 넣어주고
				matched_contour_idx.append(d2['idx'])

		# d1은 기준이었으니 이제 append
		matched_contour_idx.append(d1['idx'])

		# 앞서 정한 후보군의 갯수보다 적으면 탈락
		if len(matched_contour_idx) < MIN_N_MATCHED:
			continue

		# 최종 contour를 입력
		matched_result_idx.append(matched_contour_idx)

		# 최종에 들지 못한 아닌애들도 한 번 더 비교
		unmatched_contour_idx = []
		for d4 in contour_list:
			if d4['idx'] not in matched_contour_idx:
				unmatched_contour_idx.append(d4['idx'])

		# np.take(a,idx)   a배열에서 idx를 뽑아냄
		unmatched_contour = np.take(pos_cnt, unmatched_contour_idx)

		# 재귀적으로 한 번 더 돌림
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
#plt.savefig("Plate_Contour")
plt.show()


#########################################################
# 7. 회전(변환) 시키기
#########################################################
PLATE_WIDTH_PADDING = 1.3
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

for i, matched_chars in enumerate(matched_result):
	orig_img = image.copy()

	# lambda함수로 소팅하는 방법을 채택. 'cx'의 키값을 오름차순으로 정렬한다. (contours들이 좌측부터 차례대로 정렬됨)
	sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])

	# 0번째 중앙 x좌표에서 마지막 중앙 x좌표까지의 길이
	plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
	plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

	##########################################################
	# ADD
	##########################################################
	# / 형태이면 1
	# 추후 딕셔너리 형태로 바꿔서 이용하는 것이 더 편할 것 같기도 하고,,, 이럴 때는 변수 선언을 어떻게 하는 것이 효율적일까? 생각해봐야 할 듯!
	leftDownX = sorted_chars[0]['x']
	leftDownY = sorted_chars[0]['y'] + sorted_chars[0]['h']
	rightUpX = sorted_chars[-1]['x'] + sorted_chars[-1]['w']
	rightUpY = sorted_chars[-1]['y']
	leftUpX = sorted_chars[0]['x']
	leftUpY = sorted_chars[0]['y']
	rightDownX = sorted_chars[-1]['x'] + sorted_chars[-1]['w']
	rightDownY = sorted_chars[-1]['y'] + sorted_chars[-1]['h']

	# 좌상->좌하->우상->우하
	pts1 = np.float32([[leftUpX, leftUpY], [leftDownX, leftDownY], [rightUpX, rightUpY], [rightDownX, rightDownY]])
	pts2 = np.float32([[0, 0], [0, 110], [520, 0], [520, 110]])
	###########
	tmpPts = np.array([[leftUpX, leftUpY], [leftDownX, leftDownY], [rightDownX, rightDownY], [rightUpX, rightUpY]])

	M = cv2.getPerspectiveTransform(pts1, pts2)
	dst = cv2.warpPerspective(thr, M, (520, 110))

	orig_img = cv2.polylines(orig_img, [tmpPts], True, (0, 255, 255),2)
	cv2.circle(orig_img, (leftUpX, leftUpY), 10, (255, 0, 0), -1)
	cv2.circle(orig_img, (leftDownX, leftDownY), 10, (0, 255, 0), -1)
	cv2.circle(orig_img, (rightUpX, rightUpY), 10, (0, 0, 255), -1)
	cv2.circle(orig_img, (rightDownX, rightDownY), 10, (0, 0, 0), -1)

	plt.subplot(121), plt.imshow(orig_img), plt.title('image')
	plt.subplot(122), plt.imshow(dst,'gray'), plt.title('Perspective')
	plt.show()


	text = pytesseract.image_to_string(dst, lang='kor', config='--psm 7')
	print(text)
