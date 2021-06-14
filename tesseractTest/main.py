import cv2
import os
try:
	from PIL import Image
except ImportError:
	import Image
import pytesseract
import numpy as np
from matplotlib import pyplot as plt

# 설치한 tesseract 프로그램 경로 (64비트)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'

# 이미지 불러오기, Gray 프로세싱
####### 파일명 for testing #########
# 일반 명암차 큰 것 OJTSample2
# 전기차 NumTest2
# 옛날 번호판 NumTest3
# 전기경차 OJTSample3
image = cv2.imread("00ah.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# 원본 이미지 보여주기
#cv2.imshow("Image", image)
#cv2.waitKey(0)

# 가우시안 블러/중앙값 블러/Bilateral필터 : 노이즈 줄이기(Bilateral은 경계선 유지)
#img_blurred = cv2.GaussianBlur(gray, ksize=(5,5), sigmaX=0)
#img_blurred = cv2.medianBlur(img_blurred, 3)

#(src, 픽셀 지름, 컬러 고려 공간, 멀리있는 픽셀까지 고려할지)
img_blurred = cv2.bilateralFilter(gray,20,20,250);


# 스레시 홀드
# 옵션에 대한 설명은 링크 참고 : https://opencv-python.readthedocs.io/en/latest/doc/09.imageThresholding/imageThresholding.html

# 기본 스레시홀드(OTSU를 이용하는건 히스토그램이 2개의 고점을 가질 때 효율적)
ret, th1 = cv2.threshold(img_blurred,127,255,cv2.THRESH_BINARY)
th2 = cv2.threshold(img_blurred, 0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

# Adaptive Threshold : 영역별로 스레시홀드
th3 = cv2.adaptiveThreshold(img_blurred,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
cv2.THRESH_BINARY,15,4)
th4 = cv2.adaptiveThreshold(img_blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
cv2.THRESH_BINARY,19,3)

# 캐니()
th5 = cv2.Canny(gray, 50,150)

titles = ['Original','Basic', 'Basic: Otsu','Adaptive: Mean','Adaptive: Gaussian', 'canny']

images = [image, th1, th2, th3, th4, th5]

for i in range(6):
	plt.subplot(2,3,i+1),plt.imshow(images[i],'gray')
	plt.title(titles[i])
	plt.xticks([]),plt.yticks([])
plt.show()


##########Contours 경계선 찾기
orig_img=image.copy()
contours,_ = cv2.findContours(th3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
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
		'cx': x + (w / 2), #중앙 x, 중앙 y
		'cy': y + (h / 2)
	})

plt.figure(figsize=(12, 8))
plt.imshow(orig_img[:, :, ::-1])
plt.show()



###### Contours 추리기
orig_img = image.copy()
count = 0

for d in contours_dict:
	rect_area = d['w'] * d['h']  # 영역 크기
	aspect_ratio = d['w'] / d['h']

	# 이 부분을 내가 원하는 사이즈로 바꿔야 함...#
	if (aspect_ratio >= 0.3) and (aspect_ratio <= 1.0) and (rect_area >= 800) and (rect_area <= 2000):
		cv2.rectangle(orig_img, (d['x'], d['y']), (d['x'] + d['w'], d['y'] + d['h']), (0, 255, 0), 2)
		d['idx'] = count
		count += 1
		pos_cnt.append(d)

plt.figure(figsize=(20, 20))
plt.imshow(orig_img[:, :, ::-1])
plt.show()

# 글자 프로세싱을 위해 Gray 이미지 임시파일 형태로 저장.
filename = "{}.jpg".format(os.getpid())
cv2.imwrite(filename, th4)


text = pytesseract.image_to_string(Image.open(filename), lang='kor')
os.remove(filename)

print(text)