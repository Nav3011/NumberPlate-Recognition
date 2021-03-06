import cv2
import matplotlib.pyplot as plt


horizontal = list()
vertical = list()
segmentsV = list()
segmentsH = list()
final_v_segments = list()

def calculate_histogram(img):
	
	for i in range(img.shape[0]):
		hist = 0
		for j in range(img.shape[1]):
			if img[i][j]==0:
				hist = hist + 1
		horizontal.append(hist)
	# print(horizontal)
	for i in range(img.shape[1]):
		hist = 0 
		for j in range(img.shape[0]):
			if img[j][i]==0:
				hist = hist + 1
		vertical.append(hist)
	# print(vertical)

	# plt.plot(range(img.shape[0]), horizontal, 'r')
	# plt.plot(range(img.shape[1]), vertical, 'b')
	# plt.show()


def segmentation():
	finalV_start = list()
	finalV_end = list()
	finalH_start = list()
	finalH_end = list()

	vertical_thresh = 3
	horizontal_thresh = 5

	for i in range(len(vertical)):
		if vertical[i] < vertical_thresh:
			segmentsV.append(i)
	for i in range(1,len(segmentsV)):
		if segmentsV[i] - segmentsV[i-1] > 1:
			finalV_start.append(segmentsV[i-1])
			finalV_end.append(segmentsV[i])
	finalV_start[0] = 0
	finalV_end[len(finalV_end)-1]=roi.shape[1]
	final_v_segments.append(finalV_start[0])
	for i in range(1,len(finalV_start)):
		final_v_segments.append(round((finalV_start[i]+finalV_end[i-1])/2))
	final_v_segments.append(finalV_end[len(finalV_end)-1])
	print(final_v_segments)


	for i in range(len(horizontal)):
		if horizontal[i] < horizontal_thresh:
			segmentsH.append(i)
	for i in range(1,len(segmentsH)):
		if segmentsH[i] - segmentsH[i-1] > 1:
			finalH_start.append(segmentsH[i-1])
			finalH_end.append(segmentsH[i])

	# return finalV_start, finalV_end, finalH_start, finalH_end
	return final_v_segments

refPt = []
cropping  = False

def click_and_crop(event, x, y, flags, param):
	global refPt, cropping
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True
	elif event == cv2.EVENT_LBUTTONUP:
		refPt.append((x, y))
		cropping = False
		cv2.rectangle(numberPlate, refPt[0], refPt[1], (0, 255, 0), 2)
		cv2.imshow("detector", numberPlate)

numberPlate = cv2.imread('NP_img.jpg',1)		
clone = numberPlate.copy()
cv2.namedWindow('detector')
cv2.setMouseCallback("detector", click_and_crop)
while True:
	cv2.imshow('detector', numberPlate)
	key = cv2.waitKey(1) & 0xFF
	if key == ord("r"):
		numberPlate = clone.copy()
	elif key == ord("c"):
		break

if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
	ret, thresh1 = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
	cv2.imshow("Thresh", thresh1)
	cv2.imwrite("thresh_plate.jpg", thresh1)
	calculate_histogram(thresh1)
	seg = segmentation()
	# for i in range(len(V_lines_start)):
	# 	image = cv2.line(roi, (V_lines_start[i],0), (V_lines_start[i], roi.shape[0]), (0,255,0), 2)
	# for i in range(len(V_lines_end)):
	# 	image = cv2.line(roi, (V_lines_end[i],0), (V_lines_end[i], roi.shape[0]), (0,0,255), 2)
	for i in range(len(seg)):
		image = cv2.line(roi, (seg[i],0), (seg[i], roi.shape[0]), (0,255,0), 2)
	# for i in range(len(H_lines_start)):
	# 	image = cv2.line(roi, (0,H_lines_start[i]), (roi.shape[1], H_lines_start[i]), (0,255,0), 2)
	# for i in range(len(H_lines_end)):
	# 	image = cv2.line(roi, (0,H_lines_end[i]), (roi.shape[1], H_lines_end[i]), (0,0,255), 2)
	# for i in range(len(H_lines)):
	# 	image = cv2.line(roi, (0,H_lines[i]), (roi.shape[1], H_lines[i]), (0,255,0), 1)

	cv2.imshow("final", image)
	cv2.waitKey(0)
cv2.destroyAllWindows()