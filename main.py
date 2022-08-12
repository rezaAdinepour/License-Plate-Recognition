import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import imutils


orgImg = cv.imread('Images\image6.jpg') 
img = orgImg.copy()
grayImg = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#plt.imshow(cv.cvtColor(grayImg, cv.COLOR_BGR2RGB))


nrFilter = cv.bilateralFilter(grayImg, 11, 17, 17)
edg = cv.Canny(nrFilter, 50, 300)
#plt.imshow(cv.cvtColor(edg, cv.COLOR_BGR2RGB))

keyContours = cv.findContours(edg.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keyContours)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    if(len(approx) == 4):
        location = approx
        #print(location)
        break


mask = np.zeros(grayImg.shape, np.uint8)
newImg = cv.drawContours(mask, [location], 0, 255, -1)
newImg = cv.bitwise_and(img, img, mask=mask)
#plt.imshow(cv.cvtColor(newImg, cv.COLOR_BGR2RGB))


(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
croppImg = grayImg[x1:x2+1, y1:y2+1]
#plt.imshow(cv.cvtColor(croppImg, cv.COLOR_BGR2RGB))


reader = easyocr.Reader(['en'])
result = reader.readtext(croppImg)
print(result)


text = result[0][-2]
res = cv.putText(img, text, (approx[0][0][0], approx[1][0][1]+60), cv.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2, cv.LINE_AA)
res = cv.rectangle(img, tuple(approx[0][0]), tuple(approx[2][0]), (0,255,0), 2)
plt.figure(2)
plt.imshow(cv.cvtColor(res, cv.COLOR_BGR2RGB))
plt.title('Result')

images = [orgImg, edg, newImg, croppImg]
titles = ['original img', 'filtered img', 'number plate', 'zoomed plate']
for i in range(len(images)):
    plt.figure(1)
    plt.subplot(2, 2, i+1)
    plt.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])


plt.show()