# OpenCV 常用函数汇总

## 一、色彩空间类型转换

### 1、cv2.cvtColor

`dst = cv2.cvtColor( src, code [, dstCn] )`

式中：

dst 表示输出图像，与原始输入图像具有同样的数据类型和深度。  

src 表示原始输入图像。可以是8位无符号图像、16位无符号图像，或者单精度浮点数等。  

code 是色彩空间转换码，表4-2展示了其枚举值。  

dstCn 是目标图像的通道数。如果参数为默认的 0，则通道数自动通过原始输入图像和 code 得到。  

### 2、HSV 色彩表

![alt text](fe1cfffebcb54caf93a19d6a8509db48.png)

### 3、cv2.inRange

`dst = cv2.inRange( src, lowerb, upperb )`

式中：

 dst 表示输出结果，大小和src一致。

 src 表示要检查的数组或图像。

 lowerb 表示范围下界。

 upperb 表示范围上界。 返回值dst 与src 等大小，其值取决于src中对应位置上的值是否处于区间[lowerb,upperb] 内：

 如果src值处于该指定区间内，则dst中对应位置上的值为255。  如果src值不处于该指定区间内，则dst中对应位置上的值为0。

#### 确定颜色区域

`mask = cv2.inRange(hsv, lower, upper)`

## 二、阈值处理

### 1、cv2.threshold()

`retval, dst = cv2.threshold( src, thresh, maxval, type )`

 retval 代表返回的阈值。  

 dst 代表阈值分割结果图像，与原始图像具有相同的大小和类型。  

 src 代表要进行阈值分割的图像，可以是多通道的，8位或32位浮点型数值。  

 thresh 代表要设定的阈值。  

 maxval 代表当type参数为THRESH_BINARY或者THRESH_BINARY_INV类型时，需要设定的最大值。  

 type 代表阈值分割的类型。  

#### 二值化阈值处理（cv2.THRESH_BINARY）

所有大于127的像素点会被处理为255。 其余值会被处理为0。

`t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY)`

```python
import cv2 
img=cv2.imread("lena.bmp") 
t,rst=cv2.threshold(img,127,255,cv2.THRESH_BINARY) 
cv2.imshow("img",img) 
cv2.imshow("rst",rst) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

### 2、cv2.adaptiveThreshold() 自适应阈值处理

`dst = cv.adaptiveThreshold( src, maxValue, adaptiveMethod, thresholdType, blockSize, C )`

`athdGAUS=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)`

 dst 代表自适应阈值处理结果。

 src 代表要进行处理的原始图像。需要注意的是，该图像必须是8位单通道的图像。

 maxValue 代表最大值。

 adaptiveMethod 代表自适应方法。

 thresholdType 代表阈值处理方式，该值必须是 cv2.THRESH_BINARY 或者cv2.THRESH_BINARY_INV 中的一个。

 blockSize 代表块大小。表示一个像素在计算其阈值时所使用的邻域尺寸，通常为3、5、 7 等。

 C是常量。

函数cv2.adaptiveThreshold()根据参数 adaptiveMethod 来确定自适应阈值的计算方法，函数 包含cv2.ADAPTIVE_THRESH_MEAN_C 和 cv2.ADAPTIVE_THRESH_GAUSSIAN_C 两种不 同的方法。这两种方法都是逐个像素地计算自适应阈值，自适应阈值等于每个像素由参数 blockSize 所指定邻域的加权平均值减去常量C。两种不同的方法在计算邻域的加权平均值时所 采用的方式不同：  cv2.ADAPTIVE_THRESH_MEAN_C：邻域所有像素点的权重值是一致的。  cv2.ADAPTIVE_THRESH_GAUSSIAN_C：与邻域各个像素点到中心点的距离有关，通过高斯方程得到各个点的权重值。

## 三、图像平滑处理

### 1、均值滤波 cv2.blur()

`dst = cv2.blur( src, ksize, anchor, borderType )`

`r=cv2.blur(o,(5,5))`

### 2、方框滤波 cv2.boxFilter()

`dst = cv2.boxFilter( src, ddepth, ksize, anchor, normalize, borderType )`

`r=cv2.boxFilter(o,-1,(5,5))`

### 3、高斯滤波 cv2.GaussianBlur()

`dst = cv2.GaussianBlur( src, ksize, sigmaX, sigmaY, borderType )`

```python
import cv2 
o=cv2.imread("image\\lenaNoise.png") 
r=cv2.GaussianBlur(o,(5,5),0,0) 
cv2.imshow("original",o) 
cv2.imshow("result",r) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

### 4、中值滤波 cv2.medianBlur()

`dst = cv2.medianBlur( src, ksize)`
`r=cv2.medianBlur(o,3)`

### 5、双边滤波 cv2.bilateralFilter()

`dst = cv2.bilateralFilter( src, d, sigmaColor, sigmaSpace, borderType )`

```python
import cv2 
o=cv2.imread("image\\bilTest.bmp") 
b=cv2.bilateralFilter(o,55,100,100) 
cv2.imshow("original",o) 
cv2.imshow("bilateral",b) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

## 四、形态学操作

`dst = cv2.morphologyEx( src, op, kernel[, anchor[, iterations[, borderType[, borderValue]]]] )`

![alt text](image.png)

### 开运算

```python
import cv2 
import numpy as np 
img = cv2.imread("opening.bmp") 
kernal = np.ones((3,3),np.uint8) 
mask = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
cv2.imshow("mask", mask) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

### 闭运算

```python
import cv2 
import numpy as np 
img=cv2.imread("closing.bmp") 
kernal=np.ones((10,10),np.uint8) 
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
cv2.imshow("img",img) 
cv2.imshow("mask",mask) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

## 五、Canny 边缘检测 cv2.Canny()

`edges = cv.Canny( image, threshold1, threshold2[, apertureSize[, L2gradient]])`

```python
import cv2 
o=cv2.imread("lena.bmp",cv2.IMREAD_GRAYSCALE) 
r1=cv2.Canny(o,128,200) 
r2=cv2.Canny(o,32,128) 
cv2.imshow("original",o) 
cv2.imshow("result1",r1) 
cv2.imshow("result2",r2) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

## 六、图像轮廓

### 1、cv2.findContours()

`image, contours, hierarchy = cv2.findContours( image, mode, method)`

`image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)`

### 2、cv2.drawContours()

```python
image=cv2.drawContours( 
image,  
contours,  
contourIdx,  
color[,  
thickness[,  
lineType[,  
hierarchy[,  
maxLevel[,  
offset]]]]] )
```

`mask=cv2.drawContours(mask,contours,-1,(255,255,255),-1)`

### 3、矩特征

#### （1）矩的计算：moments 函数

`retval = cv2.moments( array[, binaryImage] )`

式中有两个参数：

 array：可以是点集，也可以是灰度图像或者二值图像。当 array 是点集时，函数会把这
些点集当成轮廓中的顶点，把整个点集作为一条轮廓，而不是把它们当成独立的点来看
待。

 binaryImage：该参数为 True 时，array 内所有的非零值都被处理为1。该参数仅在参数
array 为图像时有效。

该函数的返回值 retval 是矩特征，主要包括：

（1）空间矩

 零阶矩：m00

 一阶矩：m10, m01

 二阶矩：m20, m11, m02

 三阶矩：m30, m21, m12, m03

（2）中心矩

 二阶中心矩：mu20, mu11, mu02

 三阶中心矩：mu30, mu21, mu12, mu03

（3）归一化中心矩

 二阶Hu矩：nu20, nu11, nu02

 三阶Hu矩：nu30, nu21, nu12, nu03

上述矩都是根据公式计算得到的，大多数矩比较抽象。但是很明显，如果两个轮廓的矩一
致，那么这两个轮廓就是一致的。虽然大多数矩都是通过数学公式计算得到的抽象特征，但是
零阶矩“m00”的含义比较直观，它表示一个轮廓的面积。

在 OpenCV 中，函数 cv2.moments() 会同时计算上述空间矩、中心矩和归一化中心距。

```python
import cv2 
import numpy as np 
o = cv2.imread('moments.bmp')   
cv2.imshow("original",o) 
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)   
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)   
image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
n=len(contours) 
contoursImg=[] 
for i in range(n): 
    temp=np.zeros(image.shape,np.uint8) 
    contoursImg.append(temp) 
    contoursImg[i]=cv2.drawContours(contoursImg[i],contours,i,255,3)  
    cv2.imshow("contours[" + str(i)+"]",contoursImg[i])  
print("观察各个轮廓的矩（moments）:") 
for i in range(n): 
    print("轮廓"+str(i)+"的矩:\n",cv2.moments(contours[i])) 
print("观察各个轮廓的面积:") 
for i in range(n): 
    print("轮廓"+str(i)+"的面积:%d" %cv2.moments(contours[i])['m00']) 
cv2.waitKey() 
cv2.destroyAllWindows()
```

```python
M = cv2.moments(largest_contour)

if M['m00'] != 0:
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
```

#### （2）计算轮廓的面积：contourArea 函数

`retval =cv2.contourArea(contour [, oriented] )`

```python
import cv2 
import numpy as np 
o = cv2.imread('contours.bmp')   
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)   
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)   
image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
cv2.imshow("original",o) 
n=len(contours) 
contoursImg=[] 
for i in range(n): 
    print("contours["+str(i)+"]面积=",cv2.contourArea(contours[i])) 
    temp=np.zeros(o.shape,np.uint8) 
    contoursImg.append(temp) 
    contoursImg[i]=cv2.drawContours(contoursImg[i], contours, i, (255,255,255), 3) 
    cv2.imshow("contours[" + str(i)+"]",contoursImg[i])    
cv2.waitKey() 
cv2.destroyAllWindows()
```

#### （3）计算轮廓的长度：arcLength 函数

`retval = cv2.arcLength( curve, closed )`

上式中有两个参数：

 curve 是轮廓。

 closed 是布尔型值，用来表示轮廓是否是封闭的。该值为True时，表示轮廓是封闭的。

```python
import cv2 
import numpy as np 
#--------------读取及显示原始图像-------------------- 
o = cv2.imread('contours0.bmp')   
cv2.imshow("original",o) 
#--------------获取轮廓-------------------- 
gray = cv2.cvtColor(o,cv2.COLOR_BGR2GRAY)   
ret, binary = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)   
image,contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)   
#--------------计算各轮廓的长度之和、平均长度-------------------- 
n=len(contours)   # 获取轮廓的个数 
cntLen=[]          # 存储各轮廓的长度 
for i in range(n): 
    cntLen.append(cv2.arcLength(contours[i],True)) 
    print("第"+str(i)+"个轮廓的长度:%d"%cntLen[i]) 
cntLenSum=np.sum(cntLen)  # 各轮廓的长度之和 
cntLenAvr=cntLenSum/n      # 轮廓长度的平均值 
print("轮廓的总长度为：%d"%cntLenSum) 
print("轮廓的平均长度为：%d"%cntLenAvr)
#--------------显示长度超过平均值的轮廓-------------------- 
contoursImg=[] 
for i in range(n): 
    temp=np.zeros(o.shape,np.uint8) 
    contoursImg.append(temp) 
    contoursImg[i]=cv2.drawContours(contoursImg[i], contours, i, (255,255,255), 3) 
    if cv2.arcLength(contours[i],True)>cntLenAvr: 
        cv2.imshow("contours[" + str(i)+"]",contoursImg[i])     
cv2.waitKey() 
cv2.destroyAllWindows()
```

## 七、角点检测

### Shi-Tomasi 角点检测

```python
cv2.goodFeaturesToTrack(img, 
                        corners,
                        maxCorners,
                        qualityLevel,
                        minDistance,
                        mask,
                        blockSize,
                        gradientSize,
                        useHarrisDetector=False, 
                        k=0.04
                       )
```

```python
import cv2
import numpy as np
 
# Shi-Tomasi角点检测部分
# 图像预处理
img = cv2.imread('Example4.png', flags=cv2.IMREAD_GRAYSCALE)
img_bgr = cv2.imread('Example.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)
img_bgr = cv2.resize(img_bgr, None, fx=0.5, fy=0.5)
kernel = np.ones((3,3), dtype=np.uint8)
img_open = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=3)
img_close = cv2.morphologyEx(img_open, cv2.MORPH_CLOSE, kernel, iterations=10)
img_erode = cv2.morphologyEx(img_close, cv2.MORPH_ERODE, kernel, iterations=4)
img_blur = cv2.medianBlur(img_erode, 39)
cv2.imshow('blur', img_blur)
#  shi-tomasi角点检测
corners = cv2.goodFeaturesToTrack(img_blur, 4, 0.01, 300)
print(corners)
#  显示角点
for i in corners:
    x,y = i.ravel()
    cv2.circle(img_bgr,(int(x),int(y)),5,(0,0,255),-1)
cv2.imshow('dst', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 八、霍夫变换

### 1、霍夫直线变换 cv2.HoughLines()

`lines=cv2.HoughLines(image, rho, theta, threshold)`

式中：

 image 是输入图像，即源图像，必须是8位的单通道二值图像。如果是其他类型的图像，
在进行霍夫变换之前，需要将其修改为指定格式。

 rho 为以像素为单位的距离r的精度。一般情况下，使用的精度是1。

 theta 为角度𝜃的精度。一般情况下，使用的精度是π/180，表示要搜索所有可能的角度。

 threshold 是阈值。该值越小，判定出的直线就越多。通过上一节的分析可知，识别直线
时，要判定有多少个点位于该直线上。在判定直线是否存在时，对直线所穿过的点的数
量进行评估，如果直线所穿过的点的数量小于阈值，则认为这些点恰好（偶然）在算法
上构成直线，但是在源图像中该直线并不存在；如果大于阈值，则认为直线存在。所以，
如果阈值较小，就会得到较多的直线；阈值较大，就会得到较少的直线。

 返回值 lines 中的每个元素都是一对浮点数，表示检测到的直线的参数，即(r, θ)，是
numpy.ndarray 类型。

有一点需要强调的是，使用函数 cv2.HoughLines()检测到的是图像中的直线而不是线段，
因此检测到的直线是没有端点的。所以，我们在进行霍夫直线变换时所绘制的直线都是穿过整
幅图像的。

#### 概率霍夫变换 cv2.HoughLinesP()

`lines =cv2.HoughLinesP(image, rho, theta, threshold, minLineLength, maxLineGap)`

式中参数与返回值的含义如下：

 image 是输入图像，即源图像，必须为 8 位的单通道二值图像。对于其他类型的图像，
在进行霍夫变换之前，需要将其修改为这个指定的格式。

 rho为以像素为单位的距离r的精度。一般情况下，使用的精度是1。

 theta 是角度𝜃的精度。一般情况下，使用的精度是np.pi/180，表示要搜索可能的角度。

 threshold 是阈值。该值越小，判定出的直线越多；值越大，判定出的直线就越少。

 minLineLength 用来控制“接受直线的最小长度”的值，默认值为0。

 maxLineGap 用来控制接受共线线段之间的最小间隔，即在一条线中两点的最大间隔。
如果两点间的间隔超过了参数maxLineGap的值，就认为这两点不在一条线上。默认值
为0。

 返回值lines是由numpy.ndarray类型的元素构成的，其中每个元素都是一对浮点数，表
示检测到的直线的参数，即(r, θ)。

与函数 cv2.HoughLines() 不同的是，函数 cv2.HoughLinesP() 返回的是直线的端点坐标，
而不是直线的参数。

```python
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
img = cv2.imread('computer.jpg',-1) 
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
edges = cv2.Canny(gray,50,150,apertureSize =3) 
orgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB) 
oShow=orgb.copy() 
lines = cv2.HoughLinesP(edges,1,np.pi/180,1,minLineLength=100,maxLineGap=10) 
for line in lines: 
x1,y1,x2,y2 = line[0] 
cv2.line(orgb,(x1,y1),(x2,y2),(255,0,0),5) 
plt.subplot(121) 
plt.imshow(oShow) 
plt.axis('off') 
plt.subplot(122) 
plt.imshow(orgb) 
plt.axis('off') 
```

### 2、霍夫圆环变换 cv2.HoughCircles()

```python
circles=cv2.HoughCircles(image,  
method,  
dp,  
minDist,  
param1,  
param2,  
minRadius,  
maxRadius) 
```

式中参数与返回值的含义如下：

 image：输入图像，即源图像，类型为8位的单通道灰度图像。

 method：检测方法。截止到OpenCV 4.0.0-pre版本，HOUGH_GRADIENT是唯一可用
的参数值。该参数代表的是霍夫圆检测中两轮检测所使用的方法。

 dp：累计器分辨率，它是一个分割比率，用来指定图像分辨率与圆心累加器分辨率的比
例。例如，如果dp=1，则输入图像和累加器具有相同的分辨率。

 minDist：圆心间的最小间距。该值被作为阈值使用，如果存在圆心间距离小于该值的
多个圆，则仅有一个会被检测出来。因此，如果该值太小，则会有多个临近的圆被检测出来；如果该值太大，则可能会在检测时漏掉一些圆。

 param1：该参数是缺省的，在缺省时默认值为100。它对应的是Canny边缘检测器的高
阈值（低阈值是高阈值的二分之一）。

 param2：圆心位置必须收到的投票数。只有在第1轮筛选过程中，投票数超过该值的圆，
才有资格进入第2轮的筛选。因此，该值越大，检测到的圆越少；该值越小，检测到的
圆越多。这个参数是缺省的，在缺省时具有默认值100。

 minRadius：圆半径的最小值，小于该值的圆不会被检测出来。该参数是缺省的，在缺
省时具有默认值0，此时该参数不起作用。

 maxRadius：圆半径的最大值，大于该值的圆不会被检测出来。该参数是缺省的，在缺
省时具有默认值0，此时该参数不起作用。

 circles：返回值，由圆心坐标和半径构成的numpy.ndarray。

需要特别注意，在调用函数 cv2.HoughLinesCircles()之前，要对源图像进行平滑操作，以
减少图像中的噪声，避免发生误判。

```python
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
img = cv2.imread('chess.jpg',0) 
imgo=cv2.imread('chess.jpg',-1) 
o=cv2.cvtColor(imgo,cv2.COLOR_BGR2RGB) 
oshow=o.copy() 
img = cv2.medianBlur(img,5) 
circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,300, 
param1=50,param2=30,minRadius=100,maxRadius=200) 
circles = np.uint16(np.around(circles)) 
for i in circles[0,:]: 
    cv2.circle(o,(i[0],i[1]),i[2],(255,0,0),12) 
    cv2.circle(o,(i[0],i[1]),2,(255,0,0),12) 
plt.subplot(121) 
plt.imshow(oshow) 
plt.axis('off') 
plt.subplot(122) 
plt.imshow(o) 
plt.axis('off') 
```

## 九、视频处理

### VideoCapture 类

```python
cap = cv2.VideoCapture(0) 
import numpy as np 
import cv2 
cap = cv2.VideoCapture(0) 
while(cap.isOpened()): 
    ret, frame = cap.read() 
    cv2.imshow('frame',frame) 
    c = cv2.waitKey(1) 
    if c==27:   #ESC 键 
        break 
cap.release()
cv2.destroyAllWindows() 
```

#### 播放视频文件

```python
import numpy as np 
import cv2 
cap = cv2.VideoCapture('viptrain.avi') 
while(cap.isOpened()): 
    ret, frame = cap.read() 
    cv2.imshow('frame',frame) 
    c = cv2.waitKey(25) 
    if c==27:   #ESC 键 
        break 
cap.release() 
cv2.destroyAllWindows()
```

## 以上内容摘自：OpenCV轻松入门：面向 Python by 李立宗
