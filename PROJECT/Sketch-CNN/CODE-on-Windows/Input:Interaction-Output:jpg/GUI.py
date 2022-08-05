#可以在同一窗口画所有线了
from turtle import shape
import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from model import do_predict
 
#窗口长宽
window_width = 256
window_height = 256 

#用于存储mouse_move_one鼠标拖动经过的像素,用于2dMask,sharpMask,clmask,ds and dsMask五个图像的绘制
pixels_one=[]
#用于存储mouse_move_two鼠标拖动经过的像素,用于fLInvMask and fLMask二个图像的绘制
pixels_two=[]
#用于存储mouse_move_three鼠标拖动经过的像素,用于sLMask一个图像的绘制
pixels_three=[]
#用于存储mouse_move_four鼠标拖动经过的像素,用于npr一个图像的绘制
pixels_four=[]
#用于存储mouse_move_five鼠标拖动经过的像素,用于curvMag一个图像的绘制
pixels_five=[]

# 输入模式 
# 1代表2dMask,sharpMask,clmask,ds and dsMask 所需数据输入
# 2代表fLInvMask and fLMask 所需数据输入
# 3代表sLMask 所需数据输入
# 4代表npr 所需数据输入
# 5代表cur_curvMag所需数据输入
input_mode=0

#填充色(黑色)
color = [0, 0, 0]
#模板路径
template_pic=".\\template\\template.png"
#绘制点的大小
pointsize=5

#np数组用于存储图像
mask   = np.zeros((256, 256, 1), dtype=np.uint8)    #2dMask&sharpMask
clmask = np.zeros((256, 256, 1), dtype=np.uint8)    #clMask
dsk    = np.zeros((256, 256, 1), dtype=np.uint8)    #ds&dsMask
flim   = np.zeros((256, 256, 1), dtype=np.uint8)    #fLinvMask
flmask = np.zeros((256, 256, 1), dtype=np.uint8)    #flMask
slmask = np.zeros((256, 256, 1), dtype=np.uint8)    #slMask
npr    = np.zeros((256, 256, 1), dtype=np.uint8)    #npr
curvmag= np.zeros((256, 256, 1), dtype=np.uint8)    #curvM

#每次鼠标移动事件结束，都会捕获当前视图，将当前视图减去其他鼠标移动事件视图就是当前鼠标移动事件所留下的轨迹 这是为了方便画图 使得即使在一张图上画也能分开不同的笔迹
catch_view0 = np.zeros((256, 256, 1), dtype=np.uint8)
catch_view1 = np.zeros((256, 256, 1), dtype=np.uint8)    
catch_view2 = np.zeros((256, 256, 1), dtype=np.uint8)    
catch_view3 = np.zeros((256, 256, 1), dtype=np.uint8)    
catch_view4 = np.zeros((256, 256, 1), dtype=np.uint8)    
catch_view5 = np.zeros((256, 256, 1), dtype=np.uint8)    

#填充算法：points: 点集合 img: 图片 color: 填充所用颜色 
#return:将图片上点包围的区域涂上颜色
def point2area(points, img, color):
    a=np.array(points)
    res = cv2.fillPoly(img, np.int32([a]) ,color)
    return res

#灰度图转换为二值图
def gray2blackwhite(img):
    #二值化
    thresh = 125
    img[img > thresh] = 255  
    img[img <= thresh] = 0 
    return img

#将Opengl窗口显示的图像转换为(256,256,1)的灰度图
def storewin2pic():
    buffer = ( GLubyte * (3*window_height*window_width) )(0)
    glReadPixels(0, 0, window_height, window_width, GL_RGB, GL_UNSIGNED_BYTE, buffer)
    image = Image.frombytes(mode="RGB", size=(window_height, window_width), data=buffer)
    img1 = list(image.getdata())
    obj = []
    for t in img1:
        obj.append([t[0]*0.299 + t[1]*0.587 + t[2]*0.114]) #灰度化
    img1 = np.array(obj).reshape(window_height, window_width, 1)
    img1 = np.flip(img1, 0)                 # img1.shape = (256, 256, 1)
    return img1

# 获取输入模板，允许按照模板(模板图片是白底颜色线，绘制窗口是黑底白线)进行描点绘制，若模板不给出默认为全黑背景
def get_template(template):
    if(not os.path.exists(template)):
        print("Not found the picture,please check the path and picture's name:",template)
        return np.zeros((256,256),dtype=np.uint8)
    #print("Get the picture successfully:",template)
    return 255-cv2.imread(template,0)

#数组参数传送进网络
def read10pics():
    global mask,clmask,dsk,flim,flmask,slmask,npr,curvmag

    npr = npr
    npr = npr[np.newaxis, :, :, :]
    npr.astype(np.float32)
    npr = npr / 255
    npr = tf.convert_to_tensor(npr)

    ds = dsk
    ds = ds[np.newaxis, :, :, :]
    ds.astype(np.float32)
    ds = ds / 255
    ds = tf.convert_to_tensor(ds)

    fm = flmask
    fm = fm[np.newaxis, :, :, :]
    fm.astype(np.float32)
    fm = fm / 255
    fm = tf.convert_to_tensor(fm)

    fmi = flim
    fmi = fmi[np.newaxis, :, :, :]
    fmi.astype(np.float32)
    fmi = fmi / 255
    fmi = tf.convert_to_tensor(fmi)

    cm = clmask
    cm = cm[np.newaxis, :, :, :]
    cm.astype(np.float32)
    cm = cm / 255
    cm = tf.convert_to_tensor(cm)

    sm = mask
    sm = sm[np.newaxis, :, :, :]
    sm.astype(np.float32)
    sm = sm / 255
    sm = tf.convert_to_tensor(sm)

    dsm = dsk
    dsm = dsm[np.newaxis, :, :, :]
    dsm.astype(np.float32)
    dsm = dsm / 255
    dsm = tf.convert_to_tensor(dsm)

    msk = mask
    msk = msk[np.newaxis, :, :, :]
    msk.astype(np.float32)
    msk = msk / 255
    msk = tf.convert_to_tensor(msk)

    slm = slmask
    slm = slm[np.newaxis, :, :, :]
    slm.astype(np.float32)
    slm = slm / 255
    slm = tf.convert_to_tensor(slm)

    cur = curvmag
    cur = cur[np.newaxis, :, :, :]
    cur.astype(np.float32)
    cur = cur / 255
    cur = tf.convert_to_tensor(cur)

    return npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur

#显示事件
def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, window_width, window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_width, window_height, 0)
    global pixels_one,pixels_two,pixels_three,pixels_four,pixels_five,template_pic
    global catch_view0,catch_view1,catch_view2,catch_view3,catch_view4,catch_view5

    #获取背景
    tmplt=get_template(template_pic)
    nonzero_clm,nonzero_row=np.nonzero(tmplt)#返回值为两个一维np数组 分别代表不为0元素的列行坐标索引号
    glPointSize(0.1)
    glColor3f(1.0, 1.0, 1.0)#白色背景线
    glBegin(GL_POINTS)
    length=len(nonzero_row)
    for i in range(length-1):
        glVertex2f(nonzero_row[i], nonzero_clm[i])
    glEnd()
    #存储数据
    cur_view0=storewin2pic()
    catch_view0=cur_view0
    #cv2.imwrite(".\\image\\cur_view0.jpg",cur_view0)
    #cv2.imwrite(".\\image\\catch_view0.jpg",catch_view0)

    # 2dMask,sharpMask,clmask,ds and dsMask 所需数据输入
    glPointSize(pointsize)
    glColor3f(0.0, 1.0, 0.0)#画笔绿色
    glBegin(GL_POINTS)
    length=len(pixels_one)
    for i in range(length-1):
        glVertex2f(pixels_one[i][0], pixels_one[i][1])
    glEnd()
    #存储数据
    cur_view1=storewin2pic()
    catch_view1=cur_view1-catch_view0
    #cv2.imwrite(".\\image\\cur_view1.jpg",cur_view1)
    #cv2.imwrite(".\\image\\catch_view1.jpg",catch_view1)
    #所画的是clmask灰度图，转换为二值图
    global clmask
    clmask = 255-catch_view1       #颜色翻转为白底黑线
    clmask = gray2blackwhite(clmask)
    #cv2.imwrite(".\\image\\clMask.jpg",clmask)

    # fLInvMask and fLMask 所需数据输入
    glPointSize(pointsize)
    glColor3f(1.0, 1.0, 0.0)#画笔黄色
    glBegin(GL_POINTS)
    length=len(pixels_two)
    for i in range(length-1):
        glVertex2f(pixels_two[i][0], pixels_two[i][1])
    glEnd()
    #存储数据
    cur_view2=storewin2pic()
    #cv2.imwrite(".\\image\\cur_view2.jpg",cur_view2)
    catch_view2=cur_view2-catch_view0-catch_view1
    #cv2.imwrite(".\\image\\catch_view2.jpg",catch_view2)
    #所画的是flmask灰度图，转换为二值图
    global flim
    global flmask
    flmask = gray2blackwhite(catch_view2)
    #cv2.imwrite(".\\image\\fLMask.jpg",flmask)
    flim = 255-flmask
    #cv2.imwrite(".\\image\\fLInvMask.jpg",flim)

    # sLMask所需数据输入
    glPointSize(pointsize)
    glColor3f(0.5, 0.5, 0.5)#画笔灰色
    glBegin(GL_POINTS)
    length=len(pixels_three)
    for i in range(length-1):
        glVertex2f(pixels_three[i][0], pixels_three[i][1])
    glEnd()
    #存储数据
    cur_view3=storewin2pic()
    #cv2.imwrite(".\\image\\cur_view3.jpg",cur_view3)
    catch_view3=cur_view3-catch_view0-catch_view1-catch_view2
    #cv2.imwrite(".\\image\\catch_view3.jpg",catch_view3)
    #所画的是slMask灰度图，转换为二值图
    global slmask
    slmask = gray2blackwhite(catch_view3)
    #cv2.imwrite(".\\image\\sLMask.jpg",slmask)

    # npr所需数据输入  (需要不同粗细颜色的画笔)
    glPointSize(pointsize)
    glColor3f(1.0, 1.0, 1.0)#画笔白色 和模板一个颜色
    glBegin(GL_POINTS)
    length=len(pixels_four)
    for i in range(length-1):
        glVertex2f(pixels_four[i][0], pixels_four[i][1])
    glEnd()
    #存储数据
    cur_view4=storewin2pic()
    #cv2.imwrite(".\\image\\cur_view4.jpg",cur_view4)
    catch_view4=cur_view4-catch_view1-catch_view2-catch_view3#这个没有减去catch_view0 是因为我想把模板也作为npr曲线输入
    #cv2.imwrite(".\\image\\catch_view4.jpg",catch_view4)
    #所画的是npr灰度图
    global npr
    #npr = gray2blackwhite(img_gray)
    npr = 255-catch_view4
    #cv2.imwrite(".\\image\\npr.jpg",npr)

    # curvMag所需数据输入
    glPointSize(pointsize)
    glColor3f(0.0, 1.0, 1.0)#画笔浅蓝色
    glBegin(GL_POINTS)
    length=len(pixels_five)
    for i in range(length-1):
        glVertex2f(pixels_five[i][0], pixels_five[i][1])
    glEnd()
    #存储数据
    cur_view5=storewin2pic()
    #cv2.imwrite(".\\image\\cur_view5.jpg",cur_view5)
    catch_view5=cur_view5-catch_view0-catch_view1-catch_view2-catch_view3-catch_view4
    #cv2.imwrite(".\\image\\catch_view5.jpg",catch_view5)
    #所画的是curvMag灰度图
    global curvmag
    curvmag = catch_view5
    #cv2.imwrite(".\\image\\curvMag.jpg",curvmag)

    #双缓存交换缓存以显示图像
    glutSwapBuffers()
    #每次更新显示
    glutPostRedisplay()

def mouse_hit_one(button,  state,  x,  y):
    global pixels_one
    global mask
    global dsk
    #鼠标操作基本结构
    if(button==GLUT_LEFT_BUTTON):
        if (state == GLUT_UP):	#左键抬起时
	    #2dMask&sharpMask
            mask = point2area(pixels_one,clmask,color)
            mask = 255-mask#颜色翻转
            mask = gray2blackwhite(mask)
            #cv2.imwrite(".\\image\\2dMask&sharpMask.jpg",mask)

	    #ds&dsMask
            for pixel in pixels_one:
                dsk[pixel[1],pixel[0],:]=255
            dsk = gray2blackwhite(dsk)
            #cv2.imwrite(".\\image\\ds&dsMask.jpg",dsk)

def mouse_move_one( x,  y):
    global pixels_one
    tmp= () #存储鼠标拖动经过的xy
    tmp=(x,y)
    pixels_one.append(tmp)
    #print("pixels_one:",pixels_one)

def mouse_move_two( x,  y):
    global pixels_two
    tmp= ()
    tmp=(x,y)
    pixels_two.append(tmp)
    #print("pixels_two:",pixels_two)

def mouse_move_three( x,  y):
    global pixels_three
    tmp= () 
    tmp=(x,y)
    pixels_three.append(tmp)
    #print("pixels_three:",pixels_three)

def mouse_move_four( x,  y):
    global pixels_four
    tmp= () 
    tmp=(x,y)
    pixels_four.append(tmp)
    #print("pixels_four:",pixels_four)

def mouse_move_five( x,  y):
    global pixels_five
    tmp= () 
    tmp=(x,y)
    pixels_five.append(tmp)
    #print("pixels_five:",pixels_five)

#主菜单
def menufunc( data):
    global input_mode
    if(data==1):
        print("Ready to draw 2dMask,sharpMask,clmask,ds and dsMask.\n")
        input_mode=1
        glutMotionFunc(mouse_move_one)
        glutMouseFunc(mouse_hit_one)
    elif(data==2):
        print("Ready to draw fLInvMask and fLMask.\n")
        input_mode=2
        glutMotionFunc(mouse_move_two)
    elif(data==3):
        print("Ready to draw sLMask.\n")
        input_mode=3
        glutMotionFunc(mouse_move_three)
    elif(data==4):
        print("Ready to draw npr.\n")
        input_mode=4
        glutMotionFunc(mouse_move_four)
    elif(data==5):
        print("Ready to draw curvMag.\n")
        input_mode=5
        glutMotionFunc(mouse_move_five)
    else:
        print("OK.Ready to push lines to network\n")
        npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur=read10pics()
        do_predict(npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur)
    return 0

if __name__ == "__main__":
    #初始化GL
    glutInit()
    #设置显示参数(双缓存，RGB格式)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
    #设置窗口位置：在屏幕左上角像素值(100,100)处
    glutInitWindowPosition(0,0)
    #设置窗口尺寸：width*height
    glutInitWindowSize(window_height,window_width)
    #设置窗口名称
    glutCreateWindow("Interaction Window")
    #显示函数
    glutDisplayFunc(display)

    #构建主菜单的内容
    menu = glutCreateMenu(menufunc)
    glutAddMenuEntry("2dMask,sharpMask,clmask,ds and dsMask",1)
    glutAddMenuEntry("fLInvMask and fLMask",2)
    glutAddMenuEntry("sLMask",3)
    glutAddMenuEntry("npr",4)
    glutAddMenuEntry("curvMag",5)
    glutAddMenuEntry("OK!",6)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    
    #重复循环GLUT事件
    glutMainLoop()
