import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
from model import do_predict
'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++初始化++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
#窗口长宽
window_width = 256
window_height = 256 
#用于存储各自模式下鼠标拖动经过的像素，用于各个二值图的绘制
pixels_2dmask=[]
pixels_shapemask=[]
pixels_clmask=[]
pixels_flinvMaskandflmask=[]
pixels_slmask=[]
#以下三个图像是灰度图，所以提供了四档灰度用于灰度图不同区域的灰度绘制，每张图的每种灰度由一个list存储
pixels_ddm_63=[]
pixels_ddm_127=[]
pixels_ddm_191=[]
pixels_ddm_255=[]
pixels_npr_63=[]
pixels_npr_127=[]
pixels_npr_191=[]
pixels_npr_255=[]
pixels_cur_63=[]
pixels_cur_127=[]
pixels_cur_191=[]
pixels_cur_255=[]
#输入模式：数字1 2 3 4 5 6 7 8 分别代表2dMask,sharpMask,clmask,flinvMaskandflmask,sLMask,ds and dsmask,npr,curvmag正在进行数据输入
input_mode=0
#输入模式：数字1 2 3 4 分别代表ds,npr,curvmag的63,127,191,255灰度正在进行数据输入
dsmode=0
nprmode=0
curmode=0
#填充色(白色)
color = [ 255, 255, 255]
#模板路径
template_pic=".\\template\\template.png"
#绘制点的大小
pointsize=5
#np数组用于存储图像
dmask   =np.zeros((256, 256, 1), dtype=np.float32)    #2dMask
spmask  =np.zeros((256, 256, 1), dtype=np.float32)    #sharpMask
clmask  =np.zeros((256, 256, 1), dtype=np.float32)    #clMask
flim    =np.zeros((256, 256, 1), dtype=np.float32)    #fLinvMask
flmask  =np.zeros((256, 256, 1), dtype=np.float32)    #flMask
slmask  =np.zeros((256, 256, 1), dtype=np.float32)    #slMask
ds      =np.zeros((256, 256, 1), dtype=np.float32)    #ds
dsmask  =np.zeros((256, 256, 1), dtype=np.float32)    #dsMask
npr     =np.zeros((256, 256, 1), dtype=np.float32)    #npr
curmag  =np.zeros((256, 256, 1), dtype=np.float32)    #curvMag


#不同模式下鼠标移动事件结束，所捕获的当前视图
catch_view_bk = np.zeros((256, 256, 1), dtype=np.float32)#背景源数据  
catch_view_cl = np.zeros((256, 256, 1), dtype=np.float32)  #clmask源数据
catch_view_fl = np.zeros((256, 256, 1), dtype=np.float32)   #flim&flmask源数据
catch_view_sl = np.zeros((256, 256, 1), dtype=np.float32) #slmask源数据 
catch_view_np = np.zeros((256, 256, 1), dtype=np.float32)   #npr源数据
catch_view_cu = np.zeros((256, 256, 1), dtype=np.float32)   #curmag源数据     
'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++工具函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
#填充算法：将图片上点包围的区域涂上颜色
#points: 点集合 img: 图片 color: 填充所用颜色 
def point2area(points, img, color):
    a=np.array(points)
    cv2.fillPoly(img, np.int32([a]) ,color)
    return img

#连线算法：将图片上点连接起来
#points: 点集合 img: 图片  isClosed:是否闭合 0 不闭合 1 闭合 color：颜色
def point2line(points,img, isClosed=0,color=255):
    a=np.array(points)
    cv2.polylines(img,  np.int32([a]), isClosed, color)
    return img
#画点 因为用链表存储像素点 再用point2line画线，如果是两条等灰度的ds一定会首末相连，所以尝试用散点代表线 类似于之前display中用点代替线的解决办法
def drawpoints(points,img,color=255):
    for point in points:
        img[point[1],point[0],:]=color
    #img = gray2blackwhite(img)
    return img

#灰度图转换为二值图
def gray2blackwhite(img):
    #二值化
    img1=img
    thresh = 125
    img1[img1 > thresh] = 255  
    img1[img1 <= thresh] = 0 
    return img1

#灰度图转换为二值图 弱化版
def gray2blackwhitelow(img):
    #二值化
    img1=img
    thresh = 10
    img1[img1 > thresh] = 255  
    img1[img1 <= thresh] = 0 
    return img1

#将Opengl窗口显示的图像转换为(window_height,window_width,1)的灰度图
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

# 获取输入模板，允许按照模板(模板图片是白底颜色线，绘制窗口是黑底白线)进行描点绘制，输入进网络的npr线是白底黑线加自己绘制的灰度线，若模板不给出默认为输入进网络的npr是全白背景
def get_template(template):
    if(not os.path.exists(template)):
        print("Not found the picture,please check the path and picture's name:",template)
        return np.zeros((256,256),dtype=np.uint8)
    #print("Get the picture successfully:",template)
    return 255-gray2blackwhite(cv2.imread(template,0))

# 数组参数传送进网络
def read10pics():
    global dmask,spmask,clmask,flim,flmask,slmask,ds,dsmask,npr,curmag
    npr = npr
    npr = npr[np.newaxis, :, :, :]
    npr.astype(np.float32)
    npr = npr / 255
    npr = tf.convert_to_tensor(npr)
    ds = ds
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
    sm = spmask
    sm = sm[np.newaxis, :, :, :]
    sm.astype(np.float32)
    sm = sm / 255
    sm = tf.convert_to_tensor(sm)
    dsm = dsmask
    dsm = dsm[np.newaxis, :, :, :]
    dsm.astype(np.float32)
    dsm = dsm / 255
    dsm = tf.convert_to_tensor(dsm)
    msk = dmask
    msk = msk[np.newaxis, :, :, :]
    msk.astype(np.float32)
    msk = msk / 255
    msk = tf.convert_to_tensor(msk)
    slm = slmask
    slm = slm[np.newaxis, :, :, :]
    slm.astype(np.float32)
    slm = slm / 255
    slm = tf.convert_to_tensor(slm)
    cur = curmag
    cur = cur[np.newaxis, :, :, :]
    cur.astype(np.float32)
    cur = cur / 255
    cur = tf.convert_to_tensor(cur)
    return npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur
'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++显示函数++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''

def display():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glViewport(0, 0, window_width, window_height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluOrtho2D(0, window_width, window_height, 0)
    
    global pixels_2dmask,pixels_shapemask,pixels_clmask,pixels_flinvMaskandflmask,pixels_slmask,template_pic
    global pixels_ddm_63,pixels_ddm_127,pixels_ddm_191,pixels_ddm_255
    global pixels_npr_63,pixels_npr_127,pixels_npr_191,pixels_npr_255
    global pixels_cur_63,pixels_cur_127,pixels_cur_191,pixels_cur_255
    global catch_view_bk,catch_view_cl,catch_view_fl,catch_view_sl,catch_view_np,catch_view_cu

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
    cur_view0=storewin2pic()#没给出模板
    catch_view_bk=cur_view0
    #cv2.imwrite(".\\image\\cur_view0.jpg",cur_view0)
    #cv2.imwrite(".\\image\\catch_view_bk.jpg",catch_view_bk)

    #2dmask
    if(input_mode==1):
        glPointSize(pointsize)
        glColor3f(0.0, 1.0, 0.0)#画笔绿色
        glBegin(GL_POINTS)
        length=len(pixels_2dmask)
        for i in range(length-1):
                glVertex2f(pixels_2dmask[i][0], pixels_2dmask[i][1])
        glEnd()
        
    #shapemask
    elif(input_mode==2):
        # fLInvMask and fLMask 所需数据输入
        glPointSize(pointsize)
        glColor3f(1.0, 1.0, 0.0)#画笔黄色
        glBegin(GL_POINTS)
        length=len(pixels_shapemask)
        for i in range(length-1):
            glVertex2f(pixels_shapemask[i][0], pixels_shapemask[i][1])
        glEnd()
        
    #clmask
    elif(input_mode==3):
        glPointSize(pointsize)
        glColor3f(0.5, 0.5, 0.5)#画笔灰色
        glBegin(GL_POINTS)
        length=len(pixels_clmask)
        for i in range(length-1):
            glVertex2f(pixels_clmask[i][0], pixels_clmask[i][1])
        glEnd()
        #存储数据
        cur_view3=storewin2pic()
        catch_view_cl=cur_view3-catch_view_bk#去除背景
        #所画的是clmask灰度图，转换为二值图
        global clmask
        clmask = 255-gray2blackwhite(catch_view_cl)

    #flinvMask and flmask
    elif(input_mode==4):
        glPointSize(pointsize)
        glColor3f(0.8, 0.5, 0.0) #画笔棕色
        glBegin(GL_POINTS)
        length=len(pixels_flinvMaskandflmask)
        for i in range(length-1):
            glVertex2f(pixels_flinvMaskandflmask[i][0], pixels_flinvMaskandflmask[i][1])
        glEnd()
        #存储数据
        cur_view4=storewin2pic()
        catch_view_fl=cur_view4-catch_view_bk
        global flim,flmask
        flmask = gray2blackwhite(catch_view_fl)
        flim = 255-flmask

        '''
    #slmask
    elif(input_mode==5):
        glPointSize(pointsize)
        glColor3f(0.6, 0.5, 0.6)#画笔粉色
        glBegin(GL_POINTS)
        length=len(pixels_slmask)
        for i in range(length-1):
            glVertex2f(pixels_slmask[i][0], pixels_slmask[i][1])
        glEnd()
        #存储数据
        cur_view5=storewin2pic()
        catch_view_sl=cur_view5-catch_view_bk
        global slmask
        slmask = gray2blackwhite(catch_view_sl)
        ''' 

    #ds and dsmask
    elif(input_mode==6):
        glPointSize(pointsize)
        glBegin(GL_POINTS)
        length63=len(pixels_ddm_63)
        length127=len(pixels_ddm_127)
        length191=len(pixels_ddm_191)
        length255=len(pixels_ddm_255)
        glColor3f(1.0, 1.0, 1.0)#画笔白色
        for i in range(length63-1):
            glVertex2f(pixels_ddm_63[i][0], pixels_ddm_63[i][1])
        glColor3f(1.0, 0.5, 0.7)#画笔红粉
        for i in range(length127-1):
            glVertex2f(pixels_ddm_127[i][0], pixels_ddm_127[i][1])
        glColor3f(0.0, 1.0, 1.0)#画笔天蓝
        for i in range(length191-1):
            glVertex2f(pixels_ddm_191[i][0], pixels_ddm_191[i][1])
        glColor3f(0.7, 0.5, 0.7)#画笔浅紫
        for i in range(length255-1):
            glVertex2f(pixels_ddm_255[i][0], pixels_ddm_255[i][1])
        glEnd()

    #npr
    elif(input_mode==7):
        glPointSize(pointsize)
        glBegin(GL_POINTS)
        length63=len(pixels_npr_63)
        length127=len(pixels_npr_127)
        length191=len(pixels_npr_191)
        length255=len(pixels_npr_255)
        #以下四个颜色灰度化并不是255-63,255-127,255-192,255-255 我只是随便找的四个灰度化后可以区分的颜色 以后需要再改颜色就行
        glColor3f(1.0, 1.0, 1.0)#画笔白色
        for i in range(length63-1):
            glVertex2f(pixels_npr_63[i][0], pixels_npr_63[i][1])
        glColor3f(1.0, 0.5, 0.7)#画笔红粉
        for i in range(length127-1):
            glVertex2f(pixels_npr_127[i][0], pixels_npr_127[i][1])
        glColor3f(0.0, 1.0, 1.0)#画笔天蓝
        for i in range(length191-1):
            glVertex2f(pixels_npr_191[i][0], pixels_npr_191[i][1])
        glColor3f(0.2,0.2,0.2)#画笔浅紫
        for i in range(length255-1):
            glVertex2f(pixels_npr_255[i][0], pixels_npr_255[i][1])
        glEnd()
        #存储数据
        cur_view7=storewin2pic()
        catch_view_np=cur_view7#这里不减去catch_view_bk，因为把模板也作为npr输入
        global npr
        npr = 255-catch_view_np

    #curmag
    elif(input_mode==8):
        glPointSize(pointsize)
        glBegin(GL_POINTS)
        length63=len(pixels_cur_63)
        length127=len(pixels_cur_127)
        length191=len(pixels_cur_191)
        length255=len(pixels_cur_255)
        #以下四个颜色灰度化并不是255-63,255-127,255-192,255-255 我只是随便找的四个灰度化后可以区分的颜色 以后需要再改颜色就行
        glColor3f(0.2,0.2,0.2)#画笔灰色
        for i in range(length63-1):
            glVertex2f(pixels_cur_63[i][0], pixels_cur_63[i][1])
        glColor3f(0.0, 1.0, 1.0)#画笔天蓝
        for i in range(length127-1):
            glVertex2f(pixels_cur_127[i][0], pixels_cur_127[i][1])
        glColor3f(1.0, 0.5, 0.7)#画笔红粉
        for i in range(length191-1):
            glVertex2f(pixels_cur_191[i][0], pixels_cur_191[i][1])
        glColor3f(1.0, 1.0, 1.0)#画笔白色
        for i in range(length255-1):
            glVertex2f(pixels_cur_255[i][0], pixels_cur_255[i][1])
        glEnd()
        #存储数据
        cur_view8=storewin2pic()
        catch_view_cu=cur_view8-catch_view_bk
        global curmag,slmask
        curmag = cur_view8-catch_view_bk
        slmask = gray2blackwhitelow(catch_view_cu)

    #全局变量显示
    cv2.imwrite(".\\image\\2dMask.jpg",dmask)
    cv2.imwrite(".\\image\\shapeMask.jpg",spmask)
    cv2.imwrite(".\\image\\clIMask.jpg",clmask)
    cv2.imwrite(".\\image\\fLInvMask.jpg",flim)
    cv2.imwrite(".\\image\\fLMask.jpg",flmask)
    cv2.imwrite(".\\image\\sLMask.jpg",slmask)
    cv2.imwrite(".\\image\\ds.jpg",ds)
    cv2.imwrite(".\\image\\dsMask.jpg",dsmask)
    cv2.imwrite(".\\image\\npr.jpg",npr)
    cv2.imwrite(".\\image\\curvMag.jpg",curmag)
    
    #双缓存交换缓存以显示图像
    glutSwapBuffers()
    #每次更新显示
    glutPostRedisplay()
'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++鼠标事件++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
def mouse_hit(button,  state,  x,  y):
    global input_mode
    global pixels_2dmask,pixels_shapemask,dmask,spmask
    global ds,dsmask,dsmode,pixels_ddm_63,pixels_ddm_127,pixels_ddm_191,pixels_ddm_255

    #鼠标操作基本结构
    if(button==GLUT_LEFT_BUTTON):
        if (state == GLUT_UP):	#左键抬起时
            if(input_mode==1):
	        #2dMask
                dmask = point2area(pixels_2dmask,dmask,color)
            elif(input_mode==2): 
                #shapemask
                spmask = point2area(pixels_shapemask,spmask,color)
            #ds&dsMask
            elif(input_mode==6):
                if(dsmode==1):
                    ds     = point2line(pixels_ddm_63,ds,0,63)
                    dsmask = point2line(pixels_ddm_63,dsmask,0,255)
                    #ds     = drawpoints(pixels_ddm_63,ds,63)
                    #dsmask = drawpoints(pixels_ddm_63,dsmask,255)
                elif(dsmode==2):
                    ds = point2line(pixels_ddm_127,ds,0,150)
                    dsmask = point2line(pixels_ddm_127,dsmask,0,255)
                    #ds     = drawpoints(pixels_ddm_127,ds,127)
                    #dsmask = drawpoints(pixels_ddm_127,dsmask,255)
                elif(dsmode==3):
                    ds = point2line(pixels_ddm_191,ds,0,191)
                    dsmask = point2line(pixels_ddm_191,dsmask,0,255)
                    #ds     = drawpoints(pixels_ddm_191,ds,191)
                    #dsmask = drawpoints(pixels_ddm_191,dsmask,255)
                elif(dsmode==4):
                    ds = point2line(pixels_ddm_255,ds,0,255)
                    dsmask = point2line(pixels_ddm_255,dsmask,0,255)
                    #ds     = drawpoints(pixels_ddm_255,ds,255)
                    #dsmask = drawpoints(pixels_ddm_255,dsmask,255)
                
def mouse_move( x,  y):
    global pixels_2dmask,pixels_shapemask,pixels_clmask,pixels_flinvMaskandflmask,pixels_slmask
    global pixels_ddm_63,pixels_ddm_127,pixels_ddm_191,pixels_ddm_255
    global pixels_npr_63,pixels_npr_127,pixels_npr_191,pixels_npr_255
    global pixels_cur_63,pixels_cur_127,pixels_cur_191,pixels_cur_255
    tmp= () #存储鼠标拖动经过的xy
    tmp=(x,y)
    if(input_mode==1):
        pixels_2dmask.append(tmp)
    elif(input_mode==2):
        pixels_shapemask.append(tmp)
    elif(input_mode==3):
        pixels_clmask.append(tmp)
    elif(input_mode==4):
        pixels_flinvMaskandflmask.append(tmp)
    elif(input_mode==5):
        pixels_slmask.append(tmp)
    elif(input_mode==6):
        if(dsmode==1):
            pixels_ddm_63.append(tmp)
        elif(dsmode==2):
            pixels_ddm_127.append(tmp)
        elif(dsmode==3):
            pixels_ddm_191.append(tmp)
        elif(dsmode==4):
            pixels_ddm_255.append(tmp)
    elif(input_mode==7):
        if(nprmode==1):
            pixels_npr_63.append(tmp)
        elif(nprmode==2):
            pixels_npr_127.append(tmp)
        elif(nprmode==3):
            pixels_npr_191.append(tmp)
        elif(nprmode==4):
            pixels_npr_255.append(tmp)
    elif(input_mode==8):
        if(curmode==1):
            pixels_cur_63.append(tmp)
        elif(curmode==2):
            pixels_cur_127.append(tmp)
        elif(curmode==3):
            pixels_cur_191.append(tmp)
        elif(curmode==4):
            pixels_cur_255.append(tmp)
'''++++++++++++++++++++++++++++++++++++++++++++++++++++++++++菜单设置++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'''
#ds灰度选项
def dsGrayOpt(data):
    global input_mode,dsmode
    print("Ready to draw ds and dsmask.\n")
    input_mode=6
    if(data==1):
        print("gray=63.\n")
        dsmode=1
        glutMotionFunc(mouse_move)
    elif(data==2):
        print("gray=127.\n")
        dsmode=2
        glutMotionFunc(mouse_move)
    elif(data==3):
        print("gray=191.\n")
        dsmode=3
        glutMotionFunc(mouse_move)
    else:
        print("gray=255.\n")
        dsmode=4
        glutMotionFunc(mouse_move)
    return 0
#npr灰度选项
def nprGrayOpt(data):
    print("Ready to draw npr.\n")
    global input_mode,nprmode
    input_mode=7
    if(data==1):
        print("gray=63.\n")
        nprmode=1
        glutMotionFunc(mouse_move)
    elif(data==2):
        print("gray=127.\n")
        nprmode=2
        glutMotionFunc(mouse_move)
    elif(data==3):
        print("gray=191.\n")
        nprmode=3
        glutMotionFunc(mouse_move)
    else:
        print("gray=255.\n")
        nprmode=4
        glutMotionFunc(mouse_move)
    return 0
#curMag灰度选项
def curGrayOpt(data):
    print("Ready to draw curMag.\n")
    global input_mode,curmode
    input_mode=8
    if(data==1):
        print("gray=63.\n")
        curmode=1
        glutMotionFunc(mouse_move)
    elif(data==2):
        print("gray=127.\n")
        curmode=2
        glutMotionFunc(mouse_move)
    elif(data==3):
        print("gray=191.\n")
        curmode=3
        glutMotionFunc(mouse_move)
    else:
        print("gray=255.\n")
        curmode=4
        glutMotionFunc(mouse_move)
    return 0
#主菜单
def main_menu( data):
    global input_mode
    if(data==1):
        print("Ready to draw 2dmask.\n")
        input_mode=1

    elif(data==2):
        print("Ready to draw shapemask.\n")
        input_mode=2
    elif(data==3):
        print("Ready to draw clmask.\n")
        input_mode=3
        glutMotionFunc(mouse_move)
    elif(data==4):
        print("Ready to draw flinvMask and flmask.\n")
        input_mode=4
        glutMotionFunc(mouse_move)
    elif(data==5):
        print("Ready to draw slmask.\n")
        input_mode=5
        glutMotionFunc(mouse_move)
    else:
        print("OK.Ready to push lines to network.\n")
        npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur=read10pics()
        do_predict(npr, ds, fm, fmi, cm, sm, dsm, msk, slm, cur)
    return 0

if __name__ == "__main__":
    #初始化GL
    glutInit()
    #设置显示参数(双缓存，RGB格式)
    glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE)
    #设置窗口位置：在屏幕左上角像素值(100,100)处
    glutInitWindowPosition(800,400)
    #设置窗口尺寸：width*height
    glutInitWindowSize(window_height,window_width)
    #设置窗口名称
    glutCreateWindow("Interaction Window")
    #显示函数
    glutDisplayFunc(display)
    #鼠标事件
    glutMotionFunc(mouse_move)
    glutMouseFunc(mouse_hit)
    #创建ds灰度子菜单
    dssubmenu = glutCreateMenu(dsGrayOpt)
    glutAddMenuEntry("gray=63",1)
    glutAddMenuEntry("gray=127",2)
    glutAddMenuEntry("gray=191",3)
    glutAddMenuEntry("gray=255",4)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    #创建npr灰度子菜单
    nprsubmenu = glutCreateMenu(nprGrayOpt)
    glutAddMenuEntry("gray=63",1)
    glutAddMenuEntry("gray=127",2)
    glutAddMenuEntry("gray=191",3)
    glutAddMenuEntry("gray=255",4)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    #创建curMag灰度子菜单
    cursubmenu = glutCreateMenu(curGrayOpt)
    glutAddMenuEntry("gray=63",1)
    glutAddMenuEntry("gray=127",2)
    glutAddMenuEntry("gray=191",3)
    glutAddMenuEntry("gray=255",4)
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    #创建主菜单
    menu = glutCreateMenu(main_menu)
    glutAddMenuEntry("2dmask",1)
    glutAddMenuEntry("shapemask",2)
    glutAddMenuEntry("clmask",3)
    glutAddMenuEntry("flinvMask and flmask",4)
    glutAddMenuEntry("slmask",5)
    glutAddMenuEntry("OK!",100)
    #将两个菜单变为另一个菜单的子菜单
    glutAddSubMenu("ds and dsmask",dssubmenu)
    glutAddSubMenu("npr",nprsubmenu)
    glutAddSubMenu("cur",cursubmenu)
    #点击鼠标右键时显示菜单
    glutAttachMenu(GLUT_RIGHT_BUTTON)
    glutMainLoop()
