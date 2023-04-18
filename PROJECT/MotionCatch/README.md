# kinect M13运行配置
client ->Release Win32

model_bind ->Release Win32

sever ->Release x64

record_player->四种搭配均可 推荐Release x64 和sever保持一致

##### Sever运行自身的.b文件方式：(打开顺序不固定，先ctrl+f5打开哪个都可以。调试选项固定为Release x64)首先更改../data../Net.conf中的IP地址。设置sever为启动项，ctrl+f5；再设置record_player为启动项，ctrl+f5，选择.b文件。


# 配置的时候 属性最好选择所有配置 所有平台！！！
# WindowsSDKS的kinect是之前预先安装好的 

# 问题一：
在vs2017中重新编译项目Kclient时显示：无法找到 v120 的生成工具(平台工具集 =“v120”)。若要使用 v120 生成工具进行生成，请安装 v120 生成工具。或者，可以升级到当前 Visual Studio 工具，方式是通过选择“项目”菜单或右键单击该解决方案，然后选择“重定解决方案目标”
# 原因：
这个项目是从另外一台电脑copy过来的，之前的编译器VS2013，现在是VS2017，所以报的这个错误。工具集和VS版本有以下对应关系：
  v142–>VS2019
  v141–>VS2017
  v140–>VS2015
  v120–>VS2013
 所以该问题产生的原因是没有安装v120工具集。但是我们不需要额外下载安装v120，仅需更改为vs2017的工具集v141即可。
# 解决方法：
将该解决方案更换成当前VS2017已有的生成工具v141，打开项目-属性-平台工具集，将其由v120改为v141。然后右键选择“重定向项目”，再重新编译该项目。

![1](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E9%97%AE%E9%A2%98%E4%B8%80%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88.png)
![2](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E9%97%AE%E9%A2%98%E4%B8%80%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%882.png)

此时Kclient和Kserver虽然编译不通过但是报错明显减少，并且报错信息中已经没有刚才所述的问题一。record_player则可以直接通过编译，甚至设置为启动项目后可以直接跑。而项目model_bind则仍然会报错“无法找到 v140 的生成工具(平台工具集 =“v140”)。若要使用 v140 生成工具进行生成，请安装 v140 生成工具。或者，可以升级到当前 Visual Studio 工具，方式是通过选择“项目”菜单或右键单击该解决方案，然后选择“重定解决方案目标””

这是因为这个项目是用vs2015写的，需要继续更改一次项目-属性-平台工具集，将其由v140改为v141.然后右键选择“重定向项目”，再重新编译该项目。

![3](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E9%97%AE%E9%A2%98%E4%B8%80%E9%97%AE%E9%A2%98.png)

# 问题二：
显示“无法找到glXXX”
# 原因：
没有安装freeglut
# 解决方法：
安装方式见以下帖子中对于freeglut安装的描述

(注意：这个版本是我之前安装过的freeglut3.2.1，和需要的3.0.0不一致，现在使用的是3.2.1的lib include，链接器的输入也是freeglutd.lib freeglut_stasticd.lib 同时把Kclient和model_bind文件夹的原freeglut.dll备份后更换为3.2.1的freeglutd.dll。目前影响未知)问问问？？？

[20220707另加freeglut](https://blog.csdn.net/weixin_44848751/article/details/124830818?spm=1001.2014.3001.5501)

安装完成后，要对代码进行简单的修改。
Kserver-base_class.h 21行 #include<gl/freeglut.h>->#include<GL/freeglut.h>   

目前Kclient也已经编译完成

# 问题三：
显示“无法打开包括文件Eigen/Dense”

![4](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E5%AE%89%E8%A3%85Eigen.png)
# 原因：
没有安装Eigen

# 解决方法：
[安装Eigen](https://blog.csdn.net/weixin_43940314/article/details/115456199)

# 问题四：
无法打开源文件: “..\..\..\Kinect\Kinect\CCode\Include\ctools\RigidReg.cpp”: No such file or directory
![4](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E6%97%A0%E6%B3%95%E6%89%93%E5%BC%80%E6%96%87%E4%BB%B6.png)

# 原因：
把别人的项目拿来用，但是你所用电脑上VC所安装的位置和原作者的不一样，也会出现这样的错误
[参考原因](https://blog.csdn.net/u010550883/article/details/40739415)
# 解决方法：
记事本更改vcproj中关于RigidReg.cpp的路径,然后重新加载
![4](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E6%89%8B%E5%8A%A8%E6%9B%B4%E6%94%B9.png)
![4](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E6%94%B9%E6%88%90%E8%BF%99%E6%A0%B7.png)
至此Kserver也已经编译完成


# 问题五：
无法解析的外部符号 __imp_glutReshapeFunc	model_bind ...
![4](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E6%97%A0%E6%B3%95%E8%A7%A3%E6%9E%90%E5%A4%96%E9%83%A8%E7%AC%A6%E5%8F%B7.png)

# 原因：
因为自己用cmake默认生成的freeglut只有win32的选项~，所以需要生成64位的版本。
[参考原因](http://pkxpp.github.io/2019/10/20/win10%E4%B8%8BDeepMimic%E5%AE%89%E8%A3%85/)


# 解决方法：
测试的时候也可以将这里设置为win32,或者重新编译64版本的

![6](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/MotionCatch/PICS/%E8%AE%BE%E7%BD%AE.png)

至此model_bind也已经编译完成



