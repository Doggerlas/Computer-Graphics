交互方式的项目文件有：
	GUI.py model.py network.py util_func.py 函数入口在GUI.py 
	启动方式为交互画线
	输出方式为查看图片

直接读取input_pics文件夹里面的十张图片进行输出的方式的项目文件有：
	model_org.py interaction.py network.py util_func.py 函数入口在model_bak0803.py 
	启动方式为参数行 见launch.json
	输出方式为查看图片

example里的example 1-6是网上找的opengl例子 与项目无直接关系

template文件夹是里面的template.png是交互视图的模板 应该是白底颜色线 交互视图中是黑底白线 最后算在npr曲线内 是白底黑线

savedModel存储的是训练的模型

output是网络的输出 
	test_input_img是输入进网络的十张图
	test_output_img是网络输出的五张图
	log是日志 param是超参数 pbtxt是计算图(因为不用C++部署所以不重要，用C++需要转化为pb文件，我用的Python部署没有用到它 用到的是savedModel的四个文件)

image是GUI.py的调试图片输出文件夹 里面的图片是由GUI.py里的cv2.imwrite绘制的 生产时在GUI.py里注释掉了 调试的时候可以取消注释 





