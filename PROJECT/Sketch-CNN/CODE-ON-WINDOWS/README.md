## 项目一为交互画线进行输入：
	程序输入为交互画线
	项目文件包括 GUI model network  Depth2Ply util_func函数入口在GUI 
	显示结果为3Dply及.\\output\\test_output_img五张图
#### image是项目一GUI.py的调试图片输出文件夹 里面的图片是由GUI.py里的cv2.imwrite绘制的 可以实时显示各笔划
#### template文件夹是里面的template.png是项目一交互视图的模板 应该是白底颜色线 交互视图中是黑底白线 最后算在npr曲线内 是白底黑线
## 项目二为直接读取input_pics文件夹里面的十张图片，经过模型预测进行深度和法线图输出，再经open3d脚本实现3D可视化
	程序输入为读取的十张图片 
	项目文件包括 Input10pics interaction model network Depth2Ply  util_func函数入口在Input10pics 
	输出文件为当前目录下的Test_model.ply(点云文件)及.\\output\\test_output_img中的五张图
#### input_pics是网络输入，需要自行放进
### 以下两个文件夹是两个项目共用的：
#### savedModel存储的是训练的模型
#### output是网络的输出 
	test_input_img是输入进网络的十张图
	test_output_img是网络输出的五张图
	log是日志 param是超参数 pbtxt是计算图(因为不用C++部署所以不重要，用C++需要转化为pb文件，我用的Python部署没有用到它 用到的是savedModel的四个文件)
