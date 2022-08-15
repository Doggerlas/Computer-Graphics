# CODE: 7月份升级的代码 包括linux上改造完的训练 测试程序 编译出的decoder等 ； windows上的改造完但是没有使用的cpp文件
# DOCS: 汇报的PPT汇总及原始paper
# PICS: readme用到的图片
# CODE on Windows: 在Win10上开发的交互代码工具 包括交互输入窗口 还有open3d方法的输出

# 代码升级指令
tf_upgrade_v2 \
  --intree SketchCNN/ \
  --outtree SketchCNN_v2/ \
  --reportfile report.txt

# 单文件升级指令
tf_upgrade_v2  --infile ./test_geomNet.py  --outfile ./test_geomNet_v2.py --reportfile report.txt

# docker相关指令
##### 启动docker(因为没有root 所以不用管)
    systemctl start docker         
##### 查看nvidia-docker安装情况(已经有人安装完了)   
    apt show nvidia-docker2 
##### 创建一个可以使用显卡nvidia_docker的容器(这个需要之前有人装过nvidia-docker)
    nvidia-docker run -it --name first_container nvidia/cuda:11.0-base /bin/bash
##### 删除指定的容器    
    docker rm 482ee69cb441     
##### 启动容器
    docker start 482ee69cb441   
##### 关闭容器
    docker stop 482ee69cb441   
##### 进入容器
    docker attach 482ee69cb441         
##### 查看正在运行的容器
    docker ps      
##### 列出所有容器
    docker ps -a            
##### 退出容器    
    exit                               
##### 显示完全id：482ee69cb4415710228d1e64fdf10fb572a86b4fcbdf9924868746457c2b4c6b
    docker inspect --format="{{.Id}}"  first_container      
##### 主机到docker文件传输
    docker cp /home/user1/Wlm/Anaconda3-5.3.0-Linux-x86_64.sh 482ee69cb4415710228d1e64fdf10fb572a86b4fcbdf9924868746457c2b4c6b:/home   
##### docker到主机文件传输
    docker cp 482ee69cb4415710228d1e64fdf10fb572a86b4fcbdf9924868746457c2b4c6b:/home/SketchCNN/report.txt /home/user1
##### 我的docker信息
    CONTAINER ID   IMAGE                   COMMAND       CREATED        STATUS                         PORTS     NAMES
    482ee69cb441   nvidia/cuda:11.0-base   "/bin/bash"   2 hours ago    Up About an hour                         first_container
# [conda相关指令](https://blog.csdn.net/fyuanfena/article/details/52080270)
##### 检查conda版本:
    conda --version
##### 升级当前版本的conda
    conda update conda
##### 创建并激活一个python3.6的环境snowflake
    conda create --name snowflake biopython
##### 激活环境
    conda activate snowflakes
##### 列出所有环境
    conda info -e
##### 切换到另一个环境
    conda activate snowflakes
##### 从你当前工作环境的路径切换到系统根目录
    conda deactivate
##### 通过克隆snowfllakes来创建一个称为flowers的副本
    conda create -n flowers --clone snowflakes
##### 删除一个环境
    conda env remove -n flowers
##### 假设你需要python3来编译程序，但是你不想覆盖掉你的python2.7来升级，你可以创建并激活一个名为snakes的环境，并通过下面的命令来安装最新版本的python3：
    conda create -n snakes python=3 然后检查新的环境中的python版本 python --version
##### 查看已安装包
    conda list
##### 向bunnies环境安装beautifulsoup4包
    conda install --name bunnies beautifulsoup4或者 先激活环境activate bunnies 再安装包conda install beautifulsoup4
##### 通过pip命令来安装包
    先激活source activate bunnies在安装pip install see    

# 20220528 在482ee69cb441容器中
## 1.为了使用GPU新建一个nvidia_docker用户first_container  
##### 创建一个可以使用显卡nvidia_docker的容器
    nvidia-docker run -it --name first_container nvidia/cuda:11.0-base /bin/bash
##### 查看完全id
    docker inspect --format="{{.Id}}"  first_container           #482ee69cb4415710228d1e64fdf10fb572a86b4fcbdf9924868746457c2b4c6b
##### 复制anaconda.sh到容器
    docker cp /home/user1/Wlm/Anaconda3-5.3.0-Linux-x86_64.sh 482ee69cb4415710228d1e64fdf10fb572a86b4fcbdf9924868746457c2b4c6b:/home


## 2.安装vi
	apt-get update
	apt-get upgrade
	apt-get install vim
	
	
## 3.[安装Anaconda](https://blog.csdn.net/LZW15082682930/article/details/116062237)
##### [下载版本](https://blog.csdn.net/weixin_39929687/article/details/111023432)
	Anaconda3-5.3.0-Linux-x86_64.sh	636.9M	2018-09-27 16:01:35	4321e9389b648b5a02824d4473cfdbf	
##### 安装
	bash Anaconda3-5.3.0-Linux-x86_64.sh 安装位置/root/anaconda3
##### 配置
	vi ~/.bashrc 添加环境变量	export PATH="/root/anaconda3/bin:$PATH" 然后source ~/.bashrc
##### 查看版本
	conda --version或conda -V
##### 打开jupyter(暂时用不到)
	jupyter notebook  --allow-root
	
## 4.[anaconda3安装TensorFlow2.6.0-gpu](https://blog.csdn.net/a745233700/article/details/109377039)
### 为什么不能按论文配置成tf1.3.0 python3.6？
   想在docker里用GPU就得使用nvidia-docker，这个之前已经被安装在宿主机了。所以我创造了一个名为first_container的nvidia-docker容器，这个容器的nvidia-smi与宿主机的一致，cuda版本也是与宿主机相同的11.4，当然显卡和cudnn也是一样的，属于是宿主机的映射。
   因为CUDA版本11.4与TensorFlow版本对应，已经不支持1.3了。如果在first_container里硬安装符合1.3.0的CUDA和cudnn版本，造成CUDA与显卡不匹配肯定会出问题。所以我参考宿主机的python版本与TensorFlow版本，打算在first_container里也打算安装相同的TensorFlow2.6.0与python3.8.8。我用anaconda创建了python3.8.8版本的环境tensorflow2.6.0_gpu并在里面安装了tensorflow2.6.0和11.3的cudatoolkit及8.2.1的cudnn，成功支持了GPU选项。
##### 参考宿主机支持GPU和conda 升级conda后再进行布置tf2.6.0环境
	conda update conda		#升级到最新版4.13.0
##### 创建环境
	conda create -n tensorfolw2.6.0_gpu python=3.8.8
	conda info -e
##### 进入环境
	conda activate tensorfolw2.6.0_gpu
	conda list
	python -V
##### 安装tensorfolw2.6.0_gpu
	pip install tensorflow-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple   
###### 出现问题1:TypeError: Descriptors cannot not be created directly
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20220528221544.png)
###### [解决方案:手动降低 protobuf 为 3.x](https://github.com/PaddlePaddle/PaddleSpeech/issues/1970)
	pip install protobuf==3.20.1
##### 出现问题2：安装cudatoolkit和cudnn 避免出现链接库.so找不到的问题
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20220528221305.png)
###### 解决方案:安装cudatoolkit与cudnn
	conda install cudatoolkit=11 
	conda install cudnn

##### 以下在python3打开
		import tensorflow as tf
		tf.test.is_built_with_cuda()	#检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
		tf.test.is_gpu_available()	#检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
![结果](https://github.com/Doggerlas/C-primer/blob/main/pics/1212.png)
###### 卸载指令(不是版本问题不要卸载)
	pip uninstall tensorflow tensorflow-gpu

# 20220531 在482ee69cb441容器中
## 1.安装Zlib
	apt-get update
	apt-get install zlib1g-dev
![问题](https://github.com/Doggerlas/C-primer/blob/main/pics/zlib.png)
## 2.安装opencv-python
	pip install opencv-python
![问题](https://github.com/Doggerlas/C-primer/blob/main/pics/opencv-python.png)
######  出现问题3：python3执行import cv2出现:ImportError: libGL.so.1: cannot open shared object file: No such file or directory
######  [解决方案](https://blog.csdn.net/iLOVEJohnny/article/details/121077928)
	apt-get update
	apt install libgl1-mesa-glx
## 3.[安装opencv](https://blog.csdn.net/qq_38660394/article/details/80581383)
######  原代码库安装方式 apt-get install libopencv-dev已经不能使用，因为那种方式支持的是cv2.0版本。现在手动安装opencv3.2.0版本
######  安装依赖
	apt-get update  
	apt-get install build-essential
	apt-get install cmake
	apt-get install git 
	apt-get install pkg-config
	apt-get install libgtk2.0-dev
	apt-get install libavcodec-dev libavformat-dev libswscale-dev
######  出现问题4：获取资源报错
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20220531124312.png)
######  解决方案：尝试了很多安装源，修改/etc/apt/source.list文件，但是都不行，还会出现umet依赖问题。困扰了30号一天未解决。31号重新执行apt-get install libavcodec-dev libavformat-dev libswscale-dev成功，推测是动态ip问题。
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_202205311.png)
######  下载压缩包
	wget -O opencv.zip https://github.com/Itseez/opencv/archive/3.2.0.zip 
	unzip opencv.zip  
######  编译
	cd opencv-3.2.0 
    	mkdir build  
    	cd build  
    	cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local ..  
    	make -j8
######  出现问题5:编译报错
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/que.png)
######  [解决方案:版本问题，更改源码](https://blog.csdn.net/goodxin_ie/article/details/82856008)
	vi /root/opencv-3.2.0/modules/videoio/src/cap_ffmpeg_impl.hpp
######  增加以下宏，删除build重新编译
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1solv.png)
######  部署
	make install

# 20220628 在482ee69cb441容器中
## 1.本次编译custom_dataDecoder.so 文件
由于opencv3版本已经删除了部分so文件，为了编译通过，build.sh脚本也做了相应的更改，将opencv3.2.0版本没有的依赖so文件删除。有以下五个

![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1%E5%9B%BE%E7%89%87_20220628153012.png)

原build为build.sh.bak.二者对比如下
新build.sh

![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%96%B0build.png)

原build.sh

![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%8E%9Fbuild.png)
######  注意：缺少这五个文件完成的编译结果造成的影响未知

## 2.关于network和loader文件的修改
python项目相互调用是将文件夹中的每个文件看做是一个pakege(每个文件夹都有个__init__.py),该文件的方法视为一个module(比如from   utils.util_func import cropconcat_layer就是调用的utils/util_func.py的cropconcat_layer方法)

需要事先在bashrc中定义python路径 才能调用.这里我定义的pythonpath是上一级目录 也就是/root/SketchCNN/network/所以代码中就直接写network下级目录就行

![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/pythonpath.png)

修改完后，记得source ~/.bashrc
因为network与loader是方法集合，不涉及数据处理，可以直接python3 loader.py python3 network.py试试 没有错误就说明引用没错 但是并不代表这两个脚本没错

######  出现问题6：ModuleNotFoundError: No module named ‘tensorflow.contrib‘
######  原因：tensorflow 2.0以后没有 tensorflow.contrib 采用以下方法2进行代码修改
######  [解决方案](https://blog.csdn.net/qq_38251616/article/details/114820099)

## 3.训练
训练指令(以nativeNet为例)

######  出现问题7：tf.placeholder() is not compatible with eager execution  
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E9%94%99%E8%AF%AF1.png)
######  原因：TensoFlow2.0及以上的版本出现这个问题，还是版本问题
######  [解决方案](https://blog.csdn.net/weixin_43763859/article/details/104537392)
![解决](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E8%A7%A3%E5%86%B31.png)

######  出现问题8：ImportError: cannot import name 'dtensor' from 'tensorflow.compat.v2.experimental'
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E9%94%99%E8%AF%AF2.png)
######  原因：缺少keras 
######  [解决方案](https://codecary.com/solved-importerror-cannot-import-name-dtensor-from-tensorflow-compat-v2-experimental/)：pip install keras==2.6

######  出现问题9：AttributeError: module 'tensorflow' has no attribute 'contrib'
![问题](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%BE%AE%E4%BF%A1%E6%88%AA%E5%9B%BE_20220628162133.png)
######  原因：过时了 contrib已经不用了 自己换了个函数
######  ![解决方案](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/ebb7f67d52479105997c4048dbfce6d.png)

##  	4.错误已解决 开始进行训练 :
######  训练指令(以sampledata(49M)作为数据集)
![解决方案](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%9E%B6%E6%9E%84.png)
    	
	python train_naiveNet.py --dbTrain=../sampleData/train --dbEval=../sampleData/eval --outDir=../output/train_naiveNet --nb_gpus=2 --devices=0,1 --lossId=0
	
	python train_baselineNet.py --dbTrain=../sampleData/train --dbEval=../sampleData/eval --outDir=../output/train_baselineNet --nb_gpus=2 --devices=0,1 
	
	python train_dfNet.py --dbTrain=../sampleData/train --dbEval=../sampleData/eval --outDir=../output/train_dfNet --nb_gpus=2 --devices=0,1 
	
	python train_geomNet.py --field_ckpt=../output/train_dfNet/savedModel --dbTrain=../sampleData/train --dbEval=../sampleData/eval --outDir=../output/train_geomNet --nb_gpus=2 --devices=0,1 

######  训练指令(以data(157G)作为数据集(我把data处理成了和sampledata一样的架构))
![解决方案](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%9E%B6%E6%9E%842.png)

	python train_naiveNet.py --dbTrain=../data/train --dbEval=../data/eval --outDir=../output/train_naiveNet --nb_gpus=2 --devices=0,1 --lossId=0
	
	python train_baselineNet.py --dbTrain=../data/train --dbEval=../data/eval --outDir=../output/train_baselineNet --nb_gpus=2 --devices=0,1 
	
	python train_dfNet.py --dbTrain=../data/train --dbEval=../data/eval --outDir=../output/train_dfNet --nb_gpus=2 --devices=0,1 
	
	python train_geomNet.py --field_ckpt=../output/train_dfNet/savedModel --dbTrain=../data/train --dbEval=../data/eval --outDir=../output/train_geomNet --nb_gpus=2 --devices=0,1 


# 训练图
![训练](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E8%AE%AD%E7%BB%83.png)



# 20220629 在台式机上
## 1.文件获取(网络模型由原作者提供 我想先拿原作者的成功实现简单地交互 进行下去以后 再使用自己的)
### 需要的材料：
[原始数据集](https://connecthkuhk-my.sharepoint.com/personal/changjli_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchangjli%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSketchCNN%2FRelease%2FTrainingData%2FSketchCnnFinal&ga=1)

[冻结的网络](https://connecthkuhk-my.sharepoint.com/personal/changjli_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchangjli%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSketchCNN%2FRelease%2FFinalModelFrozen&ga=1)

[检查点](https://connecthkuhk-my.sharepoint.com/personal/changjli_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchangjli%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSketchCNN%2FRelease%2FCheckpoint&ga=1)

[预构建的 tensorflow 库和 dll](https://connecthkuhk-my.sharepoint.com/personal/changjli_connect_hku_hk/_layouts/15/onedrive.aspx?id=%2Fpersonal%2Fchangjli%5Fconnect%5Fhku%5Fhk%2FDocuments%2FSketchCNN%2FRelease%2FTensorFlow&ga=1)

# 20220630 在台式机上
## 1.链接提供的库文件
### 1.首先这是我的文件存储架构 

--Checkpoint是网络检查点，里面只有我们需要的网络(baseline和nativeNet没有下载) 

--FinalModelFrozen是冻结的网络

--Pre_built_lib_and_dl是预编译的lib和dll文件

![架构](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E7%BB%93%E6%9E%84.png)

### 2.VS2015配置
--新建project，记得调试模式选择Release x64，配置的界面也要在此模式下进行。不然就会出现.o文件链接失败的问题 
(Release/Debug 以及x64/x86的四种搭配环境的搭建选择是独立的，也就是在Release x64环境下链接的文件在Release x86进行调试程序时是查不到的)
![R](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/R.png)
![R](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%A1%862.png)

#### 配置lib文件

	设置库路径
	
	Project->SketchCNN properties->Configuration Properties->VC++ Directories->General->Library Directories 添加以下两个lib文件路径
	
	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\lib;D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\cudnn;（注意不要替换原有项）

添加完成后如下红框所示

![1](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%A1%86.png)

	设置链接器
	
	Project->SketchCNN properties->Configuration Properties->Linker->Input->Additional Dependencies 添加以下两个lib文件
	
	tensorflow.lib cudnn.lib

添加完成后如下红框所示

![1](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%A1%861.png)
	
	设置附加库路径
	
	Project->SketchCNN properties->Configuration Properties->>Linker->General->Additional Library Directories .
	
	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\lib;D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\cudnn

添加完成后如下红框所示

![1](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%A1%864.png)

#### 配置dll文件
	
	将bin中的tensorflow.dll和cudnn64_6.dll放到项目工程目录C:\Users\Sim\Documents\Visual Studio 2015\Projects\SketchCNN\SketchCNN

![1](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%A1%863.png)

##### 配置其他Include头文件
	
	Project->SketchCNN properties->Configuration Properties->VC++ Directories->General->Include Directories 添加以下路径
	
	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\include;
	
	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\include\eigen_archive;
	
	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\include\google\protobuf;

	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\include\third_party\eigen3;

	D:\SketchCNN\Author_file\Module\Pre_built_lib_and_dl\cudnn;

	![1](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E6%A1%865.png)

######  出现问题10：trained_network.obj : error LNK2001: unresolved external symbol "private: void __thiscall tensorflow(编译没通过 出现了很多链接错误)
######  原因：应该在Release x64下进行环境配置
######  解决方案：调试与配置的环境统一为Release x64

######  出现问题11：编译通过 但是缺少一些cuda8.0的dll
######  原因：缺什么下载什么 都放在项目工程目录C:\Users\Sim\Documents\Visual Studio 2015\Projects\SketchCNN\SketchCNN
######  因为我没装CUDA8.0 所以这里我直接补充的依赖dll包 

[cublas64_80.dll cudart64_80.dll curand64_80.dll](https://download.csdn.net/download/qq_29592829/10704068)

[cufft64_80.dll](https://download.mersenne.ca/CUDA-DLLs/CUDA-8.0)

[cusolver64_80.dll 这个下载链接是cusolver64_100的 下载之后手动改名为cusolver64_80即可 ](https://download.csdn.net/download/t_qrqt/12433808?utm_medium=distribute.pc_relevant_download.none-task-download-2~default~BlogCommendFromBaidu~Rate-3-12433808-download-15631170.dl_show_rating&depth_1-utm_source=distribute.pc_relevant_download.none-task-download-2~default~BlogCommendFromBaidu~Rate-3-12433808-download-15631170.dl_show_rating&dest=https%3A%2F%2Fdownload.csdn.net%2Fdownload%2Ft_qrqt%2F12433808&spm=1003.2020.3001.6616.4)

######  ![解决方案：](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/%E5%B7%A5%E7%A8%8B.png)

# 20220715 在服务器上 升级test_gemoNet.py到tf2.0

######  测试指令
	python3 test_geomNet.py --cktDir=../output/train_geomNet/savedModel --dbTest=../data/test --outDir=../output/test/test_geomNet --device=0,1  --graphName=SAS_2stage_GeoNet.pbtxt
	
##### 测试sampledata指令
	python3 test_geomNet.py --cktDir=../output/train_geomNet/savedModel --dbTest=../sampleData/test --outDir=../output/test/test_geom_sampleData_Net --device=0,1  --graphName=SAS_2stage_sampleData_GeoNet.pbtxt

######  出现问题12：缩进错误Inconsistent use of tabs and spaces in indentation
######  原因：很明显问题出在了缩进上
######  解决方法：巨坑！！！python认为tab与space不是等价的，需要把所有空格删除，再用tab进行补全。但是看是一点也看不出来的。卡了我一小时，GG

######  出现问题13：Opencv版本问题 OpenEXR codec is disabled. You can enable it via 'OPENCV_IO_ENABLE_OPENEXR'
######  原因：很明显EXR这个功能被禁用了，当然我是不会做版本回退这种憨憨事的
######  解决方法：打开我的test_gemoNet.py把这一行：os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"加到文件开头就行了

其他的test文件也可以参照这个文件升级，其实没多大改动，用compare看看吧

# 20220719 在服务器上 升级freeze_graph_tool.py到tf2.0

##### (失败)冻结指令 开始会提示多了个参数：会提示variable_names_blacklist参数错误，删去这个参数，执行以下语句会提示IndexError: tuple index out of range
##### 注：--net_type 0-naiveNet、1-baselineNet、2-GeomNet

	python3 freeze_graph_tool.py --output_dir=../output/test/test_geomNet --ckpt_dir=../output/train_geomNet/savedModel --ckpt_name=SAS_2stage_GeoNet.pbtxt --graph_name=SAS_2stage_GeoNet.pb --net_type=2 

![训练](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/1112.png)

![训练](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/1113.png)

##### (失败)尝试1：想用meta文件而不是pbtxt生成模型，语句中 --ckpt_name=SAS_2stage_GeoNet.pbtxt 没意义，程序中已经写死input_graph_path = '../output/train_geomNet/savedModel/my_model498000.ckpt.meta' 会提示：google.protobuf.message.DecodeError: Error parsing message with type 'tensorflow.GraphDef'
	python3 freeze_graph_tool_v2.py --output_dir=../output/test/test_geomNet --ckpt_dir=../output/train_geomNet/savedModel --ckpt_name= --graph_name=SAS_2stage_GeoNet.pb --net_type=2 

##### (失败)尝试2：想把pbtxt转换成pb作为计算图输入，先执行pbtxt2pb.py脚本将SAS_2stage_GeoNet.pbtxt转换为tmp.pb 再执行以下语句也提示IndexError: tuple index out of range
	python3 freeze_graph_tool_v3.py --output_dir=../output/test/test_geomNet --ckpt_dir=../output/train_geomNet/savedModel --ckpt_name=tmp.pb --graph_name=SAS_2stage_GeoNet.pb --net_type=2 

##### 尝试3：为了搞明白到底是模型问题还是freeze_graph存在bug，我打算测试作者SAS_twoStage_final42K.pbtxt文件能否冻结 
##### (成功)成功冻结 在../output/Autor_give/Frozen_network/FinalModelFrozen/ 下生成SAS_twoStage_final42K_frozen.pb


	python3 freeze_graph_tool.py --output_dir=../output/Autor_give/Frozen_network/FinalModelFrozen/ --ckpt_dir=../output/Autor_give/fullNetwork --ckpt_name=SAS_twoStage_final42K.pbtxt --graph_name=SAS_twoStage_final42K_frozen.pb --net_type=2 

![训练](https://github.com/Doggerlas/Computer-Graphics/blob/main/PROJECT/Sketch-CNN/PICS/1114.png)

##### 好吧 现在我知道了tf2.0没有冻结代码这个功能

# 20220815 在笔记本2080 win0上 部署并实现
## 环境搭建
##### 完全使用anaconda执行以下指令
##### 创建环境TensorFlow2.6.0+open3d 0.7.0 
	conda create -n tf260gpu+open3d python=3.6.8
##### 激活环境
	conda activate tf260gpu+open3d
##### 代码中有几个o3d的API仅支持python3.6 所以需要安装py3.6
	pip install tensorflow-gpu==2.6.0 -i https://pypi.tuna.tsinghua.edu.cn/simple   
	conda install cudatoolkit=11 
	conda install cudnn
##### 测试是否可以使用GPU及CUDA
	import tensorflow as tf
	tf.test.is_built_with_cuda()	#检查tensorflow是否得到CUDA支持，安装成功则显示true，否则为false
	tf.test.is_gpu_available()	#检查tensorflow是否可以获取到GPU，安装成功则显示true，否则为false
##### 安装opencv-python
	pip install opencv-python
##### 不能直接使用pip安装pyopengl 因为直接安装的是32位的 需要[先预下载对应于python3.6版本的pyoengl](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyopengl)，从anaconda中cd到该whl安装包路径下进行安装
	pip install PyOpenGL_accelerate-3.1.5-cp36-cp36m-win_amd64.whl
	pip install PyOpenGL-3.1.5-cp36-cp36m-win_amd64.whl
##### 以下包用于3D显示
	pip install Pillow pandas numpy open3d-python
##### 以下包用于模型测试
	pip install tf_slim
	pip install keras==2.6
