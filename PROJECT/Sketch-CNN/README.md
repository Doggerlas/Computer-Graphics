
# 代码升级指令
tf_upgrade_v2 \
  --intree SketchCNN/ \
  --outtree SketchCNN_v2/ \
  --reportfile report.txt
 
# 训练指令
    python train_naiveNet.py --dbTrain=../sampleData/train --dbEval=../sampleData/eval --outDir=../output/train_naiveNet --nb_gpus=2 --devices=0,1 --lossId=0
 
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
    conda remove -n flowers 
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
	




	

	
