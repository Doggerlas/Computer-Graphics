//#ifdef _WITH_GPU_SUPPORT

#pragma once

#define COMPILER_MSVC
#define NOMINMAX

#include <vector>
#include <string>
#include <ostream>
#include <queue>
#include <string>
#include <iostream>
#include <fstream>
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

using namespace std;


class SketchModel {

public:
	SketchModel();
	~SketchModel();

	// 加载配置文件预处理
	void load_config_and_prebuild_network( string& conf_fn, int h, int w);

	// 设定模型索引
	void set_network_index(int idx) { model_idx = idx; }

	// 设置网络
	bool setup_network(int idx,  vector<int>& ic_nb);
	int get_network_idx() { return model_idx; };

	// 预热网络 (forward fake data)
	bool warmup_network();

	// 设定输入张量
	bool set_input_tensor( vector< vector<float>>& data);

	// 预测输出
	bool predict_output( vector<int>& nodes_idx,  vector< vector<float>>& net_outputs);

	//输出向量显示
	bool display_Output(vector< vector<float>>& net_outputs);


public:
	int model_idx;
	 string model_dir;										//模型存储路径		（未初始化）
	 vector< string> model_names;						//模型名称
	 vector< vector< string>> input_node_names;		//输入节点名称
	 vector< vector<int>> input_node_channels;			//输入节点通道数
	 vector< vector< string>> output_node_names;	//输出节点名称
	int iH, iW;													//输入图像尺寸

	tensorflow::Status m_status;								//! tensorflow 状态 （未初始化）
	tensorflow::Session* m_session;								//! tensorflow 会话
	tensorflow::GraphDef m_graph_def;							//! 图定义			（未初始化）
	 vector< pair< string, tensorflow::Tensor>> m_inputs;			//输入名与张量对
};

//#endif
