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

	// ���������ļ�Ԥ����
	void load_config_and_prebuild_network( string& conf_fn, int h, int w);

	// �趨ģ������
	void set_network_index(int idx) { model_idx = idx; }

	// ��������
	bool setup_network(int idx,  vector<int>& ic_nb);
	int get_network_idx() { return model_idx; };

	// Ԥ������ (forward fake data)
	bool warmup_network();

	// �趨��������
	bool set_input_tensor( vector< vector<float>>& data);

	// Ԥ�����
	bool predict_output( vector<int>& nodes_idx,  vector< vector<float>>& net_outputs);

	//���������ʾ
	bool display_Output(vector< vector<float>>& net_outputs);


public:
	int model_idx;
	 string model_dir;										//ģ�ʹ洢·��		��δ��ʼ����
	 vector< string> model_names;						//ģ������
	 vector< vector< string>> input_node_names;		//����ڵ�����
	 vector< vector<int>> input_node_channels;			//����ڵ�ͨ����
	 vector< vector< string>> output_node_names;	//����ڵ�����
	int iH, iW;													//����ͼ��ߴ�

	tensorflow::Status m_status;								//! tensorflow ״̬ ��δ��ʼ����
	tensorflow::Session* m_session;								//! tensorflow �Ự
	tensorflow::GraphDef m_graph_def;							//! ͼ����			��δ��ʼ����
	 vector< pair< string, tensorflow::Tensor>> m_inputs;			//��������������
};

//#endif
