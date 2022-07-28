//#include "stdafx.h"

#ifndef _WITH_GPU_SUPPORT
#include "trained_network.h"



SketchModel::SketchModel()
{
	model_idx = 0;
	model_names.clear();
	input_node_names.clear();
	input_node_channels.clear();
	output_node_names.clear();
	iH = iW = -1;

	m_session = NULL;
	m_inputs.clear();
}

SketchModel::~SketchModel()
{
	//m_session->Close();
	if (m_session) delete m_session;
}

//按空格切分字符串
vector<string> split_string(string str, string pattern)
{
	string::size_type pos;
	vector<string> result;
	str += pattern;
	int size = str.size();
	for (unsigned i = 0; i < size; i++)
	{
		pos = str.find(pattern, i);
		if (pos < size)
		{
			string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}

//显示输出vector到文件中
bool SketchModel::display_Output(vector<vector<float>> &net_outputs)
{
	ofstream f("D:\\SketchCNN\\Author_file\\Module\\Frozen_network\\FinalModelFrozen\\outputs.txt", ios_base::app);
	f << "Output is below" << endl;
	for (unsigned i = 0; i < net_outputs.size(); ++i)
	{
		for (unsigned j = 0; j < net_outputs[i].size(); ++j)
		{
			f << "(" << i << "," << j << "):" << net_outputs[i][j] << endl;
		}
	}
	f.close();
	return true;
}

//读取SAS_twoStage_final42K_node_def.txt文件内容到input_node_names input_node_channels output_node_names中
void SketchModel::load_config_and_prebuild_network(string & conf_fn, int h, int w)
{
	ifstream in(conf_fn);
	if (!in.is_open())
	{
		cout << "Error: cannot open input loss file, get: " << conf_fn << endl;
		return;
	}

	// clear data
	model_names.clear();
	input_node_names.clear();
	input_node_channels.clear();
	output_node_names.clear();
   
	// set model directory
	string target="graph_models.txt";
	string tmp=conf_fn;
	int pos = tmp.find(target);
	int n = target.size();
	model_dir = tmp.erase(pos,n);

	//cout << "model_dir  is" <<  model_dir  << endl;
	//cout << "conf_fn is" <<  conf_fn  << endl;
	iH = h;
	iW = w;

	// load model name
	string content;
	while (getline(in, content))
	{
		model_names.push_back(content);
	}
	in.close();

	// load per-model node configuration
	cout << "\n--Init: load network: " << endl;
	// 依次打开以_node_def.txt结尾的文件(虽然作者给的版本中只有一个，即SAS_twoStage_final42K_node_def.txt)
	for (unsigned mitr = 0; mitr < model_names.size(); mitr++)
	{
		 cout << "\t" << model_names[mitr] << " Start load" << endl;
		 string per_model_conf_fn = model_dir + "\\" + model_names[mitr] + "_node_def.txt";
		 ifstream m_model_in(per_model_conf_fn);
		 if (!m_model_in.is_open())
		 {
		 	 cout << "Error: cannot load model " << mitr << " configuration, get: " << per_model_conf_fn <<  endl;
		 	continue;
		 }

		 vector< string> cur_inode_names;		//当前输入节点名称
		 vector<int> cur_inode_cnb;				//当前输入节点通道数
		 vector< string> cur_onode_names;		//当前输出节点

		while ( getline(m_model_in, content))
		{
			// 按空格切分
			 vector< string> sub_strs = split_string(content, " ");
			if (sub_strs[0].compare("Input:") == 0)
			{
				for (int nitr = 1; nitr < sub_strs.size(); nitr++)
				{
					cur_inode_names.push_back(sub_strs[nitr]);
					//cout<<"\t"<< model_names[mitr] <<" cur_inode_names "<<sub_strs[nitr]<<endl;
				}
			}
			else if (sub_strs[0].compare("InputChannelNb:") == 0)
			{
				for (int citr = 1; citr < sub_strs.size(); citr++)
				{
					cur_inode_cnb.push_back( atoi(sub_strs[citr].c_str()));
					//cout<<"\t"<<model_names[mitr]<<" cur_inode_cnb "<<atoi(sub_strs[citr].c_str())<<endl;
				}
			}
			else if (sub_strs[0].compare("Output:") == 0)
			{
				for (int oitr = 1; oitr < sub_strs.size(); oitr++)
				{
					cur_onode_names.push_back(sub_strs[oitr]);
					//cout<< "\t"<<model_names[mitr] <<" cur_onode_names "<<sub_strs[oitr]<<endl;
				}
			}
		}
		m_model_in.close();

		input_node_names.push_back(cur_inode_names);
		input_node_channels.push_back(cur_inode_cnb);
		output_node_names.push_back(cur_onode_names);
	}
	cout << "\n--Init: End load network: " << endl;
}

bool SketchModel::setup_network(int idx,  vector<int>& ic_nb)
{
	 cout << "\n--Setup network Start..." << endl;
	 cout << "\n--Current network: " << model_names[idx] << "\n\tInput channels:";
	// set model index
	model_idx = idx;//就是第几个模型(graph_models.txt里面有几个名字) 作者版本只有一个即SAS_twoStage_final42K，所以SAS_twoStage_final42K的model_idx就是0
	// get input tensor channel size
	ic_nb.clear();
	for (unsigned citr = 0; citr < input_node_channels[model_idx].size(); citr++)
	{
		cout << input_node_channels[model_idx][citr] << " ";
		ic_nb.push_back(input_node_channels[model_idx][citr]);
	}
	 cout << "\n" <<  endl;

	// new session
	if (m_session) delete m_session;
	tensorflow::SessionOptions session_options;
	session_options.config.mutable_gpu_options()->set_allow_growth(true);				   //设置存随实际情况而增长
	session_options.config.mutable_gpu_options()->set_per_process_gpu_memory_fraction(0.5);//控制占用GPU内存的百分比

	m_status = tensorflow::NewSession(/*tensorflow::SessionOptions()*/session_options, &m_session);
	if (!m_status.ok())//检查可否创建tf会话
	{
		 cout << "Error: cannot create tensorflow session, get: " << m_status.ToString() <<  endl;
		return false;
	}

	string model_fn = model_dir + "\\"  + model_names[model_idx] + "_frozen.pb";
	m_status = tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_fn, &m_graph_def);
	if (!m_status.ok())//检查可否加载计算图
	{
		 cout << "Error: cannot load graph definition, get: " << m_status.ToString() << "\n\tModel file: " << model_fn <<  endl;
		return false;
	}

	m_status = m_session->Create(m_graph_def);//检查能否将计算图加入到当前会话
	if (!m_status.ok())
	{
		 cout << "Error: add graph to current session, get: " << m_status.ToString() <<  endl;
		return false;
	}

	if (!warmup_network())
	{
		return false;
	}
	cout << "\n--Setup network End..." << endl;
	return true;
}

bool SketchModel::warmup_network()
{
	 cout << "\n--Warmup network Start..."<<endl;

	// set input
	vector< vector<float>> input_data;//类似于tf中 先生成空张量 这里是生成7*65536的张量
	for (unsigned itr = 0; itr < input_node_channels[model_idx].size(); itr++)//作者版本的只有一个模型就是SAS_twoStage_final42K 它的输入节点的通道数为7 所以input_node_channels[0].size()=7，itr为0到6
	{
		int dSize = 1 * iH*iW*input_node_channels[model_idx][itr];//1*256*256*1，最后一个1是因为模型0 SAS_twoStage_final42K的7个输入节点的通道数都是1
		vector<float> cur_input_data(dSize, 0.0);
		input_data.push_back(cur_input_data);//input_data大小为7 每个单元都是65536个0.0，
	}
	//display_Output(input_data);//全0
	if (!set_input_tensor(input_data))
	{
		return false;
	}

	// predict
	vector< vector<float>> net_outputs;//定义输出张量
	if (!predict_output( vector<int>(), net_outputs))
	{
		 cout << "Error: cannot predict shape, please check tensor filling!!!" <<  endl;
		return false;
	}
	 display_Output(net_outputs);//显示输出张量
	 cout << "\n--Warmup network End..." << endl;
	 cout << "\n--Process is done\n" <<  endl;
	return true;
}

//往m_inputs中填入数值
bool SketchModel::set_input_tensor( vector< vector<float>>& data)
{
	cout << "\n--Setinput Starting" << endl;
	if (data.size() != input_node_names[model_idx].size())//判断输入通道数与输入节点数是否匹配
	{
		 cout << "Error: cannot match data with input node" <<  endl;
		return false;
	}

	m_inputs.clear();
	for (unsigned itr = 0; itr < input_node_names[model_idx].size(); itr++)//input_node_names[model_idx].size()=input_node_names[0].size()=7
	{
		tensorflow::Tensor cur_input(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, iH, iW, input_node_channels[model_idx][itr] }));//针对二值图像的4维张量 1代表每次处理一张 然后是图片长宽 最后是图片通道数 1代表灰度图 3代表rgb
		/*auto cur_tensor_data_ptr = cur_input.flat<float>().data();//感觉这句话没啥用,注释掉也没啥奇怪的，后期有问题再解除注释吧
		for (unsigned ditr = 0; ditr < data[itr].size(); ditr++)//0到65535
		{
			cur_tensor_data_ptr[ditr] = data[itr][ditr];
		}
		*/
		//display_Output(data);//全0
		m_inputs.push_back( pair< string, tensorflow::Tensor>(input_node_names[model_idx][itr], cur_input));//输入节点名称与该节点对应的1*256*256*1的张量 对
	}
	cout << "\n--Setinput End: " << endl;
	return true;
}

bool SketchModel::predict_output( vector<int>& nodes_idx,  vector< vector<float>>& net_outputs)
{
	// forward network
	cout << "\n--Predict Starting: " << endl;
	 vector<string> output_nodes;
	if (nodes_idx.size() == 0)
	{
		output_nodes = output_node_names[model_idx];
	}
	else
	{
		for (unsigned nitr = 0; nitr < nodes_idx.size(); nitr++)
		{
			output_nodes.push_back(output_node_names[model_idx][nitr]);
		}
	}

	// run the forward pass and store the resulting tensor to outputs
	 vector<tensorflow::Tensor> outputs;
	m_status = m_session->Run(m_inputs, output_nodes, {}, &outputs);

	if (!m_status.ok())
	{
		 cout << "Error: cannot run predictions, get: \n" << m_status.ToString() <<  endl;
		return false;
	}

	// fetch output tensor
	net_outputs.clear();
	for (unsigned opt_itr = 0; opt_itr < output_nodes.size(); opt_itr++)
	{
		tensorflow::Tensor cur_t = outputs[opt_itr];
		 vector<float> cur_net_out;

		int nb_elts = cur_t.NumElements();

		auto tensor_data = cur_t.flat<float>().data();
		for (int d_itr = 0; d_itr < nb_elts; d_itr++)
		{
			cur_net_out.push_back(tensor_data[d_itr]);
		}

		net_outputs.push_back(cur_net_out);
	}
	cout << "\n--Predict: End " << endl;
	return true;
}

int main(int argc, char* argv[])
{
	SketchModel SK;
	SK.load_config_and_prebuild_network(string("D:\\SketchCNN\\Author_file\\Module\\Frozen_network\\FinalModelFrozen\\graph_models.txt"), 256, 256);
	vector<int> ic_nb;
	for (unsigned i = 0; i < SK.model_names.size(); ++i)
	{
		SK.setup_network(i, ic_nb);
	}
	system("pause");
}

#endif