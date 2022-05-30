#ifndef DETECTOR_H
#define DETECTOR_H
#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/dnn/dnn.hpp>
#include <cmath>
#include <inference_engine.hpp>
#include <samples/common.hpp>
#include <samples/ocv_common.hpp>
#include <samples/classification_results.h>

using namespace std;
using namespace cv;
using namespace InferenceEngine;

class Detector
{
public:
	typedef struct {
		float prob;
		std::string name;
		cv::Rect rect;
	} Object;
	Detector();
	~Detector();
	//初始化
	bool init(string xml_path, string bin_path, double cof_threshold, double nms_area_threshold, const int keep_top_k = 10);
	//处理图像\进行推理
	bool process_frame(Mat& srcOriginal, vector<Object> &detected_objects);
	
private:

	//参数区 ===================================
	string _xml_path;             //OpenVINO模型xml文件路径
	string _bin_path;             //OpenVINO模型bin文件路径
	double _cof_threshold;       //置信度阈值,计算方法是框置信度乘以物品种类置信度
	double _nms_area_threshold;  //nms最小重叠面积阈值
	int _keep_top_k;

	const int target_size = 550;
	const float MEANS[3] = { 123.68, 116.78, 103.94 };
	const float STD[3] = { 58.40, 57.12, 57.38 };

	//获得全部 priorbox 生成规则超参
	int num_priors;
	float* priorbox;
	const int conv_ws[5] = { 69, 35, 18, 9, 5 };
	const int conv_hs[5] = { 69, 35, 18, 9, 5 };
	const float aspect_ratios[3] = { 1.f, 0.5f, 2.f };
	const float scales[5] = { 24.f, 48.f, 96.f, 192.f, 384.f };
	const float var[4] = { 0.1f, 0.1f, 0.2f, 0.2f };

	const int mask_h = 138;
	const int mask_w = 138;

	//属性区 ===================================
	//存储初始化获得的可执行网络
	ExecutableNetwork _network;
	OutputsDataMap _outputinfo;
	string _input_name;

	//方法区 ===================================
	//减均值除方差
	//void normalize(Mat& img);
	//将OpenCV Mat对象中的图像数据传给为InferenceEngine Blob对象
	void frameToBlob(const cv::Mat& frame, InferRequest::Ptr& inferRequest, const std::string& inputName);
	// 生成prior
	void make_priors();
	void sigmoid(Mat& out, int length);
	//后处理
	//bool parse_yolact(const Blob::Ptr& blob, int net_grid, float cof_threshold,
		//vector<Rect>& o_rect, vector<float>& o_rect_cof);

	const char* class_names[81] = { "background",
										"person", "bicycle", "car", "motorcycle", "airplane", "bus",
										"train", "truck", "boat", "traffic light", "fire hydrant",
										"stop sign", "parking meter", "bench", "bird", "cat", "dog",
										"horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
										"backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
										"skis", "snowboard", "sports ball", "kite", "baseball bat",
										"baseball glove", "skateboard", "surfboard", "tennis racket",
										"bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
										"banana", "apple", "sandwich", "orange", "broccoli", "carrot",
										"hot dog", "pizza", "donut", "cake", "chair", "couch",
										"potted plant", "bed", "dining table", "toilet", "tv", "laptop",
										"mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
										"toaster", "sink", "refrigerator", "book", "clock", "vase",
										"scissors", "teddy bear", "hair drier", "toothbrush"
	};

	const unsigned char colors[81][3] = {
		{56, 0, 255},
		{226, 255, 0},
		{0, 94, 255},
		{0, 37, 255},
		{0, 255, 94},
		{255, 226, 0},
		{0, 18, 255},
		{255, 151, 0},
		{170, 0, 255},
		{0, 255, 56},
		{255, 0, 75},
		{0, 75, 255},
		{0, 255, 169},
		{255, 0, 207},
		{75, 255, 0},
		{207, 0, 255},
		{37, 0, 255},
		{0, 207, 255},
		{94, 0, 255},
		{0, 255, 113},
		{255, 18, 0},
		{255, 0, 56},
		{18, 0, 255},
		{0, 255, 226},
		{170, 255, 0},
		{255, 0, 245},
		{151, 255, 0},
		{132, 255, 0},
		{75, 0, 255},
		{151, 0, 255},
		{0, 151, 255},
		{132, 0, 255},
		{0, 255, 245},
		{255, 132, 0},
		{226, 0, 255},
		{255, 37, 0},
		{207, 255, 0},
		{0, 255, 207},
		{94, 255, 0},
		{0, 226, 255},
		{56, 255, 0},
		{255, 94, 0},
		{255, 113, 0},
		{0, 132, 255},
		{255, 0, 132},
		{255, 170, 0},
		{255, 0, 188},
		{113, 255, 0},
		{245, 0, 255},
		{113, 0, 255},
		{255, 188, 0},
		{0, 113, 255},
		{255, 0, 0},
		{0, 56, 255},
		{255, 0, 113},
		{0, 255, 188},
		{255, 0, 94},
		{255, 0, 18},
		{18, 255, 0},
		{0, 255, 132},
		{0, 188, 255},
		{0, 245, 255},
		{0, 169, 255},
		{37, 255, 0},
		{255, 0, 151},
		{188, 0, 255},
		{0, 255, 37},
		{0, 255, 0},
		{255, 0, 170},
		{255, 0, 37},
		{255, 75, 0},
		{0, 0, 255},
		{255, 207, 0},
		{255, 0, 226},
		{255, 245, 0},
		{188, 255, 0},
		{0, 255, 18},
		{0, 255, 75},
		{0, 255, 151},
		{255, 56, 0},
		{245, 255, 0}
	};
};
#endif