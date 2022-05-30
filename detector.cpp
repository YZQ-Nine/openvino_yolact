#include "detector.h"
//#include <opencv2/dnn.hpp>

//std::string DEVICE = "MYRIAD";
std::string DEVICE = "CPU";

Detector::Detector() {}

Detector::~Detector() {}


//减均值、归一化处理
//void Detector::normalize(Mat& img) {
//	img.convertTo(img, CV_32F);
//	int i = 0, j = 0;
//	for (i = 0; i < img.rows; i++)
//	{
//		float* pdata = (float*)(img.data + i * img.step);
//		for (j = 0; j < img.cols; j++)
//		{
//			pdata[0] = (pdata[0] - this->MEANS[0]) / this->STD[0];
//			pdata[1] = (pdata[1] - this->MEANS[1]) / this->STD[1];
//			pdata[2] = (pdata[2] - this->MEANS[2]) / this->STD[2];
//			pdata += 3;
//		}
//	}
//}

//https://zhuanlan.zhihu.com/p/456238585

//void Detector::substract_mean_normalize(const float* mean_vals, const float* norm_vals) { 
//	int size = w * h;
//
//	for (int q = 0; q < c; q++)
//	{
//		float* ptr = data + cstep * q;
//		const float mean = mean_vals[q];
//		const float norm = norm_vals[q];
//
//		int remain = size;
//
//		for (; remain > 0; remain--)
//		{
//			*ptr = (*ptr - mean) * norm;
//			ptr++;
//		}
//	}
//}

void Detector::sigmoid(Mat& out, int length)
{
	float* pdata = (float*)(out.data);
	int i = 0;
	for (i = 0; i < length; i++)
	{
		pdata[i] = 1.0 / (1 + expf(-pdata[i]));
	}
}

//将OpenCV Mat对象中的图像数据传给为InferenceEngine Blob对象
void Detector::frameToBlob(const cv::Mat& frame, InferRequest::Ptr& inferRequest, const std::string& inputName) {
	//从OpenCV Mat对象中拷贝图像数据到InferenceEngine 输入Blob对象
	Blob::Ptr frameBlob = inferRequest->GetBlob(inputName);
	matU8ToBlob<uint8_t>(frame, frameBlob);
}

void Detector::make_priors() {
	this->num_priors = 0;
		int p = 0;
		for (p = 0; p < 5; p++)
		{
			this->num_priors += this->conv_ws[p] * this->conv_hs[p] * 3;
		}
		//std::cout << "num_priors = "<< this->num_priors << std::endl;
		this->priorbox = new float[4 * this->num_priors];
		////generate priorbox
		float* pb = priorbox;
		for (p = 0; p < 5; p++)
		{
			int conv_w = this->conv_ws[p];
			int conv_h = this->conv_hs[p];

			float scale = this->scales[p];

			for (int i = 0; i < conv_h; i++)
			{
				for (int j = 0; j < conv_w; j++)
				{
					// +0.5 because priors are in center-size notation
					float cx = (j + 0.5f) / conv_w;
					float cy = (i + 0.5f) / conv_h;

					for (int k = 0; k < 3; k++)
					{
						float ar = aspect_ratios[k];

						ar = sqrt(ar);

						float w = scale * ar / this->target_size;
						float h = scale / ar / this->target_size;

						// This is for backward compatability with a bug where I made everything square by accident
						// cfg.backbone.use_square_anchors:
						h = w;
						pb[0] = cx;
						pb[1] = cy;
						pb[2] = w;
						pb[3] = h;
						pb += 4;
					}
				}
			}
		}

}

//初始化
bool Detector::init(string xml_path, string bin_path, double cof_threshold, double nms_area_threshold, const int keep_top_k) {
	try {

		// 生成prior
		make_priors();

		_xml_path = xml_path;
		_bin_path = bin_path;

		_cof_threshold = cof_threshold;
		_nms_area_threshold = nms_area_threshold;
		_keep_top_k = keep_top_k;

		// --------------------------- 1. 创建Core对象 --------------------------------------
		// 加载推理引擎Core，该引擎需要从当前路径加载plugins.xml文件
		Core ie;

		// ------------------- 2. 读取IR文件 (.xml and .bin files) --------------------------
		auto cnnNetwork = ie.ReadNetwork(_xml_path);
	
		// -------------------- 3. 配置网络输入输出 -----------------------------------------
		InputsDataMap inputInfo(cnnNetwork.getInputsInfo());
		InputInfo::Ptr& input = inputInfo.begin()->second;
		_input_name = inputInfo.begin()->first;

		input->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
		input->getInputData()->setLayout(Layout::NCHW);
		input->setPrecision(Precision::U8);

		//输出设置 输出格式设置 Yolact 4个输出
		_outputinfo = OutputsDataMap(cnnNetwork.getOutputsInfo());
		for (auto &output : _outputinfo) {
			std::string output_name = output.first;
			auto output_data = output.second;
			output_data->setPrecision(Precision::FP32);
		}

		// --------------------------- 4. 载入模型到AI推理计算设备---------------------------------------
		_network = ie.LoadNetwork(cnnNetwork, DEVICE);

		////https://zhuanlan.zhihu.com/p/133508719
		//std::map<std::string, std::string> config = { { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } };
		//_network = ie.LoadNetwork(cnnNetwork, "CPU", config);
		return true;
	}
	catch (std::exception & e) {
		return false;
	}
}

//处理图像获取结果
//模型的输入维度是RGB
bool Detector::process_frame(Mat& srcOriginal, vector<Object>& detected_objects) {

	int img_w = srcOriginal.cols;
	int img_h = srcOriginal.rows;

	// --------------------------- 5. 创建Infer Request--------------------------------------------
	InferRequest::Ptr infer_request = _network.CreateInferRequestPtr();

	//======================== 输入格式方法 ================

	//1. 自己通道转换RGB

	////获取输入的Blob 格式转换类对象
	auto input = infer_request->GetBlob(_input_name);//获取input的Blob(进行输入的格式设置的类对象)
	size_t num_channels = input->getTensorDesc().getDims()[1]; //3
	size_t H = input->getTensorDesc().getDims()[2];    // 550
	size_t W = input->getTensorDesc().getDims()[3];    // 550
	
	// 2. 将通道转换和mean_values、scale_values 直接集成到模型中
	cv::Mat src;
	cv::resize(srcOriginal, src, cv::Size(550, 550));//转换大小
	Blob::Ptr imgBlob = wrapMat2Blob(src); 
	infer_request->SetBlob(_input_name, imgBlob);

	//========================================= end =============================================

	
	// --------------------------- 7. 执行推理计算 ------------------------------------------------
	auto start = std::chrono::system_clock::now();
	infer_request->Infer();
	auto end = std::chrono::system_clock::now();
	std::cout << "inference time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl << std::endl;

	// ---------------------------8. 输出处理 ------------------------------------------------//
	Blob::Ptr output = infer_request->GetBlob("1270");
	// Print classification results
	ClassificationResultW classificationResult(output, { L"E:\\Dev\\Demo\\NCS2\\Test\\yolact_ncs2_openvino_2021_4\\img\\test.jpg"});
	classificationResult.print();

	//======================================== 输出处理 ===========================================

	//[1,138,138,32]
	const float* proto      = infer_request->GetBlob("1072")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
	
	//[1,19248,32]
	const float* maskdim    = infer_request->GetBlob("1270")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
	cv::Mat maskdim_mat(19248, 32, CV_32FC1, (void*)maskdim);
	
	// [1,19248,81]
	const float* confidence = infer_request->GetBlob("1271")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
	cv::Mat cof_mat(19248,81, CV_32FC1, (void*)confidence);

	
	//[1,19248,4]
	const float* location   = infer_request->GetBlob("1314")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
	cv::Mat loc_mat(19248, 4, CV_32FC1, (void*)location);
	
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;
	vector<int> maskIds;
	const int num_class = cof_mat.cols;

	for (int i = 0; i < this->num_priors; i++)
	{
		Mat scores = cof_mat.row(i).colRange(1, num_class);
		Point classIdPoint;
		double score;
		// Get the value and location of the maximum score
		minMaxLoc(scores, 0, &score, 0, &classIdPoint);
		if (score > this->_cof_threshold)
		{
			const float* loc = (float*)loc_mat.data + i * 4;
			const float* pb = this->priorbox + i * 4;
			float pb_cx = pb[0];
			float pb_cy = pb[1];
			float pb_w = pb[2];
			float pb_h = pb[3];

			float bbox_cx = var[0] * loc[0] * pb_w + pb_cx;
			float bbox_cy = var[1] * loc[1] * pb_h + pb_cy;
			float bbox_w = (float)(exp(var[2] * loc[2]) * pb_w);
			float bbox_h = (float)(exp(var[3] * loc[3]) * pb_h);
			float obj_x1 = bbox_cx - bbox_w * 0.5f;
			float obj_y1 = bbox_cy - bbox_h * 0.5f;
			float obj_x2 = bbox_cx + bbox_w * 0.5f;
			float obj_y2 = bbox_cy + bbox_h * 0.5f;

			// clip
			obj_x1 = max(min(obj_x1 * img_w, (float)(img_w - 1)), 0.f);
			obj_y1 = max(min(obj_y1 * img_h, (float)(img_h - 1)), 0.f);
			obj_x2 = max(min(obj_x2 * img_w, (float)(img_w - 1)), 0.f);
			obj_y2 = max(min(obj_y2 * img_h, (float)(img_h - 1)), 0.f);
			classIds.push_back(classIdPoint.x);
			confidences.push_back(score);
			boxes.push_back(Rect((int)obj_x1, (int)obj_y1, (int)(obj_x2 - obj_x1 + 1), (int)(obj_y2 - obj_y1 + 1)));
			maskIds.push_back(i);
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, this->_cof_threshold, this->_nms_area_threshold, indices, 1.f, this->_keep_top_k);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		int xmax = box.x + box.width;
		int ymax = box.y + box.height;
		rectangle(srcOriginal, Point(box.x, box.y), Point(xmax, ymax), Scalar(0, 0, 255), 3);
		//Get the label for the class name and its confidence
		char text[256];
		sprintf(text, "%s: %.2f", class_names[classIds[idx] + 1], confidences[idx]);


		//Display the label at the top of the bounding box
		int baseLine;
		Size labelSize = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
		int ymin = max(box.y, labelSize.height);
		//rectangle(frame, Point(left, top - int(1.5 * labelSize.height)), Point(left + int(1.5 * labelSize.width), top + baseLine), Scalar(0, 255, 0), FILLED);
		putText(srcOriginal, text, Point(box.x, ymin), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);

		Mat mask(this->mask_h, this->mask_w, CV_32FC1);
		mask = cv::Scalar(0.f);
		int channel = maskdim_mat.cols;
		int area = this->mask_h * this->mask_w;
		float* coeff = (float*)maskdim_mat.data + maskIds[idx] * channel;
		float* pm = (float*)mask.data;
		//const float* pmaskmap = (float*)outs[3].data;
		const float* pmaskmap = infer_request->GetBlob("1072")->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();
		for (int j = 0; j < area; j++)
		{
			for (int p = 0; p < channel; p++)
			{
				pm[j] += pmaskmap[p] * coeff[p];
			}
			pmaskmap += channel;
		}

		this->sigmoid(mask, area);
		Mat mask2;
		resize(mask, mask2, Size(img_w, img_h));
		// draw mask
		for (int y = 0; y < img_h; y++)
		{
			const float* pmask = (float*)mask2.data + y * img_w;
			uchar* p = srcOriginal.data + y * img_w * 3;
			for (int x = 0; x < img_w; x++)
			{
				if (pmask[x] > 0.5)
				{
					p[0] = (uchar)(p[0] * 0.5 + colors[classIds[idx] + 1][0] * 0.5);
					p[1] = (uchar)(p[1] * 0.5 + colors[classIds[idx] + 1][1] * 0.5);
					p[2] = (uchar)(p[2] * 0.5 + colors[classIds[idx] + 1][2] * 0.5);
				}
				p += 3;
			}
		}
	}

	//======================================== 输出处理 ===========================================

	return true;
}
