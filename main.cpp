// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "detector.h"

using namespace std;
using namespace cv;

int main(int argc, char *argv[]) {

	Detector* detector = new Detector;

	string xml_path_process_fp16 = "E:\\Dev\\Demo\\NCS2\\Models\\Yolact\\openvino2021\\fp16\\process\\yolact_resent101_54_fp16_process.xml";
	string bin_path_process_fp16 = "E:\\Dev\\Demo\\NCS2\\Models\\Yolact\\openvino2021\\fp16\\process\\yolact_resent101_54_fp16_process.bin";

	detector->init(xml_path_process_fp16, bin_path_process_fp16, 0.3, 0.5);

	Mat src = imread("E:\\Dev\\Demo\\NCS2\\Test\\yolact_ncs2_openvino_2021_4\\img\\test.jpg");
	vector<Detector::Object> detected_objects;
	detector->process_frame(src, detected_objects);

	static const string kWinName = "Deep learning object detection in OpenCV";
	namedWindow(kWinName, WINDOW_NORMAL);
	imshow(kWinName, src);
	waitKey(0);
	destroyAllWindows();

}