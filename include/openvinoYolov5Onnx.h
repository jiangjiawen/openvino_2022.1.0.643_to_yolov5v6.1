//
// Created by JiangJiawen on 2022/6/29.
//

#ifndef OPENVINOYOLONEWVERSION_OPENVINOYOLOV5ONNX_H
#define OPENVINOYOLONEWVERSION_OPENVINOYOLOV5ONNX_H

#include <vector>
#include "openvino/openvino.hpp"
#include <opencv2/opencv.hpp>

#include "slog.h"

using namespace ov::preprocess;

struct Detection
{
    int class_id;
    float confidence;
    cv::Rect box;
};

class openvinoYolov5Onnx{
public:
    explicit openvinoYolov5Onnx(const std::string &model_path, const int& labels_size_);
    ~openvinoYolov5Onnx();
    void initInputOutputInfo();
    static void printInputAndOutputsInfo(const ov::Model& network);
    void preprocessImage(const cv::Mat& oimg);
    static cv::Mat format_yolov5(const cv::Mat& source);
    std::vector<Detection> run();
private:
    ov::Core core;
    std::shared_ptr<ov::Model> model;
    PrePostProcessor *ppp;

    ov::CompiledModel compiled_model;
    ov::InferRequest infer_request;

    std::string input_tensor_name;
    std::string output_tensor_name;

    std::shared_ptr<unsigned char> input_data;

    int o_width = 2000;
    int o_height = 1500;

    double scale_w = 1.6;
    double scale_h = 1.6;

    int input_width = 1280;
    int input_height = 1280;

    const float SCORE_THRESHOLD = 0.2f;
    const float NMS_THRESHOLD = 0.35f;
    const float CONFIDENCE_THRESHOLD = 0.35f;

    int cols = 7;
    int rows = 102000;

    cv::Mat oImage;

    int labels_size = 80;
};

#endif //OPENVINOYOLONEWVERSION_OPENVINOYOLOV5ONNX_H
