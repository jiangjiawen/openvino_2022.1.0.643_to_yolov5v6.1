//
// Created by JiangJiawen on 2022/6/29.
//

#include "openvinoYolov5Onnx.h"

openvinoYolov5Onnx::openvinoYolov5Onnx(const std::string& model_path, const int& labels_size_) {
    this->labels_size = labels_size_;
    model = core.read_model(model_path);
//    printInputAndOutputsInfo(*model);
    ppp = new PrePostProcessor(model);
    initInputOutputInfo();
    model = ppp->build();
    compiled_model = core.compile_model(model, "CPU");
    infer_request = compiled_model.create_infer_request();
}

openvinoYolov5Onnx::~openvinoYolov5Onnx() {
    delete ppp;
}

void openvinoYolov5Onnx::initInputOutputInfo() {
    input_tensor_name = model->input().get_any_name();
    const ov::Shape& input_shape = model->input().get_shape();
    input_height = input_shape[2];
    input_width = input_shape[3];

    output_tensor_name = model->output().get_any_name();
    const ov::Shape& out_shape = model->output().get_shape();
    rows = out_shape[1];
    cols = out_shape[2];

    InputInfo& input_info = ppp->input(input_tensor_name);

    const ov::Layout tensor_layout{"NHWC"};
    input_info.tensor()
            .set_element_type(ov::element::u8)
            .set_color_format(ColorFormat::BGR)
            .set_layout(tensor_layout)
            .set_spatial_static_shape(input_height, input_width);

    input_info.preprocess()
            .convert_element_type(ov::element::f32)
            .convert_color(ColorFormat::RGB)
            .scale(255);

    input_info.model().set_layout("NCHW");

    OutputInfo& output_info = ppp->output(output_tensor_name);
    output_info.tensor().set_element_type(ov::element::f32);
}

void openvinoYolov5Onnx::preprocessImage(const cv::Mat& oimg)
{
    this->o_width = oimg.cols;
    this->o_height = oimg.rows;

    //cv::Mat imageBGR = oimg.clone();
//    cv::Mat imageRGB;
//    cv::cvtColor(oimg, imageRGB, cv::COLOR_BGR2RGB);
    cv::Mat input_image = format_yolov5(oimg);

    this->scale_w = input_image.cols * 1.0 / input_width;
    this->scale_h = input_image.rows * 1.0 / input_height;

    size_t size = input_width * input_height * oimg.channels();
    input_data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());

    cv::Mat resized(cv::Size(input_width, input_height), input_image.type(), input_data.get());

    cv::resize(input_image, resized, cv::Size(input_width, input_height));
}

cv::Mat openvinoYolov5Onnx::format_yolov5(const cv::Mat& source)
{
    int col = source.cols;
    int row = source.rows;
    int _max = MAX(col, row);
    cv::Mat result = cv::Mat::zeros(_max, _max, CV_8UC3);
    source.copyTo(result(cv::Rect(0, 0, col, row)));
    return result;
}

void openvinoYolov5Onnx::printInputAndOutputsInfo(const ov::Model& network) {
    slog::info << "model name: " << network.get_friendly_name() << slog::endl;

    const std::vector<ov::Output<const ov::Node>> inputs = network.inputs();
    for (const ov::Output<const ov::Node>& input : inputs) {
        slog::info << "    inputs" << slog::endl;

        const std::string name = input.get_names().empty() ? "NONE" : input.get_any_name();
        slog::info << "        input name: " << name << slog::endl;

        const ov::element::Type type = input.get_element_type();
        slog::info << "        input type: " << type << slog::endl;

        const ov::Shape& shape = input.get_shape();
        slog::info << "        input shape: " << shape << slog::endl;
    }

    const std::vector<ov::Output<const ov::Node>> outputs = network.outputs();
    for (const ov::Output<const ov::Node>& output : outputs) {
        slog::info << "    outputs" << slog::endl;

        const std::string name = output.get_names().empty() ? "NONE" : output.get_any_name();
        slog::info << "        output name: " << name << slog::endl;

        const ov::element::Type type = output.get_element_type();
        slog::info << "        output type: " << type << slog::endl;

        const ov::Shape& shape = output.get_shape();
        slog::info << "        output shape: " << shape << slog::endl;
    }
}

std::vector<Detection> openvinoYolov5Onnx::run() {
    ov::element::Type input_type = ov::element::u8;
    ov::Shape input_shape = {1, static_cast<unsigned long long>(input_height), static_cast<unsigned long long>(input_width), 3};

    ov::Tensor input_tensor = ov::Tensor(input_type, input_shape, input_data.get());

    infer_request.set_tensor(input_tensor_name, input_tensor);
    infer_request.infer();

    ov::Tensor output_tensor = infer_request.get_tensor(output_tensor_name);

    const auto* outs = output_tensor.data<const float>();

    std::vector<std::vector<int>> class_ids(labels_size);
    std::vector<std::vector<float>> confidences(labels_size);
    std::vector<std::vector<cv::Rect>> boxes(labels_size);


    for (int i = 0; i < rows; ++i) {

        float confidence = outs[4];
        if (confidence >= CONFIDENCE_THRESHOLD) {
            float max_class_score = -INFINITY;
            int class_id = 0;

            for (int j = 5; j < cols; j++) {
                float temp_class_score = *(outs + j);
                if (temp_class_score> max_class_score) {
                    max_class_score = temp_class_score;
                    class_id = j - 5;
                }
            }

            if (max_class_score > SCORE_THRESHOLD) {
                float x = outs[0] + 1;
                float w = outs[2];
                float y = outs[1] + 1;
                float h = outs[3];
                int left = (std::max)(int((x - 0.5 * w) * scale_w), 0);
                int top = (std::max)(int((y - 0.5 * h) * scale_h), 0);

                int right = (std::min)(int(w * scale_w + left), this->o_width);
                int bottom = (std::min)(int(h * scale_h + top), this->o_height);

                int width = (right - left) > 0 ? (right - left) : 1;
                int height = (bottom - top) > 0 ? (bottom - top) : 1;

                boxes[class_id].emplace_back(cv::Rect(left, top, width, height));
                confidences[class_id].emplace_back(confidence);
                class_ids[class_id].emplace_back(class_id);

                // delete corner data
                //if (!( (left < 10) || (top < 10) || (right > this->o_width - 10) || (bottom > this->o_height - 10) ) ) {
                //    boxes.emplace_back(cv::Rect(left, top, width, height));
                //    confidences.emplace_back(confidence);
                //    class_ids.emplace_back(class_id);
                //}
            }

        }
        outs += cols;
    }

    std::vector<Detection> output;
    for(size_t i=0;i<boxes.size();i++){
        if(!(boxes[i].empty())) {
            std::vector<int> nms_result;
            cv::dnn::NMSBoxes(boxes[i], confidences[i], SCORE_THRESHOLD, NMS_THRESHOLD, nms_result);
            for (int idx: nms_result) {
                Detection result;
                result.class_id = class_ids[i][idx];
                result.confidence = confidences[i][idx];
                result.box = boxes[i][idx];
                output.push_back(result);
            }
        }
    }

    return output;
}
