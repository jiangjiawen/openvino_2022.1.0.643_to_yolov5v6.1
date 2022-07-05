#include <iostream>
#include <openvinoYolov5Onnx.h>
#include <chrono>

int main() {

    std::string model_path{"../model/yolov5s.onnx"};
    int labels_size = 80;
    std::chrono::steady_clock::time_point begin_load_model =
            std::chrono::steady_clock::now();

    openvinoYolov5Onnx openvinoYolov5OnnxTest(model_path, labels_size);

    std::chrono::steady_clock::time_point begin =
            std::chrono::steady_clock::now();

    std::cout << "load model Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(begin -
                                                                               begin_load_model)
                      .count()
              << " ms" << std::endl;

    cv::Mat img = cv::imread("../testImages/zidane.jpg");
    openvinoYolov5OnnxTest.preprocessImage(img);
    std::vector<Detection> outs = openvinoYolov5OnnxTest.run();

    const std::vector<cv::Scalar> colors = {cv::Scalar(255, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 0)};
    for (int i = 0; i < outs.size(); ++i)
    {
        auto detection = outs[i];
        auto box = detection.box;
        auto classId = detection.class_id;
        cv::Scalar color = colors[classId % colors.size()];

        cv::rectangle(img, box, color, 1);
        cv::putText(img, std::to_string(classId), cv::Point(box.x, box.y - 5), cv::FONT_HERSHEY_SIMPLEX, 2, cv::Scalar(114, 225, 10));
    }
    std::chrono::steady_clock::time_point end =
            std::chrono::steady_clock::now();
    std::cout << "Inference Latency: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end -
                                                                       begin)
                      .count()
              << " ms" << std::endl;
    cv::imshow("res", img);
    cv::waitKey(0);
    return 0;
}
