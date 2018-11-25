#include <stdio.h>
#include <sys/shm.h>
#include <unistd.h>
#include <stdlib.h>

#include "network.h"
#include "mtcnn.h"
#include <math.h>
#include <time.h>
#include <thread>
#include <algorithm>
#include <curl/curl.h>

#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <termios.h>
#include <errno.h>
#include <string.h>

#include <netinet/in.h>
#include <netdb.h>
#include <fstream>
#include <vector>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/default_device.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/lib/io/path.h"

#include <dlib/gui_widgets.h>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>

#include "utils.h"
#include "serial_port_com.hpp"
#include "mqtt_client.hpp"
#include "public_function.hpp"

using namespace cv;
using namespace std;
//using namespace dlib;
#define DISP_WINNANE "camera"
#define QUIT_KEY     'q'
#define CAMID         0

using namespace tensorflow;
using tensorflow::Tensor;
using tensorflow::Status;
char ch;
std::string LABEL = "";
bool RECOG_RESULT = false;
std::vector<double> TESTSCORE;
int FACE_DETECT_NUM = 0;
int FACE_RECOGNITION_NUM = 0;
int BEHAVIOR_RECOGNITION_NUM = 0;
int OLD_NUM = 0;
cv::Mat SRC_FRAME;
time_t back_recog_start_time, back_recog_end_time;

struct face_info {
    cv::Mat face;
    float score = 100000.0;
    int num = 0;
    bool blink_sign = false;
};

struct msg {
    std::vector<int> SERIAL_PORT_WRITE_DATA_ID;
    std::vector<string> msgId;
    std::vector<string> content;
};

struct face_info FACE_INFO;
cv::Rect faceRect = cv::Rect(0,0,0,0);
float *RESULT = new float[512];//128
bool face_detect_function_exit_sign = false;
bool face_recognition_function_exit_sign = false;

//是否需要重置识别超时时间
bool NEED_RESET_TIME = false;
//检测识别线程是否可以运行
bool THREAD_CAN_RUNNING = true;
//下载图片URL
char* downloadImage_url;
//人脸遮挡是否已经提示
bool isDetectFace_voice_tip = false;
//语音播放标志
bool face_too_far_video_flag = false;
bool IS_SEND_BLOCKED_CAMERA = false;




void getImageTensor(tensorflow::Tensor &input_tensor, Mat& Image, int height, int width, int depth){
    if(Image.cols != width || Image.rows != height){
        resize(Image, Image, Size(width, height));
    }
    //int64_t start = cv::getTickCount();
    // cv::Mat Image = cv::imread(path);
    auto input_tensor_mapped = input_tensor.tensor<float, 4>();

    //mean and std
    //c * r * 3 => c * 3r * 1
    cv::Mat temp = Image.reshape(1, Image.rows * 3);

    cv::Mat mean3;
    cv::Mat stddev3;
    cv::meanStdDev(temp, mean3, stddev3);

    double mean_pxl = mean3.at<double>(0);
    double stddev_pxl = stddev3.at<double>(0);

    //prewhiten
    Image.convertTo(Image, CV_64FC3);
    Image = Image - cv::Vec3d(mean_pxl, mean_pxl, mean_pxl);
    Image = Image / stddev_pxl;

    // copying the data into the corresponding tensor
    for (int y = 0; y < height; ++y) {
        const double* source_row = Image.ptr<double>(y);
        for (int x = 0; x < width; ++x) {
            const double* source_pixel = source_row + (x * depth);
            for (int c = 0; c < depth; ++c) {
                const double* source_value = source_pixel + (2-c);//RGB->BGR
                input_tensor_mapped(0, y, x, c) = *source_value;
            }
        }
    }
    mean3.release();
    stddev3.release();
}


double Recogize(const std::unique_ptr<tensorflow::Session> &session, Tensor& tensor_face, float * res,int method = 1){

    std::vector<double> ret;

    Tensor phase_train(tensorflow::DT_BOOL, tensorflow::TensorShape());
    phase_train.scalar<bool>()() = false;

    std::vector<Tensor> outputs;
    string input_layer_1 = "input:0";
    string input_layer_2 = "phase_train:0";
    string output_layer = "embeddings:0";

    Status run_status = session->Run({{input_layer_1, tensor_face},
    							 {input_layer_2,phase_train }},
    							 {output_layer},
    							 {},
    							 &outputs);

    if(!run_status.ok()){
        LOG(ERROR) << "Running model failed"<<run_status;
        return 0.0;
    }
    auto outMap = outputs[0].tensor<float, 2>();
    if(res != NULL){
        for(int i = 0; i < 512; i++)
            res[i] = outMap(i);
    }else{
        double sum = 0;
        double dot = 0;
        double dot_1 = 0;
        double dot_2 = 0;
        if(method == 0){
            for(int i = 0; i < 512; i++) {
                sum += (outMap(i) - RESULT[i]) * (outMap(i) - RESULT[i]) ;
            }
            double norm = sqrt(dot_1)*sqrt(dot_2);
            double similarity = dot / norm;
            //TODO:返回距离 sqrt(sum)
            return sqrt(sum);
        }else {
            for(int i = 0; i < 512; i++) {
                dot += outMap(i)*RESULT[i];
                dot_1 += outMap(i)*outMap(i);
                dot_2 += RESULT[i]*RESULT[i];
            }
            double norm = sqrt(dot_1)*sqrt(dot_2);
            double similarity = dot / norm;
            //TODO:返回距离 sqrt(sum)
            return similarity;
        }
    }
    return 0.0;
}

void getSession(string graph_path, std::unique_ptr<tensorflow::Session> &session){
    tensorflow::GraphDef graph_def;
    if (!ReadEncryptedProto(tensorflow::Env::Default(), graph_path, &graph_def).ok()) {
    //ReadBinaryProto  ReadEncryptedProto
        LOG(ERROR) << "Read proto";
        return ;
    }
    tensorflow::SessionOptions sess_opt;
    // sess_opt.config.mutable_gpu_options()->set_allow_growth(true);

    (&session)->reset(tensorflow::NewSession(sess_opt));

    if (!session->Create(graph_def).ok()) {
        LOG(ERROR) << "Create graph";
        return ;
    }
}

void eyeCenters_to_eyeROIs(std::vector<cv::Point2i> &eyeCenters, std::vector<cv::Rect> &eyeROIs) {

    float alph = 0.3;
    float sub_x = eyeCenters[0].x - eyeCenters[1].x;
    float sub_y = eyeCenters[0].y - eyeCenters[1].y;
    float d = sqrt(sub_x*sub_x + sub_y*sub_y);

    float beta = d*alph;

    eyeROIs.clear();
    eyeROIs.resize(2);
    for (int i = 0; i < 2; i++) {
        int top_x = eyeCenters[i].x - beta;
        int top_y = eyeCenters[i].y - beta;
        int eye_width = beta*2;
        int eye_hight = eye_width;
        eyeROIs[i] = cv::Rect(top_x, top_y, eye_width, eye_hight);
    }

}

bool is_file_exist(string filepath)
{
    fstream _file;
    _file.open(filepath, ios::in);
    if (!_file)
    {
        return false;
    }
    return true;
}

bool init_eyelid_detect(string sp_eyelid_train_file, dlib::shape_predictor &pose_model_eyelid)
{
    if (!is_file_exist(sp_eyelid_train_file)) {
        return false;
    }
    dlib::deserialize(sp_eyelid_train_file) >> pose_model_eyelid;
    return true;
}

bool detect_eyelid(cv::Mat &tempROI_100k,
	               dlib::shape_predictor &pose_model_eyelid,
                   std::vector<cv::Point2f> &eyelid_100k)
{
	dlib::cv_image<dlib::bgr_pixel> cimg_d(tempROI_100k);
	//opencv的图像Mat需转换为dlib所用格式
	dlib::rectangle face_p(0, 0, tempROI_100k.size().width, tempROI_100k.size().height);
	dlib::full_object_detection shape_eyelid_temp = pose_model_eyelid(cimg_d, face_p);
	//标定眼部特征点
	bool Sign_eyelid_d = false;

	if (shape_eyelid_temp.num_parts() == 7) {
	//眼部特征点检测成功
		bool Sign_shape_error = false;
		for (int i = 0; i < shape_eyelid_temp.num_parts(); i++) {
			if (shape_eyelid_temp.part(i).x() < 0 || shape_eyelid_temp.part(i).y() < 0) {
			//确保检测到的shape点无错误
				Sign_shape_error = true;
				break;
			}
		}
		if (Sign_shape_error == false) {
			Sign_eyelid_d = true;
		}
	}

	if (Sign_eyelid_d == true) {
	//如果眼部检测成功

		for (int i = 0; i < shape_eyelid_temp.num_parts() - 1; i++) {
			eyelid_100k.push_back(cv::Point2f(shape_eyelid_temp.part(i).x(), shape_eyelid_temp.part(i).y()));
		}

		return true;
	}
	return false;
}

bool blink_eye(cv::Mat &SRC_FRAME,
               std::vector<cv::Rect> &eyeRects,
               dlib::shape_predictor &pose_model_eyelid,
               std::vector<float> &closed_eye_coef_left,
               std::vector<float> &closed_eye_coef_right)
{
    bool ret = false;
    for(int i=0; i<eyeRects.size(); i++) {
        if (eyeRects[i].x < 0 ||eyeRects[i].y < 0 ||
            eyeRects[i].x+eyeRects[i].width > SRC_FRAME.size().width || eyeRects[i].y+eyeRects[i].height > SRC_FRAME.size().height) {
            continue;
        }
        cv::Mat tempROI = SRC_FRAME(eyeRects[i]);
        cv::Mat tempROI_100k;//100K的眼部ROI区域
        float zoom_rate = (float)tempROI.size().area() / 100000.0;//眼部ROI区域归一化为100K做检测，此值可影响效果
        float scalingFactor_temp;//将眼部ROI区域像素值调整为100K大小的缩放因子
        if (zoom_rate < 1.0) {
            scalingFactor_temp = std::sqrt(1 / zoom_rate);
            cv::resize(tempROI, tempROI_100k, cv::Size(), scalingFactor_temp, scalingFactor_temp, cv::INTER_LINEAR);//INTER_CUBIC
        }
        else {
            scalingFactor_temp = std::sqrt(1 / zoom_rate);
            cv::resize(tempROI, tempROI_100k, cv::Size(), scalingFactor_temp, scalingFactor_temp, cv::INTER_AREA);//INTER_CUBIC
        }

        cv::Point2f center, center_100k;
        float radius, radius_100k;
        std::vector<cv::Point2f> shape_eyelid, shape_eyelid_100k;//6 points
        bool Sign_detect_eyelid_and_pupile = detect_eyelid(tempROI_100k, pose_model_eyelid, shape_eyelid_100k);

        if (Sign_detect_eyelid_and_pupile) {//如果眼球与眼皮都检测成功
            cv::Point2f eyelid_top_100k, eyelid_bottom_100k;
            eyelid_top_100k.x = (shape_eyelid_100k[1].x + shape_eyelid_100k[2].x)/2;
            eyelid_top_100k.x = (shape_eyelid_100k[1].y + shape_eyelid_100k[2].y)/2;
            eyelid_bottom_100k.x = (shape_eyelid_100k[4].x + shape_eyelid_100k[5].x)/2;
            eyelid_bottom_100k.x = (shape_eyelid_100k[4].y + shape_eyelid_100k[5].y)/2;
            float dx = (shape_eyelid_100k[0].x-shape_eyelid_100k[3].x)*(shape_eyelid_100k[0].x-shape_eyelid_100k[3].x) +
                       (shape_eyelid_100k[0].y-shape_eyelid_100k[3].y)*(shape_eyelid_100k[0].y-shape_eyelid_100k[3].y);
            float dy = (eyelid_top_100k.x-eyelid_bottom_100k.x)*(eyelid_top_100k.x-eyelid_bottom_100k.x) +
                       (eyelid_top_100k.y-eyelid_bottom_100k.y)*(eyelid_top_100k.y-eyelid_bottom_100k.y);
            if (i == 0) {
                closed_eye_coef_left.push_back(dy/dx);
            } else {
                closed_eye_coef_right.push_back(dy/dx);
            }

        }
        tempROI.release();
        tempROI_100k.release();
    }
    if (closed_eye_coef_left.size() >= 3) {//检测眨眼次数
       auto max_left = max_element(closed_eye_coef_left.begin(), closed_eye_coef_left.end());
       auto min_left = min_element(closed_eye_coef_left.begin(), closed_eye_coef_left.end());
       auto max_right = max_element(closed_eye_coef_right.begin(), closed_eye_coef_right.end());
       auto min_right = min_element(closed_eye_coef_right.begin(), closed_eye_coef_right.end());
       if ((*min_left)/(*max_left) < 0.7 || (*min_right)/(*max_right) < 0.7) {//闭眼指数阈值
            ret = true;
       } else {
            ret = false;
       }
       closed_eye_coef_left.clear();
       closed_eye_coef_right.clear();
    }
    return ret;
}

void behavior_recognition_function(int num, int br_height, int br_width, int br_depth, string &graph_path, string &labels_path)
{
    // Set dirs variables
 //   string labels_path = "../model/smoking_detect/my_labels_map.pbtxt";
 //   string graph_path = "../model/smoking_detect/encrypted_inference_graph.pb";
  //  int64_t start;

    // Set input & output nodes names
    string inputLayer = "image_tensor:0";
    std::vector<string> outputLayer = {"detection_boxes:0", "detection_scores:0", "detection_classes:0", "num_detections:0"};

    // Load and initialize the model from .pb file
    std::unique_ptr<tensorflow::Session> session;
    //string graphPath = tensorflow::io::JoinPath(ROOTDIR, GRAPH);
    //LOG(INFO) << "graphPath:" << graphPath;
    Status loadGraphStatus = loadGraph(graph_path, &session);
    if (!loadGraphStatus.ok()) {
        LOG(ERROR) << "loadGraph(): ERROR" << loadGraphStatus;
        return;
    } else
        LOG(INFO) << "loadGraph(): frozen graph loaded" << endl;


    // Load labels map from .pbtxt file
    std::map<int, std::string> labelsMap = std::map<int,std::string>();
    Status readLabelsMapStatus = readLabelsMapFile(labels_path, labelsMap);
    if (!readLabelsMapStatus.ok()) {
        LOG(ERROR) << "readLabelsMapFile(): ERROR" << loadGraphStatus;
        return;
    } else
        LOG(INFO) << "readLabelsMapFile(): labels map loaded with " << labelsMap.size() << " label(s)" << endl;

//    Tensor tensor_b(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, br_height, br_width, br_depth}));
    Tensor tensor_b;
    tensorflow::TensorShape shape = tensorflow::TensorShape();
    shape.AddDim(1);
    shape.AddDim(br_height);
    shape.AddDim(br_width);
    shape.AddDim(br_depth);

    std::vector<Tensor> outputs;
    double confThreshold = 0.95;
    double thresholdIOU = 0.8;

    while(true) {
        usleep(500);
        if (!SRC_FRAME.empty()) {
         //   start = getTickCount();

            cv::resize(SRC_FRAME, SRC_FRAME, cv::Size(br_width, br_height));
            // Convert mat to tensor
            cvtColor(SRC_FRAME, SRC_FRAME, COLOR_BGR2RGB);
            tensor_b = Tensor(tensorflow::DT_FLOAT, shape);
            Status readTensorStatus = readTensorFromMat(SRC_FRAME, tensor_b);
            if (!readTensorStatus.ok()) {
                LOG(ERROR) << "Mat->Tensor conversion failed: " << readTensorStatus;
                break;
            }

            // Run the graph on tensor
            outputs.clear();
            Status runStatus = session->Run({{inputLayer, tensor_b}}, outputLayer, {}, &outputs);
            if (!runStatus.ok()) {
                LOG(ERROR) << "Running model failed: " << runStatus;
                break;
            }
            cvtColor(SRC_FRAME, SRC_FRAME, COLOR_RGB2BGR);

            // Extract results from the outputs vector
            tensorflow::TTypes<float>::Flat scores = outputs[1].flat<float>();
            tensorflow::TTypes<float>::Flat classes = outputs[2].flat<float>();
            tensorflow::TTypes<float>::Flat numDetections = outputs[3].flat<float>();
            tensorflow::TTypes<float, 3>::Tensor boxes = outputs[0].flat_outer_dims<float,3>();

            std::vector<size_t> goodIdxs = filterBoxes(scores, boxes, thresholdIOU, confThreshold);
           /*     for (size_t i = 0; i < goodIdxs.size(); i++)
                    LOG(INFO) << "score:" << scores(goodIdxs.at(i)) << ",class:" << labelsMap[classes(goodIdxs.at(i))]
                              << " (" << classes(goodIdxs.at(i)) << "), box:" << "," << boxes(0, goodIdxs.at(i), 0) << ","
                              << boxes(0, goodIdxs.at(i), 1) << "," << boxes(0, goodIdxs.at(i), 2) << ","
                              << boxes(0, goodIdxs.at(i), 3);
          */

            for (int j = 0; j < goodIdxs.size(); j++) {
                BEHAVIOR_RECOGNITION_NUM++;
                //Rect2f Box = Rect2f(Point2f(boxes(0, goodIdxs.at(j), 1)*video_width, boxes(0, goodIdxs.at(j), 0)*video_height),
                //                         Point2f(boxes(0, goodIdxs.at(j), 3)*video_width, boxes(0, goodIdxs.at(j), 2)*video_height));

                //cv::circle(frame, Point2f(Box.x + Box.width/2, Box.y + Box.height/2), 2, cv::Scalar(0, 255, 0), 1);

                LABEL = labelsMap[classes(goodIdxs.at(0))];
              //  cout << to_string(num) + ": " + LABEL + "   " + to_string(BEHAVIOR_RECOGNITION_NUM) << endl;
            }

        //    cout<<"The whole process of behavior recognition " << to_string(num) << " cost "<<1000 * (double)(getTickCount()-start)/getTickFrequency()<<" ms"<<endl;

            // Draw bboxes and captions
            //drawBoundingBoxesOnImage(frame, scores, classes, boxes, labelsMap, goodIdxs);

        }
        if (ch == 27 || ch == 'q') {
            break;
        }

    }
    std::cout << "behavior recognition over!" << std::endl;

}

void face_recognition_function(int num)
{
    string graph_path,image_path,network_height_str,network_width_str,network_depth_str;
    get_value(GRAP_PATH_KEY,graph_path);
    get_value(SAVE_IMAGE_PATH_KEY,image_path);
    get_value(NETWORK_HEIGHT_KEY,network_height_str);
    get_value(NETWORK_WIDTH_KEY,network_width_str);
    get_value(NETWORK_DEPTH_KEY,network_depth_str);
    int network_height = atoi(network_height_str.data());
    int network_width = atoi(network_width_str.data());
    int network_depth = atoi(network_depth_str.data());
   // cout << graph_path << endl;
    cv::Mat image_a = imread(image_path);
    if (image_a.empty()) {
        std::cerr << "failed to open image !" << std::endl;
        image_a = cv::Mat::zeros(network_height,network_width,CV_8UC3);
    }

    Tensor tensor_a_1(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, network_height, network_width, network_depth}));
    Tensor tensor_a_2(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, network_height, network_width, network_depth}));
    unique_ptr<tensorflow::Session> session;
    //facenet model initialize
    getSession(graph_path, session);
    //project the cv::Mat into tensorflow::Tensor
    getImageTensor(tensor_a_1, image_a, network_height, network_width, network_depth);
    image_a.release();
    //for model test
    Recogize(session, tensor_a_1, RESULT);
    string min_recog_match_str,middle_recog_match_str,max_recog_match_str;
    get_value(MIN_RECOG_MATCH_KEY,min_recog_match_str);
    get_value(MIDDLE_RECOG_MATCH_KEY,middle_recog_match_str);
    get_value(MAX_RECOG_MATCH_KEY,max_recog_match_str);

    double min_recog_match = atof(min_recog_match_str.data());
    double middle_recog_match = atof(middle_recog_match_str.data());
    double max_recog_match = atof(max_recog_match_str.data());
    //facenet model initialize
    getSession(graph_path, session);
    time(&back_recog_start_time);
  //  usleep(500000);
    while(THREAD_CAN_RUNNING) {
        if(!data_ptr->DETECT_STATUS && data_ptr->BACK_RECOG_FLAG){
            time(&back_recog_end_time);
            if(difftime(back_recog_end_time, back_recog_start_time) >= (data_ptr->BACK_RECOG_TIME * 60)) {
                time(&back_recog_start_time);
            } else {
                continue;
            }
        }

        cv::Mat face = FACE_INFO.face.clone();

        if (!face.empty() && FACE_INFO.num > OLD_NUM && FACE_INFO.blink_sign && data_ptr->DETECT_STATUS) {
        //   cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
            FACE_RECOGNITION_NUM++;
            OLD_NUM = FACE_INFO.num;
          //  start = getTickCount();

            getImageTensor(tensor_a_2, face, network_height, network_width, network_depth);
            double r = Recogize(session, tensor_a_2, NULL,1);
            cout << r << endl;
            if (r >= max_recog_match) {
         //       cout << to_string(num) + ": " + testName + " (" + to_string(r[1]).substr(0, 5) + "   " + to_string(r[0]).substr(0, 5) + ")  " + to_string(FACE_RECOGNITION_NUM) << endl;
         //       TESTNAME = testName + " (" + to_string(r[1]).substr(0, 4) + ")  " + to_string(FACE_RECOGNITION_NUM);

                TESTSCORE.push_back(r);
                RECOG_RESULT = true;
                data_ptr->DETECT_STATUS = false;

            }
            else if (r >= middle_recog_match) {
         //       cout << to_string(num) + ": other (" + to_string(r[1]).subs//TODO：神经网络所需图片 宽度 、高度、 深度;tr(0, 5) + "   " + to_string(r[0]).substr(0, 5) + ")  " + to_string(FACE_RECOGNITION_NUM) << endl;
         //       TESTNAME = "other (" + to_string(r[1]).substr(0, 4) + ")  " + to_string(FACE_RECOGNITION_NUM);
                TESTSCORE.push_back(r);
                if (TESTSCORE.size() >= 2) {
                    RECOG_RESULT = true;
                    data_ptr->DETECT_STATUS = false;
                }
            }
            else if (r >= min_recog_match) {
                TESTSCORE.push_back(r);
                if (TESTSCORE.size() >= 3) {
                    RECOG_RESULT = true;
                    data_ptr->DETECT_STATUS = false;
                }
            }


        //    cout<<"The whole process of face recognition cost "<<1000 * (double)(getTickCount()-start)/getTickFrequency()<<" ms"<<endl;
        }
        face.release();
        usleep(10000);
    }
    std::cout << num << ": face recognition over!" << std::endl;
    face_recognition_function_exit_sign = true;
}

void face_detect_function(cv::Mat &frame,
                          int video_width,
                          int video_height,
                          std::vector<float> &closed_eye_coef_left,
                          std::vector<float> &closed_eye_coef_right,
                          bool &is_blocked_camera)
{
    string grap_path;
    string network_height_str;
    string network_width_str;
    string network_depth_str;
    string sp_eyelid_train_file;
    get_value(NETWORK_HEIGHT_KEY,network_height_str);
    get_value(NETWORK_WIDTH_KEY,network_width_str);
    get_value(NETWORK_DEPTH_KEY,network_depth_str);
    get_value(GRAP_PATH_KEY,grap_path);
    int network_width = atoi(network_width_str.data());
    int network_height = atoi(network_height_str.data());
    int network_depth = atoi(network_depth_str.data());
    Tensor tensor_a_1(tensorflow::DT_FLOAT, tensorflow::TensorShape({1, network_height, network_width, network_depth}));
    unique_ptr<tensorflow::Session> session;
    getSession(grap_path, session);

    mtcnn find1(video_width / 2, video_height / 2);
    get_value(EYELID_TRAIN_FILE_KEY,sp_eyelid_train_file);
    dlib::shape_predictor pose_model_eyelid;

    bool init_eyelid_detect_sign = init_eyelid_detect(sp_eyelid_train_file, pose_model_eyelid);
    if (!init_eyelid_detect_sign) {
        std::cerr << "failed to initialize eyelid detect !" << std::endl;
        return;
    }
    time(&back_recog_start_time);

    while(THREAD_CAN_RUNNING) {

        if(data_ptr->RESET_SRC_TENSOR){
            string path;
            get_value(SAVE_IMAGE_PATH_KEY,path);
            cout << "path ----------" << path << endl;
            cv::Mat image_a = imread(path);
            if (image_a.empty() || image_a.cols < network_height || image_a.rows < network_width) {
                std::cerr << "failed to open image !" << std::endl;
            }else{
                //project the cv::Mat into tensorflow::Tensor
                getImageTensor(tensor_a_1, image_a, network_height, network_width, network_depth);
                image_a.release();
                //for model test
                Recogize(session, tensor_a_1, RESULT);
            }
            data_ptr->RESET_SRC_TENSOR = false;
        }
        if(!data_ptr->DETECT_STATUS && data_ptr->BACK_RECOG_FLAG){
            time(&back_recog_end_time);
            if(difftime(back_recog_end_time, back_recog_start_time) >= (data_ptr->BACK_RECOG_TIME * 60)){
                time(&back_recog_start_time);
                data_ptr->RECOG_START_SIGN = true;
            }
            sleep(3);
            continue;
        }

        if (!frame.empty() && data_ptr->DETECT_STATUS && !is_blocked_camera) {
            find1.findFace(frame);
         //   cout << "................2" << endl;
            //etick = cv::getCPUTickCount();
            //std::cout<<"faceDetect cost: "<<(etick - ftick)/ticksPerUs / 1000<<" ms"<<std::endl;

            int old_width = 0;
            cv::Rect faceRect = cv::Rect(0,0,0,0);
            std::vector<cv::Point2i> eyeCenters;
            float two_eye_dis = 0.0;
            for(std::vector<struct Bbox>::iterator it=find1.thirdBbox_.begin(); it!=find1.thirdBbox_.end();it++) {
                if((*it).exist) {
                    int new_width = (*it).x2-(*it).x1;
		//    cout << (*it).y1 << "  " << (*it).x1 << "  " <<  (*it).y2-(*it).y1 << "  " << (*it).x2-(*it).x1 << endl;
                    if (new_width > old_width) {
                        old_width = new_width;
                        faceRect = cv::Rect((*it).y1, (*it).x1, (*it).y2-(*it).y1, (*it).x2-(*it).x1);
                        faceRect.x = faceRect.x + faceRect.width*0.1;
                        faceRect.width = faceRect.width - faceRect.width*2*0.1;
		//	cout << "......" << faceRect.x << " " << faceRect.y << " " << faceRect.width << " " << faceRect.height << endl;
                        eyeCenters.clear();
                        eyeCenters.resize(2);
                        for (int num=0; num<2; num++) {
                            eyeCenters[num] = cv::Point2i((int)*(it->ppoint+num), (int)*(it->ppoint+num+5));
                        }
                        two_eye_dis = (eyeCenters[0].x-eyeCenters[1].x)*(eyeCenters[0].x-eyeCenters[1].x) +
                                        (eyeCenters[0].y-eyeCenters[1].y)*(eyeCenters[0].y-eyeCenters[1].y);
                    }
                }
            }
            find1.thirdBbox_.clear();
            string min_eye_space_str;
            get_value(MIN_EYE_SPACE_KEY,min_eye_space_str);
            int min_eye_space = atoi(min_eye_space_str.data());
            if (two_eye_dis >= min_eye_space) {
                FACE_DETECT_NUM++;
                if (!FACE_INFO.blink_sign) {
                    std::vector<cv::Rect> eyeRects;
                    eyeCenters_to_eyeROIs(eyeCenters, eyeRects);
                    FACE_INFO.blink_sign = blink_eye(frame, eyeRects, pose_model_eyelid, closed_eye_coef_left, closed_eye_coef_right);
                }
                FACE_INFO.face = frame(faceRect).clone();
                FACE_INFO.num++;
             //   cv::imwrite("../images/"+to_string(opt1.num)+".jpg", opt1.face);

             //   cout << ".................opt1.blink_sign = " << opt1.blink_sign  << endl;
            } else {
                 int video_id = 0;
                 if (two_eye_dis == 0.0) {
                    video_id = 4;
                    std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
                    thread_play_face_video.detach();
                 } else {
                    if(!face_too_far_video_flag){
                        video_id = 5;
                        std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
                        thread_play_face_video.detach();
                    }
                    cout << "two_eye_dis = " << two_eye_dis << endl;
                 }
            }
        }
        usleep(100000);
    }
    find1.findFace(frame);
    std::cout << "face detect over!" << std::endl;
    face_detect_function_exit_sign = true;
}


void init_params(string &serial_port_path_str){
    //参数初始化
        if(FindKey(RECOG_OUT_TIME_KEY)){
            string recog_out_time_str;
            get_value(RECOG_OUT_TIME_KEY,recog_out_time_str);
            data_ptr->RECOG_OUT_TIME = atoi(recog_out_time_str.data());
        }
        if(FindKey(USER_AND_PASSWORD_KEY)){
            get_value(USER_AND_PASSWORD_KEY,username_password_str);
        }
        if(FindKey(BACK_RECOG_FLAG_KEY)){
            string back_recog_flag_str;
            get_value(BACK_RECOG_FLAG_KEY,back_recog_flag_str);
            data_ptr->BACK_RECOG_FLAG = atoi(back_recog_flag_str.data());
        }
        if(FindKey(BACK_RECOG_TIME_KEY)){
            string back_recog_time_str;
            get_value(BACK_RECOG_TIME_KEY,back_recog_time_str);
            data_ptr->BACK_RECOG_TIME = atoi(back_recog_time_str.data());
        }
        if(FindKey(MODEL_CONNECT_SERVER_KEY)){
            get_value(MODEL_CONNECT_SERVER_KEY,model_connect_url_str);
        }
        if(FindKey(UPLOAD_IMAGE_SERVER_KEY)){
            get_value(UPLOAD_IMAGE_SERVER_KEY,upload_url);
        }
        if(FindKey(SERIAL_PORT_SPEED_KEY)){
            string serial_port_speed_str;
            get_value(SERIAL_PORT_SPEED_KEY,serial_port_speed_str);
            data_ptr->SERIAL_PORT_SPEED_NUMBER = atoi(serial_port_speed_str.data());
        }
        if(FindKey(SERIAL_PORT_PATH_KEY)){
            get_value(SERIAL_PORT_PATH_KEY,serial_port_path_str);
        }
        if(FindKey(MODEL_SOUND_KEY)){
            string model_sound_str;
            get_value(MODEL_SOUND_KEY,model_sound_str);
            string src = "amixer set 'Master' ";
            string sound = src + model_sound_str;
            int result = system(sound.data());
            if(result == 0){
                cout << "set volume success" << endl;
            }else{
                cout << "set volume failed" << endl;
            }
        }
}



int main()
{
    bool read_status = read_config(MQTT_CONFIG_PARAMS_FILE);
    bool read_xml_status = ReadXml(CONFIG_FILE_PAH);
    if(!read_status || !read_xml_status ){
        std::cerr << "failed to read config file !" << std::endl;
        return -1;
    }
    /**
        if(!FindKey(SERIAL_PORT_PATH) || !FindKey(SERIAL_PORT_SPEED) || !FindKey(SAVE_IMAGE_PATH)){
        if(!FindKey(SERIAL_PORT_PATH)){
            read_data src;
            src.key = SERIAL_PORT_PATH;
            src.value = "/dev/ttyFIQ0";
            params_list.push_back(src);
        }
        if(!FindKey(SERIAL_PORT_SPEED)){
            read_data src;
            src.key = SERIAL_PORT_SPEED;
            src.value = "115200";
            params_list.push_back(src);
        }
        if(!FindKey(SAVE_IMAGE_PATH)){
            read_data src;
            src.key = SAVE_IMAGE_PATH;
            src.value = "/home/firefly/face_image/face.jpg";
            params_list.push_back(src);
        }
        }
    */
	int shmid = shmget(IPC_PRIVATE, sizeof(struct shared_data), 0664 | IPC_CREAT);
    pid_t pid = fork();
    if (pid < 0){
        printf("creat process error !\n");
        return -4;
    }
	else if(pid == 0)
	{
        printf("fork success, this is son process\n");
		data_ptr = (struct shared_data*) shmat(shmid, 0, 0);
        data_ptr->SERIAL_PORT_WRITE_SIGN = false;
        data_ptr->SERIAL_PORT_WRITE_DATA_ID = -1;
        data_ptr->RECOG_OUT_TIME = 15;//默认识别时间 秒
        data_ptr->SERIALPORT_RECEIVE_DATA_CAN_RUNNING = true;
        data_ptr->SERIAL_PORT_EXIT_STATUS = false;
        data_ptr->RECOG_START_SIGN = false;
        data_ptr->DETECT_STATUS = false;
        data_ptr->RECOG_REASON = 0;
        data_ptr->RECOG_RESULT = true;
        //初始化参数
        string serial_port_path_str;
        init_params(serial_port_path_str);


        std::thread thread_socket(open_mqtt_thread);
        thread_socket.detach();

		int fd = SerialInit(serial_port_path_str.data(), data_ptr);
        if (fd > 0) {
            receive_SerialPortData_function(fd, data_ptr);//接收串口数据函数为阻塞性
        }

        data_ptr->SERIAL_PORT_EXIT_STATUS = true;
        while(!data_ptr->MQTT_EXIT_SIGN){
            if(data_ptr->EXIT_PROGRAM){
                mosquitto_disconnect(mosq);
            }
            usleep(100000);
        }
        WriteXml(CONFIG_FILE_PAH);
        cout << "communication process exit ! " << endl;
        exit(0);
	}
	sleep(3);
    data_ptr = (struct shared_data*) shmat(shmid, 0, 0);

    time_t start1, end1;
    std::vector<float> closed_eye_coef_left, closed_eye_coef_right;
    // Create a window
    std::string kWinName = "face";
    cv::namedWindow(kWinName, WINDOW_NORMAL);

    //开启摄像头
    cv::VideoCapture cap(CAMID);
    cap.set(CAP_PROP_FRAME_WIDTH, 1280);//320
    cap.set(CAP_PROP_FRAME_HEIGHT, 720);//240
    int video_height = cap.get(CAP_PROP_FRAME_HEIGHT);
    int video_width = cap.get(CAP_PROP_FRAME_WIDTH);
 //   static int br_height = 480;
 //   static int br_width = 640;
 //   static int br_depth = 3;
    cout << video_width << "   " << video_height << endl;
    if (!cap.isOpened()) {
        std::cerr << "failed to open camera !" << std::endl;
        return -3;
    }
    int num_fr_1 = 1;//, num_fr_2 = 2, num_fr_3 = 3;
//    int num_1 = 1, num_2 = 2, num_3 = 3;
    bool is_blocked_camera = false;//遮挡摄像头
    std::thread thread_face_recognition1(face_recognition_function, std::ref(num_fr_1));
    std::thread thread_face_detect(face_detect_function,std::ref(SRC_FRAME), std::ref(video_width), std::ref(video_height), std::ref(closed_eye_coef_left), std::ref(closed_eye_coef_right),std::ref(is_blocked_camera));

    thread_face_recognition1.detach();
    thread_face_detect.detach();

    TESTSCORE.clear();

    static string rec_ret = "";
    cv::Mat src_frame_gray;//遮挡灰度图
    cv::Mat mean1, stddev1;//计算遮挡使用
    double stddev_pxl=0.0;
    double recogscore = 0.0;
    //static string blink_sign = "";

    time(&start1);
    while (true) {
        cap >> SRC_FRAME;
        if (data_ptr->RECOG_START_SIGN) {
            data_ptr->DETECT_STATUS = true;
            NEED_RESET_TIME = true;
            FACE_INFO.blink_sign = false;
            FACE_RECOGNITION_NUM = 0;
            data_ptr->RECOG_START_SIGN = false;
            int video_id = 0;
            std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
            thread_play_face_video.detach();
        }

        if(NEED_RESET_TIME){
            time(&start1);
            NEED_RESET_TIME = false;
        }
        time(&end1);
        if(difftime(end1, start1) >= data_ptr->RECOG_OUT_TIME){//识别超时
             if(data_ptr->DETECT_STATUS){
                if(stddev_pxl < 30){//存在遮挡
                    //RECEIVE_REG_FLAG = false;
                    //printf("application_bool = %d\n",RECEIVE_REG_FLAG);
                    data_ptr->SERIAL_PORT_WRITE_SIGN = true;
                    data_ptr->SERIAL_PORT_WRITE_DATA_ID = 3;
                    data_ptr->RECOG_RESULT = false;
		            data_ptr->RECOG_REASON = 1;
                }else if (FACE_RECOGNITION_NUM == 0) {//未检测到人脸
                    //RECEIVE_REG_FLAG = false;
                    data_ptr->SERIAL_PORT_WRITE_SIGN = true;
		            data_ptr->SERIAL_PORT_WRITE_DATA_ID = 1;
		            data_ptr->RECOG_RESULT = false;
		            data_ptr->RECOG_REASON = 2;
		            int video_id = 2;
		            std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
                    thread_play_face_video.detach();
                }else{//人脸不匹配
                    rec_ret = " other";
                    data_ptr->SERIAL_PORT_WRITE_SIGN = true;
		            data_ptr->SERIAL_PORT_WRITE_DATA_ID = 2;
		            data_ptr->RECOG_RESULT = false;
		            data_ptr->RECOG_REASON = 3;
		            int video_id = 3;
		            std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
                    thread_play_face_video.detach();
                    string another_path;
                    get_value(MISMATCH_FACE_IMAGE_KEY,another_path);
		            cv::imwrite(another_path, SRC_FRAME);
                }
                data_ptr->DETECT_STATUS = false;
                face_too_far_video_flag = false;
                IS_SEND_BLOCKED_CAMERA = false;//超时将播放遮挡语音标志恢复初始化状态
             }
        }else {//识别未超时
             //判断摄像头是否遮挡
            cv::cvtColor(SRC_FRAME, src_frame_gray, cv::COLOR_RGB2GRAY);//转灰度图
            cv::meanStdDev(src_frame_gray, mean1, stddev1);
            stddev_pxl = stddev1.at<double>(0);
            if (stddev_pxl < 30.0) {//check if camera is blocked
                is_blocked_camera = true;
                int video_id = 6;
                std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
                thread_play_face_video.detach();
                if(!IS_SEND_BLOCKED_CAMERA){
                    IS_SEND_BLOCKED_CAMERA = true;
                }
            } else {
                is_blocked_camera = false;
                if (!data_ptr->DETECT_STATUS && RECOG_RESULT){
                    /**
                    for (int i = 0; i < TESTSCORE.size(); i++) {
                        recogscore += TESTSCORE[i];
                    }
                    recogscore = recogscore/TESTSCORE.size();
                    rec_ret = " me (" + to_string(recogscore).substr(0, 5) + ")  ";
                    recogscore = 0.0;

                    */
                    TESTSCORE.clear();
                    RECOG_RESULT = false;
                    data_ptr->RECOG_RESULT = true;
                    face_too_far_video_flag = false;
                    data_ptr->SERIAL_PORT_WRITE_SIGN = true;
		            data_ptr->SERIAL_PORT_WRITE_DATA_ID = 0;
		            data_ptr->RECOG_REASON = 0;
                    int video_id = 1;
                    IS_SEND_BLOCKED_CAMERA = false;
                    std::thread thread_play_face_video(play_face_reg_video,std::ref(video_id),std::ref(face_too_far_video_flag));
                    thread_play_face_video.detach();
                    string save_path;
                    get_value(MATCH_FACE_IMAGE_KEY,save_path);
                    cv::imwrite(save_path.data(), SRC_FRAME);
                }
            }
        }
        /**
        if (FACE_INFO.blink_sign) {
            blink_sign = "ture";
        } else {
            blink_sign = "false";
        }
        */
        //cout << difftime(end1, start1) << endl;

    //    cv::putText(src_frame, "Blink: " + blink_sign, Point(0, ini_gap), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 8);
    //    cv::putText(src_frame, "Face Det: " + to_string(FACE_DETECT_NUM), Point(0, ini_gap+gap), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2, 8);
    //    cv::putText(src_frame, "Face Rec: " + to_string(FACE_RECOGNITION_NUM) + rec_ret, Point(0, ini_gap+2*gap), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,23,0), 2, 8);
    //    cv::putText(src_frame, "Action Rec: " + LABEL + "  " + to_string(BEHAVIOR_RECOGNITION_NUM), Point(0, ini_gap+3*gap), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,23,0), 2, 8);
    //    cv::putText(src_frame, "Camera : " + camera_status, Point(0, ini_gap+4*gap), FONT_HERSHEY_SIMPLEX, 1, Scalar(255,23,0), 2, 8);
        // cv::circle(src_frame, cv::Point(video_width/2, video_height/2), 3, cv::Scalar(0, 0, 255), -1);

        /*std::cout << "Blink: " + blink_sign << std::endl;
        std::cout << "Face Det: " + to_string(FACE_DETECT_NUM) << std::endl;
        std::cout << "Face Rec: " + to_string(FACE_RECOGNITION_NUM) + rec_ret << std::endl;
        std::cout << "Action Rec: " + LABEL + "  " + to_string(BEHAVIOR_RECOGNITION_NUM) << std::endl;
        std::cout << "Camera : " + camera_status << std::endl;
*/
        cv::imshow(kWinName, SRC_FRAME);

        ch = cv::waitKey(10);
      //  usleep(10000);
        //if (ch == 27 || ch == 'q' || data_ptr->EXIT_PROGRAM) {
        if (data_ptr->EXIT_PROGRAM || data_ptr->MQTT_EXIT_SIGN) {
            data_ptr->SERIALPORT_RECEIVE_DATA_CAN_RUNNING = false;
            THREAD_CAN_RUNNING = false;
       //     cout << ".................................." << data_ptr->RECEIVE_DATA_CAN_RUNNING << " " << THREAD_CAN_RUNNING << endl;
            while(true) {
                usleep(300000);
            //    cout << data_ptr->receive_Data_function_exit_sign << endl;
                if (face_detect_function_exit_sign &&
                    face_recognition_function_exit_sign &&
                    data_ptr->SERIAL_PORT_EXIT_STATUS &&
                    data_ptr->MQTT_EXIT_SIGN)
                {
                    usleep(100000);
                    break;
                }
            }
            break;
        }
    }

    cv::destroyAllWindows();
    src_frame_gray.release();
    cap.release();
//    sleep(3);
    if(!shmdt(data_ptr)) {
        printf("Separate the shared memory from the process.\n");
    }
    if(!shmctl(shmid, IPC_RMID, 0))
        printf("Delete the shared memory.\n");

    return 0;
}
