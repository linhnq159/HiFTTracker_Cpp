/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//!
//! sampleOnnxMNIST.cpp
//! This file contains the implementation of the ONNX MNIST sample. It creates the network using
//! the MNIST onnx model.
//! It can be run with the following command line:
//! Command: ./sample_onnx_mnist [-h or --help] [-d=/path/to/data/dir or --datadir=/path/to/data/dir]
//! [--useDLACore=<int>]
//!

#include "argsParserSiam.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

#include "trttemplate.h"
#include "trttrack.h"
#include "hifttracker.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <glob.h>
#include <experimental/filesystem>
#include <fstream>
#include <sys/stat.h>
#include <cctype>


namespace fs = std::experimental::filesystem;
using namespace std;
using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

const std::string gSampleName = "TensorRT.siamrpn_TRT";

//!
//! \brief Initializes members of the params struct using the command line args
//!
samplesCommon::TrtParams initializeSampleParams(const samplesCommon::Args& args)
{
    samplesCommon::TrtParams params;
//    // Path to File Tensorrt
    params.TrtFileTemplate = "../../model/model_template_fp16.trt";
    params.TrtFileTrack = "../../model/model_search_fp16.trt";

    // Name Input Output Template
    // Check model ONNX
    params.inputTensorNamesTemplate = {"input.0"};
    params.outputTensorNamesTemplate = {"output.1","output.2","output.3"};

    // Name Input Output Track
    params.inputTensorNamesTrack = {"input.0","input.1","input.2","input.3"};
    params.outputTensorNamesTrack = {"output.1","output.2","output.3"};

    return params;
}

//!
//! \brief Prints the help information for running this sample
//!
void printHelpInfo()
{
    std::cout
        << "Usage: ./siamrpn [-h or --help]"
        << std::endl;
}


int main(int argc, char** argv)
{
    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        printHelpInfo();
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        printHelpInfo();
        return EXIT_SUCCESS;
    }

    auto trackTest = sample::gLogger.defineTest(gSampleName, argc, argv);

    sample::gLogger.reportTestStart(trackTest);

    samplesCommon::TrtParams params = initializeSampleParams(args);

//    Create model TensorRT Template, Track
    TRTTemplate temp(params);
    TRTTrack track(params);

//     Create Tracker
    HiFTTracker tracker(&temp, &track);

//    std::string file_input_video = "path_to_video";
    std::string file_input_video = "/home/oem/Tracking/HiFT/demo/bag.avi";
    std::string video_name = "HiFTTracker" ;

    cv::VideoCapture cap (file_input_video);
    int totalFrame = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_COUNT));
    int frameCount = 1;
    int first_frame = true;

    if (!cap.isOpened()) {
        std::cerr << "Failed to open video !" << std::endl;
        return -1;
    }

    while(frameCount <= totalFrame){
        cv::Mat curFrame;
        cap >> curFrame;
        if (first_frame) {
            // Caculate template branch
//           cv::Rect2d init_rect = cv::selectROI(video_name, curFrame, false, false);
           cv::Rect2d init_rect(299,128,143,144);
           std::cout << "init_rect : " << init_rect << std::endl;
           auto time_start = std::chrono::system_clock::now();
           tracker.init(curFrame,init_rect);
           auto time_end = std::chrono::system_clock::now();
           std::chrono::duration<double> elapsed_seconds = time_end - time_start;
           std::cout << "Time init : " << elapsed_seconds.count() << std::endl;

//           time_start = std::chrono::system_clock::now();

//           tracker.init(curFrame,tracker.bbox);
//           time_end = std::chrono::system_clock::now();
//           elapsed_seconds = time_end - time_start;
           std::cout << "Time init : " << elapsed_seconds.count() << std::endl;

           first_frame = false;
       } else {
            // Caculate search branch
            auto time_start = std::chrono::system_clock::now();
            tracker.update(curFrame);
            auto time_end = std::chrono::system_clock::now();
            std::chrono::duration<double> elapsed_seconds = time_end - time_start;
            std::cout << "Time track : " << elapsed_seconds.count() << std::endl;
           rectangle(curFrame, tracker.bbox, cv::Scalar(255, 0, 255), 1);
           cv::imshow(video_name,curFrame);
           cv::waitKey(1);

        }
        frameCount++ ;
    }
    cap.release();
    cv::destroyAllWindows();

    return sample::gLogger.reportPass(trackTest);
}
