#ifndef TRTTEMPLATE_H
#define TRTTEMPLATE_H

#include "argsParserSiam.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

using namespace nvinfer1;
using samplesCommon::SampleUniquePtr;

class TRTTemplate
{
public:
    TRTTemplate(const samplesCommon::TrtParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        CLoadEngine(params.TrtFileTemplate);
    }

    bool infer(cv::Mat& frame);

    cv::Mat output1, output2, output3;


private:
    samplesCommon::TrtParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDimsTemplate;  //!< The dimensions of the input1 to the network.
    nvinfer1::Dims mOutputDimsTemplate, mOutputDimsTemplate1, mOutputDimsTemplate2; //!< The dimensions of the output1 to the network.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    bool CLoadEngine(char *TrtPath); // Load file .trt

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers, cv::Mat & frame);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);

};


#endif // TRTTEMPLATE_H
