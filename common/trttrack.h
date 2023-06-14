#ifndef TRTTRACK_H
#define TRTTRACK_H

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

class TRTTrack
{
public:
    TRTTrack(const samplesCommon::TrtParams& params)
        : mParams(params)
        , mEngine(nullptr)
    {
        CLoadEngineTrack(params.TrtFileTrack);
    }



    bool infer(cv::Mat& frame, float* hostDataBuffer1, float* hostDataBuffer2, float* hostDataBuffer3);
//    at::Tensor out_cls, out_reg;

    cv::Mat output_cls1, output_cls2, output_loc;
    int size_cls1, size_cls2, size_loc;


private:
    samplesCommon::TrtParams mParams; //!< The parameters for the sample.

    nvinfer1::Dims mInputDimsTrack, mInputDimsTrack1, mInputDimsTrack2, mInputDimsTrack3;  //!< The dimensions of the input1 to the network.
    nvinfer1::Dims mOutputDimsTrack1, mOutputDimsTrack2, mOutputDimsTrack3 ; //!< The dimensions of the output1 to the network.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
    SampleUniquePtr<nvinfer1::IExecutionContext> context;

    bool CLoadEngineTrack(char *TrtPath);

    //!
    //! \brief Reads the input  and stores the result in a managed buffer
    //!
    bool processInput(const samplesCommon::BufferManager& buffers,cv::Mat& frame, float* hostDataBuffer1, float* hostDataBuffer2, float* hostDataBuffer3);

    //!
    //! \brief Classifies digits and verify result
    //!
    bool verifyOutput(const samplesCommon::BufferManager& buffers);
};

#endif // TRTTRACK_H
