#include "trttrack.h"


bool TRTTrack::CLoadEngineTrack(char *TrtPath)
{
    std::ifstream ifile(TrtPath, std::ios::in | std::ios::binary);
    if (!ifile)
    {
        std::cout << "model file: " << TrtPath << " not found!" << std::endl;
        return false;
    }
    ifile.seekg(0, std::ios::end);
    const int mdsize = ifile.tellg();
    ifile.clear();
    ifile.seekg(0, std::ios::beg);
    std::vector<char> buf(mdsize);
    ifile.read(&buf[0], mdsize);
    ifile.close();

    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine((void *)&buf[0], mdsize, nullptr));

    // Context
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Get size Input Output
    mInputDimsTrack = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTrack[0].c_str()));
    mInputDimsTrack1 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTrack[1].c_str()));
    mInputDimsTrack2 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTrack[2].c_str()));
    mInputDimsTrack3 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTrack[3].c_str()));

    mOutputDimsTrack1 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTrack[0].c_str()));
    mOutputDimsTrack2 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTrack[1].c_str()));
    mOutputDimsTrack3 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTrack[2].c_str()));

    return true;
}

bool TRTTrack::infer(cv::Mat& frame, float* hostDataBuffer1, float* hostDataBuffer2, float* hostDataBuffer3)
{
    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

//    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
//    if (!context)
//    {
//        return false;
//    }

    // Read the input data into the managed buffers
    ASSERT(mParams.inputTensorNamesTrack.size() == 4);

    if (!processInput(buffers, frame, hostDataBuffer1, hostDataBuffer2, hostDataBuffer3))
    {
        return false;
    }

    // Memcpy from host input buffers to device input buffers
    buffers.copyInputToDevice();

    bool status = context->executeV2(buffers.getDeviceBindings().data());
    if (!status)
    {
        return false;
    }

    // Memcpy from device output buffers to host output buffers
    buffers.copyOutputToHost();

    // Verify results
    if (!verifyOutput(buffers))
    {
        return false;
    }

    return true;
}

//!
//! \brief Reads the input and stores the result in a managed buffer
//!

bool TRTTrack::processInput(const samplesCommon::BufferManager& buffers, cv::Mat& frame, float* input1, float* input2, float* input3)
{
    // size input
    const int inputC = mInputDimsTrack.d[1];
    const int inputH = mInputDimsTrack.d[2];
    const int inputW = mInputDimsTrack.d[3];

    // size input1
    const int inputC1 = mInputDimsTrack1.d[1];
    const int inputH1 = mInputDimsTrack1.d[2];
    const int inputW1 = mInputDimsTrack1.d[3];

    // size input2
    const int inputC2 = mInputDimsTrack2.d[1];
    const int inputH2 = mInputDimsTrack2.d[2];
    const int inputW2 = mInputDimsTrack2.d[3];

    // size input1
    const int inputC3 = mInputDimsTrack3.d[1];
    const int inputH3 = mInputDimsTrack3.d[2];
    const int inputW3 = mInputDimsTrack3.d[3];


    // Convert cv::Mat to tensor float 1x3x127x127
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[0]));
    float* hostDataBuffer1 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[1]));
    float* hostDataBuffer2 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[2]));
    float* hostDataBuffer3 = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTrack[3]));

    // Add input to buffers
    for (int c = 0; c < inputC; ++c){
        for (int h = 0; h < inputH; ++h){
            for (int w = 0; w < inputW; ++w){
                int dstIdx = c * inputH * inputW + h * inputW + w;
                hostDataBuffer[dstIdx] = frame.at<cv::Vec3b>(h, w)[c];
            }
        }
    }

    for (int c = 0; c < inputC1; ++c) {
        for (int h = 0; h < inputH1; ++h) {
            for (int w = 0; w < inputW1; ++w) {
                int index = c * inputH1 * inputW1 + h * inputW1 + w;
                hostDataBuffer1[index] = input1[index];
            }
        }
    }

    for (int c = 0; c < inputC2; ++c) {
        for (int h = 0; h < inputH2; ++h) {
            for (int w = 0; w < inputW2; ++w) {
                int index = c * inputH2 * inputW2 + h * inputW2 + w;
                hostDataBuffer2[index] = input2[index];
            }
        }
    }

    for (int c = 0; c < inputC3; ++c) {
        for (int h = 0; h < inputH3; ++h) {
            for (int w = 0; w < inputW3; ++w) {
                int index = c * inputH3 * inputW3 + h * inputW3 + w;
                hostDataBuffer3[index] = input3[index];
            }
        }
    }
    return true;
}

bool TRTTrack::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    float *output_cls1 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTrack[0]));
    float *output_cls2 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTrack[1]));
    float *output_loc = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTrack[2]));

    // Size *output
    int channels_loc = mOutputDimsTrack3.d[1];
    int height_loc = mOutputDimsTrack3.d[2];
    int width_loc = mOutputDimsTrack3.d[3];

    int channels_cls1 = mOutputDimsTrack1.d[1];
    int height_cls1 = mOutputDimsTrack1.d[2];
    int width_cls1 = mOutputDimsTrack1.d[3];

    int channels_cls2 = mOutputDimsTrack2.d[1];
    int height_cls2 = mOutputDimsTrack2.d[2];
    int width_cls2 = mOutputDimsTrack2.d[3];

//    int outputSizeLoc = channels_loc * height_loc * width_loc;
//    int outputSizeCls1 = channels_cls1 * height_cls1 * width_cls1;
//    int outputSizeCls2 = channels_cls2 * height_cls2 * width_cls2;

    size_loc = channels_loc * height_loc * width_loc;
    size_cls1 = channels_cls1 * height_cls1 * width_cls1;
    size_cls2 = channels_cls2 * height_cls2 * width_cls2;

    this->output_loc = cv::Mat(1,size_loc,CV_32F);
    this->output_cls1 = cv::Mat(1,size_cls1,CV_32F);
    this->output_cls2 = cv::Mat(1,size_cls2,CV_32F);
    //Cach lay con tron output float*
//    this->output1.ptr<float>(0);
    cv::Mat(1,size_loc,CV_32F,output_loc).copyTo(this->output_loc);
    cv::Mat(1,size_cls1,CV_32F,output_cls1).copyTo(this->output_cls1);
    cv::Mat(1,size_cls2,CV_32F,output_cls2).copyTo(this->output_cls2);


    return true;
}
