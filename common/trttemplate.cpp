#include "trttemplate.h"

bool TRTTemplate::CLoadEngine(char *TrtPath)
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

//    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
//    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    SampleUniquePtr<IRuntime> runtime{createInferRuntime(sample::gLogger.getTRTLogger())};
//    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), "");
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine((void *)&buf[0], mdsize, nullptr));

    // context
    context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if (!context)
    {
        return false;
    }

    // Get size Input Output
    mInputDimsTemplate = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.inputTensorNamesTemplate[0].c_str()));

    mOutputDimsTemplate = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTemplate[0].c_str()));
    mOutputDimsTemplate1 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTemplate[1].c_str()));
    mOutputDimsTemplate2 = mEngine->getBindingDimensions(mEngine->getBindingIndex(mParams.outputTensorNamesTemplate[2].c_str()));

    return true;
}

bool TRTTemplate::infer(cv::Mat& frame)
{

    // Create RAII buffer manager object
    samplesCommon::BufferManager buffers(mEngine);

//    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
//    if (!context)
//    {
//        return false;
//    }

    if (!processInput(buffers, frame))
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

bool TRTTemplate::processInput(const samplesCommon::BufferManager& buffers, cv::Mat & frame)
{
    // size input
    const int inputC = mInputDimsTemplate.d[1];
    const int inputH = mInputDimsTemplate.d[2];
    const int inputW = mInputDimsTemplate.d[3];

    // Convert cv::Mat to tensor float 1x3x127x127
    float* hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(mParams.inputTensorNamesTemplate[0]));

    // input frame to buffers
    for (int c = 0; c < inputC; ++c)
    {
        for (int h = 0; h < inputH; ++h)
        {
            for (int w = 0; w < inputW; ++w)
            {
                int dstIdx = c * inputH * inputW + h * inputW + w;
                hostDataBuffer[dstIdx] = frame.at<cv::Vec3b>(h, w)[c];
            }
        }
    }


    return true;
}

bool TRTTemplate::verifyOutput(const samplesCommon::BufferManager& buffers)
{
    // Output
    float* output = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTemplate[0]));
    float* output1 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTemplate[1]));
    float* output2 = static_cast<float*>(buffers.getHostBuffer(mParams.outputTensorNamesTemplate[2]));

    // Size output Template Branch
    int channels = mOutputDimsTemplate.d[1];
    int height = mOutputDimsTemplate.d[2];
    int width = mOutputDimsTemplate.d[3];
    int outputSize = channels * height * width;

    // Add data to output cv::Mat
    this->output1 = cv::Mat(1,outputSize,CV_32F);
    this->output2 = cv::Mat(1,outputSize,CV_32F);
    this->output3 = cv::Mat(1,outputSize,CV_32F);

    cv::Mat(1,outputSize,CV_32F,output).copyTo(this->output1);
    cv::Mat(1,outputSize,CV_32F,output1).copyTo(this->output2);
    cv::Mat(1,outputSize,CV_32F,output2).copyTo(this->output3);

    return true;
}
