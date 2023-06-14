QT -= gui

CONFIG += c++11 console
CONFIG -= app_bundle

# The following define makes your compiler emit warnings if you use
# any Qt feature that has been marked deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if it uses deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

SOURCES += \
    ../common/getOptions.cpp \
    ../common/hifttracker.cpp \
    ../common/logger.cpp \
    ../common/sampleEngines.cpp \
    ../common/sampleInference.cpp \
    ../common/sampleOptions.cpp \
    ../common/sampleReporting.cpp \
    ../common/sampleUtils.cpp \
    ../common/trttemplate.cpp \
    ../common/trttrack.cpp \
    main.cpp
    main.cpp

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

HEADERS += \
    ../common/BatchStream.h \
    ../common/EntropyCalibrator.h \
    ../common/ErrorRecorder.h \
    ../common/argsParserSiam.h \
    ../common/buffers.h \
    ../common/common.h \
    ../common/getOptions.h \
    ../common/half.h \
    ../common/hifttracker.h \
    ../common/logger.h \
    ../common/logging.h \
    ../common/parserOnnxConfig.h \
    ../common/safeCommon.h \
    ../common/sampleConfig.h \
    ../common/sampleDevice.h \
    ../common/sampleEngines.h \
    ../common/sampleInference.h \
    ../common/sampleOptions.h \
    ../common/sampleReporting.h \
    ../common/sampleUtils.h \
    ../common/trttemplate.h \
    ../common/trttrack.h

INCLUDEPATH += ../common

INCLUDEPATH += /home/oem/Downloads/TensorRT-8.5.3.1/include

INCLUDEPATH += /usr/local/cuda/include

#LIBS += -L"/usr/local/cuda/lib64" -lcudnn_ops_infer_static \
#    -lcudnn_cnn_infer_static -lcudnn_adv_infer_static \
#    -lcudnn_ops_train_static -lcudnn_cnn_train_static -lcudnn_adv_train_static

#LIBS += -lcublas_static

#LIBS += -L"../../lib" -lnvinfer_static -lnvparsers_static -lnvinfer_plugin_static \
#        -lnvonnxparser_static

LIBS += -L"/usr/local/cuda/lib64" -lcudart -lcudnn

LIBS += -lcublas -ldl

LIBS += -L"/home/oem/Downloads/TensorRT-8.5.3.1/lib" -lnvinfer -lnvparsers -lnvinfer_plugin \
        -lnvonnxparser

INCLUDEPATH += /usr/local/include/opencv4
#INCLUDEPATH += /usr/local/include/opencv4/opencv2
LIBS += -L/usr/local/lib -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_flann -lopencv_objdetect -lopencv_video -lopencv_calib3d

LIBS += -L/usr/local/lib -lopencv_core -lopencv_imgcodecs -lopencv_highgui -lopencv_imgproc -lopencv_features2d -lopencv_calib3d -lopencv_videoio

LIBS += -lstdc++fs
