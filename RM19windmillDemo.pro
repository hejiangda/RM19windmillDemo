TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp
INCLUDEPATH += -I/usr/local/include/opencv \
               -I/usr/local/include

LIBS += -L/usr/local/lib -lopencv_objdetect -lopencv_shape -lopencv_ml -lopencv_superres -lopencv_stitching -lopencv_videostab -lopencv_calib3d -lopencv_video -lopencv_dnn -lopencv_features2d -lopencv_flann -lopencv_highgui -lopencv_videoio -lopencv_imgcodecs -lopencv_photo -lopencv_imgproc -lopencv_core
