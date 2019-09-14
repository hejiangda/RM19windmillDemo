TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp
#Libraries
unix: CONFIG += link_pkgconfig

#OpenCV
unix: PKGCONFIG += opencv
DISTFILES += \
    LICENSE \
    README.md \
    SVM4_9.xml \
    red.avi \
    template/template1.jpg \
    template/template2.jpg \
    template/template3.jpg \
    template/template4.jpg \
    template/template5.jpg \
    template/template6.jpg \
    template/template7.jpg \
    template/template8.jpg
