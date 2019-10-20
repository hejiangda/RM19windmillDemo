/*******************************************************************************
MIT License

Copyright (c) 2019 何江达

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*******************************************************************************/

#include <iostream>
#include <opencv2/opencv.hpp>
#include <numeric>
#include<chrono>
using namespace cv;
using namespace cv::ml;
using namespace std;
//获取点间距离
double getDistance(Point A,Point B)
{
    double dis;
    dis=pow((A.x-B.x),2)+pow((A.y-B.y),2);
    return sqrt(dis);
}
//标准化并计算hog
vector<float> stander(Mat im)
{

    if(im.empty()==1)
    {
        cout<<"filed open"<<endl;
    }
    resize(im,im,Size(48,48));

    vector<float> result;

    HOGDescriptor hog(cvSize(48,48),cvSize(16,16),cvSize(8,8),cvSize(8,8),9,1,-1,
                      HOGDescriptor::L2Hys,0.2,false,HOGDescriptor::DEFAULT_NLEVELS);           //初始化HOG描述符
    hog.compute(im,result);
    return result;
}
//将图片转换为svm所需格式
Mat get(Mat input)
{
    vector<float> vec=stander(input);
    if(vec.size()!=900) cout<<"wrong not 900"<<endl;
    Mat output(1,900,CV_32FC1);

    Mat_<float> p=output;
    int jj=0;
    for(vector<float>::iterator iter=vec.begin();iter!=vec.end();iter++,jj++)
    {
        p(0,jj)=*(iter);
    }
    return output;
}


/*
 * 参考: http://blog.csdn.net/liyuanbhu/article/details/50889951
 * 通过最小二乘法来拟合圆的信息
 * pts: 所有点坐标
 * center: 得到的圆心坐标
 * radius: 圆的半径
 */
static bool CircleInfo2(std::vector<cv::Point2f>& pts, cv::Point2f& center, float& radius)
{
    center = cv::Point2d(0, 0);
    radius = 0.0;
    if (pts.size() < 3) return false;;

    double sumX = 0.0;
    double sumY = 0.0;
    double sumX2 = 0.0;
    double sumY2 = 0.0;
    double sumX3 = 0.0;
    double sumY3 = 0.0;
    double sumXY = 0.0;
    double sumX1Y2 = 0.0;
    double sumX2Y1 = 0.0;
    const double N = (double)pts.size();
    for (int i = 0; i < pts.size(); ++i)
    {
        double x = pts.at(i).x;
        double y = pts.at(i).y;
        double x2 = x * x;
        double y2 = y * y;
        double x3 = x2 *x;
        double y3 = y2 *y;
        double xy = x * y;
        double x1y2 = x * y2;
        double x2y1 = x2 * y;

        sumX += x;
        sumY += y;
        sumX2 += x2;
        sumY2 += y2;
        sumX3 += x3;
        sumY3 += y3;
        sumXY += xy;
        sumX1Y2 += x1y2;
        sumX2Y1 += x2y1;
    }
    double C = N * sumX2 - sumX * sumX;
    double D = N * sumXY - sumX * sumY;
    double E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX;
    double G = N * sumY2 - sumY * sumY;
    double H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY;

    double denominator = C * G - D * D;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double a = (H * D - E * G) / (denominator);
    denominator = D * D - G * C;
    if (std::abs(denominator) < DBL_EPSILON) return false;
    double b = (H * C - E * D) / (denominator);
    double c = -(a * sumX + b * sumY + sumX2 + sumY2) / N;

    center.x = a / (-2);
    center.y = b / (-2);
    radius = std::sqrt(a * a + b * b - 4 * c) / 2;
    return true;
}

//模板匹配
double TemplateMatch(cv::Mat image, cv::Mat tepl, cv::Point &point, int method)
{
    int result_cols =  image.cols - tepl.cols + 1;
    int result_rows = image.rows - tepl.rows + 1;
//    cout <<result_cols<<" "<<result_rows<<endl;
    cv::Mat result = cv::Mat( result_cols, result_rows, CV_32FC1 );
    cv::matchTemplate( image, tepl, result, method );

    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc( result, &minVal, &maxVal, &minLoc, &maxLoc, Mat() );

    switch(method)
    {
    case CV_TM_SQDIFF:
    case CV_TM_SQDIFF_NORMED:
        point = minLoc;
        return minVal;

    default:
        point = maxLoc;
        return maxVal;

    }
}

//#define USE_CAMERA
//#define SAVE_VIDEO
//#define LEAF_IMG
//#define DEBUG
//#define DEBUG_LOG
#define USE_TEMPLATE
//#define USE_SVM
#define SHOW_RESULT
#define SHOW_CIRCLE
//#define SHOW_ALL_CONTOUR
#define RED
int main(int argc, char *argv[])
{
#ifdef USE_CAMERA
    VideoCapture cap(0);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    cap.set(CV_CAP_PROP_FOURCC,CV_FOURCC('M','J','P','G'));
#else
    VideoCapture cap;
    cap.open("../RM19windmillDemo/red.avi");
#endif
#ifdef LEAF_IMG
    //用于记录扇叶编号，方便保存图片
    int cnnt=0;
#endif
#ifdef USE_SVM
    //load svm model
    Ptr<SVM> svm=SVM::create();
    svm=SVM::load("../RM19windmillDemo/SVM4_9.xml");
#endif
    // Save video
#ifdef SAVE_VIDEO
    VideoWriter writer;
    bool isRecording = false;
    time_t t;
    time(&t);
    const string fileName = "/home/happy/视频/" + to_string(t) + ".avi";
    writer.open(fileName, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(1280, 720));
    //    writer.open(fileName, CV_FOURCC('M', 'J', 'P', 'G'), 30, Size(640, 480));
#endif

    Mat srcImage;
    cap >> srcImage;
    // 画拟合圆
    Mat drawcircle=Mat(srcImage.rows,srcImage.cols, CV_8UC3, Scalar(0, 0, 0));
#ifdef USE_TEMPLATE
    Mat templ[9];
    for(int i=1;i<=8;i++)
    {
        templ[i]=imread("../RM19windmillDemo/template/template"+to_string(i)+".jpg",IMREAD_GRAYSCALE);
    }
#endif
    vector<Point2f> cirV;

    Point2f cc=Point2f(0,0);

    //程序主循环

    while(true)
    {
        cap >> srcImage;
        auto t1 = chrono::high_resolution_clock::now();
#ifdef SAVE_VIDEO
        if(!writer.isOpened())
        {
            cout << "Capture failed." << endl;
            continue;
        }
        if(isRecording)
        {
            writer << srcImage;
        }
        if(!isRecording)
            cout << "Start capture. " + fileName +" created." << endl;
        isRecording = true;
#endif
        //分割颜色通道
        vector<Mat> imgChannels;
        split(srcImage,imgChannels);
        //获得目标颜色图像的二值图
#ifdef RED
        Mat midImage2=imgChannels.at(2)-imgChannels.at(0);
#endif
#ifndef RED
        Mat midImage2=imgChannels.at(0)-imgChannels.at(2);
#endif
        //二值化，背景为黑色，图案为白色
        //用于查找扇叶
        threshold(midImage2,midImage2,100,255,CV_THRESH_BINARY);
#ifdef DEBUG
        imshow("midImage2",midImage2);
#endif  
        int structElementSize=2;
        Mat element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
        //膨胀
        dilate(midImage2,midImage2,element);
        //开运算，消除扇叶上可能存在的小洞
        structElementSize=3;
        element=getStructuringElement(MORPH_RECT,Size(2*structElementSize+1,2*structElementSize+1),Point(structElementSize,structElementSize));
        morphologyEx(midImage2,midImage2, MORPH_CLOSE, element);
#ifdef DEBUG
        imshow("dilate",midImage2);
#endif
        //查找轮廓
        vector<vector<Point>> contours2;
        vector<Vec4i> hierarchy2;
        findContours(midImage2,contours2,hierarchy2,CV_RETR_TREE,CHAIN_APPROX_SIMPLE);

        RotatedRect rect_tmp2;
        bool findTarget=0;

        //遍历轮廓
        if(hierarchy2.size())
            for(int i=0;i>=0;i=hierarchy2[i][0])
            {
                rect_tmp2=minAreaRect(contours2[i]);
                Point2f P[4];
                rect_tmp2.points(P);

                Point2f srcRect[3];
                Point2f dstRect[3];

                double width;
                double height;

                //矫正提取的叶片的宽高
                width=getDistance(P[0],P[1]);
                height=getDistance(P[1],P[2]);
                if(width>height)
                {
                    srcRect[0]=P[0];
                    srcRect[1]=P[1];
                    srcRect[2]=P[2];
                }
                else
                {
                    swap(width,height);
                    srcRect[0]=P[1];
                    srcRect[1]=P[2];
                    srcRect[2]=P[3];
                }
#ifdef SHOW_ALL_CONTOUR
                Scalar color(rand() & 255, rand() & 255, rand() & 255);
                drawContours(srcImage, contours2, i, color, 4, 8, hierarchy2);
#endif
                //通过面积筛选
                double area=height*width;
                if(area>5000){
#ifdef DEBUG_LOG
                    cout <<hierarchy2[i]<<endl;

#endif
                    dstRect[0]=Point2f(0,0);
                    dstRect[1]=Point2f(width,0);
                    dstRect[2]=Point2f(width,height);
                    // 应该用透视变换的，这里用了仿射变换，我把他们搞混了……，矫正成规则矩形
                    Mat warp_mat=getAffineTransform(srcRect,dstRect);
                    Mat warp_dst_map;
                    warpAffine(midImage2,warp_dst_map,warp_mat,warp_dst_map.size());
#ifdef DEBUG
                    imshow("warpdst",warp_dst_map);
#endif
                    // 提取扇叶图片
                    Mat testim;
                    testim = warp_dst_map(Rect(0,0,width,height));
#ifdef LEAF_IMG
                    //用于保存扇叶图片，以便接下来训练svm
                    string s="leaf"+to_string(cnnt)+".jpg";
                    cnnt++;
                    imwrite("./img/"+s,testim);
#endif

#ifdef DEBUG
                    imshow("testim",testim);
#endif
                    if(testim.empty())
                    {
                        cout<<"filed open"<<endl;
                        return -1;
                    }
#ifdef USE_TEMPLATE
                    cv::Point matchLoc;
                    double value;
                    Mat tmp1;
                    resize(testim,tmp1,Size(42,20));
#endif
#if (defined DEBUG)&&(defined USE_TEMPLATE)
                    imshow("temp1",tmp1);
#endif
#ifdef USE_TEMPLATE
                    vector<double> Vvalue1;
                    vector<double> Vvalue2;
                    for(int j=1;j<=6;j++)
                    {
                        value = TemplateMatch(tmp1, templ[j], matchLoc, CV_TM_CCOEFF_NORMED);
                        Vvalue1.push_back(value);
                    }
                    for(int j=7;j<=8;j++)
                    {
                        value = TemplateMatch(tmp1, templ[j], matchLoc, CV_TM_CCOEFF_NORMED);
                        Vvalue2.push_back(value);
                    }
                    int maxv1=0,maxv2=0;

                    for(int t1=0;t1<6;t1++)
                    {
                        if(Vvalue1[t1]>Vvalue1[maxv1])
                        {
                            maxv1=t1;
                        }
                    }
                    for(int t2=0;t2<2;t2++)
                    {
                        if(Vvalue2[t2]>Vvalue2[maxv2])
                        {
                            maxv2=t2;
                        }
                    }
#endif
#if (defined DEBUG_LOG)&&(defined USE_TEMPLATE)
                    cout<<Vvalue1[maxv1]<<endl;
                    cout<<Vvalue2[maxv2]<<endl;
#endif
#ifdef USE_SVM
                    //转化为svm所要求的格式
                    Mat test=get(testim);
#endif
                    //预测是否是要打击的扇叶
#ifdef USE_TEMPLATE
                    if(Vvalue1[maxv1]>Vvalue2[maxv2]&&Vvalue1[maxv1]>0.6)
#endif
#ifdef USE_SVM
                    if(svm->predict(test)>=0.9)
#endif
                    {
                        findTarget=true;
                        //查找装甲板
                        if(hierarchy2[i][2]>0)
                        {
                            RotatedRect rect_tmp=minAreaRect(contours2[hierarchy2[i][2]]);
                            Point2f Pnt[4];
                            rect_tmp.points(Pnt);
                            const float maxHWRatio=0.7153846;
                            const float maxArea=2000;
                            const float minArea=500;

                            float width=rect_tmp.size.width;
                            float height=rect_tmp.size.height;
                            if(height>width)
                                swap(height,width);
                            float area=width*height;

                            if(height/width>maxHWRatio||area>maxArea ||area<minArea){
#ifdef DEBUG
                                 cout<<"hw "<<height/width<<"area "<<area<<endl;
                                 for(int j=0;j<4;++j)
                                 {
                                     line(srcImage,Pnt[j],Pnt[(j+1)%4],Scalar(255,0,255),4);
                                 }
                                 for(int j=0;j<4;++j)
                                 {
                                     line(srcImage,P[j],P[(j+1)%4],Scalar(255,255,0),4);
                                 }
                                 imshow("debug",srcImage);
                                 waitKey(0);
#endif
                                 continue;
                            }
                            Point centerP=rect_tmp.center;
                            //打击点
                            circle(srcImage,centerP,1,Scalar(0,255,0),2);
#ifdef SHOW_CIRCLE
                            circle(drawcircle,centerP,1,Scalar(0,0,255),1);
                            //用于拟合圆，用30个点拟合圆
                            if(cirV.size()<30)
                            {
                                cirV.push_back(centerP);
                            }
                            else
                            {
                                float R;
                                //得到拟合的圆心
                                CircleInfo2(cirV,cc,R);
                                circle(drawcircle,cc,1,Scalar(255,0,0),2);
#endif
#if (defined DEBUG_LOG)&& (defined SHOW_CIRCLE)
                                cout<<endl<<"center "<<cc.x<<" , "<<cc.y<<endl;
#endif
#ifdef SHOW_CIRCLE
                                cirV.erase(cirV.begin());

                            }
                            if(cc.x!=0&&cc.y!=0){
                                Mat rot_mat=getRotationMatrix2D(cc,30,1);
#endif
#if (defined DEBUG_LOG)&&(defined SHOW_CIRCLE)
                                cout<<endl<<"center1 "<<cc.x<<" , "<<cc.y<<endl;
#endif
#ifdef SHOW_CIRCLE
                                float sinA=rot_mat.at<double>(0,1);//sin(60);
                                float cosA=rot_mat.at<double>(0,0);//cos(60);
                                float xx=-(cc.x-centerP.x);
                                float yy=-(cc.y-centerP.y);
                                Point2f resPoint=Point2f(cc.x+cosA*xx-sinA*yy,cc.y+sinA*xx+cosA*yy);
                                circle(srcImage,resPoint,1,Scalar(0,255,0),10);
                            }
#endif
                            for(int j=0;j<4;++j)
                            {
                                line(srcImage,Pnt[j],Pnt[(j+1)%4],Scalar(0,255,255),2);
                            }
                        }
                    }
                }
#ifdef DEBUG_LOG
                cout<<"width2 " << width << " height2 "<<height<<" Hwratio2 "<<height/width<<" area2 "<<area<<endl;
#endif

            }

#if (defined SHOW_CIRCLE)&&(defined SHOW_RESULT)
        imshow("circle",drawcircle);
#endif
#ifdef SHOW_RESULT
        imshow("Result",srcImage);
        if('q'==waitKey(1))break;
#endif
        //函数所花的时间
        auto t2 = chrono::high_resolution_clock::now();
        cout << "Total period: " << (static_cast<chrono::duration<double, std::milli>>(t2 - t1)).count() << " ms" << endl;
//        t1 = chrono::high_resolution_clock::now();

    }
    return 0;
}



