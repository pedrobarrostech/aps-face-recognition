#ifndef IMAGEUTILS_0_7_H_
#define IMAGEUTILS_0_7_H_



#include <cv.h>
//#include <cvaux.h>
#include <cxcore.h>
#ifdef USE_HIGHGUI
    #include <highgui.h>
#endif


#include <stdio.h>
#if defined WIN32 || defined _WIN32
    #include <conio.h>  
    #include <direct.h>       
    #define snprintf sprintf_s  
#else
    #include <stdio.h>        
    #include <termios.h> 
    #include <unistd.h>
    #include <sys/types.h>
    #include <sys/stat.h>   
#endif
#include <vector>
#include <string>
#include <iostream>           

#ifndef LOG

    #ifndef _MSC_VER

        #define LOG(fmt, args...) do {printf(fmt, ## args); printf("\n"); fflush(stdout);} while (0)

    #else
        #define LOG printf
    #endif

#endif


#ifdef __cplusplus
extern "C"
{
#endif



#ifndef UCHAR
    typedef unsigned char UCHAR;
#endif

#ifndef __cplusplus
    typedef int bool;
    #define true (1)
    #define false (0)
#endif

#ifdef __cplusplus
    #define DEFAULT(val) = val
#else
    #define DEFAULT(val)
#endif



#define DECLARE_TIMING(s)           int64 timeStart_##s; int64 timeDiff_##s; int64 timeTally_##s = 0; int64 countTally_##s = 0; double timeMin_##s = DBL_MAX; double timeMax_##s = 0; int64 timeEnd_##s;
#define START_TIMING(s)             timeStart_##s = cvGetTickCount()
#define STOP_TIMING(s)              do {    timeEnd_##s = cvGetTickCount(); timeDiff_##s = (timeEnd_##s - timeStart_##s); timeTally_##s += timeDiff_##s; countTally_##s++; timeMin_##s = MIN(timeMin_##s, timeDiff_##s); timeMax_##s = MAX(timeMax_##s, timeDiff_##s);    } while (0)
#define GET_TIMING(s)               (double)(0.001 * ( (double)timeDiff_##s / (double)cvGetTickFrequency() ))
#define GET_MIN_TIMING(s)           (double)(0.001 * ( (double)timeMin_##s / (double)cvGetTickFrequency() ))
#define GET_MAX_TIMING(s)           (double)(0.001 * ( (double)timeMax_##s / (double)cvGetTickFrequency() ))
#define GET_AVERAGE_TIMING(s)       (double)(countTally_##s ? 0.001 * ( (double)timeTally_##s / ((double)countTally_##s * cvGetTickFrequency()) ) : 0)
#define GET_TOTAL_TIMING(s)         (double)(0.001 * ( (double)timeTally_##s / ((double)cvGetTickFrequency()) ))
#define GET_TIMING_COUNT(s)         (int)(countTally_##s)
#define CLEAR_AVERAGE_TIMING(s)     do {    timeTally_##s = 0; countTally_##s = 0;     } while (0)
#define SHOW_TIMING(s, msg)         LOG("%s time:\t %dms\t (ave=%dms min=%dms max=%dms, across %d runs).", msg, cvRound(GET_TIMING(s)), cvRound(GET_AVERAGE_TIMING(s)), cvRound(GET_MIN_TIMING(s)), cvRound(GET_MAX_TIMING(s)), GET_TIMING_COUNT(s) )
#define SHOW_TOTAL_TIMING(s, msg)   LOG("%s total:\t %dms\t (ave=%dms min=%dms max=%dms, across %d runs).", msg, cvRound(GET_TOTAL_TIMING(s)), cvRound(GET_AVERAGE_TIMING(s)), cvRound(GET_MIN_TIMING(s)), cvRound(GET_MAX_TIMING(s)), GET_TIMING_COUNT(s) )
#define AVERAGE_TIMING(s)           SHOW_TIMING(s, #s)
#define TOTAL_TIMING(s)             do {    SHOW_TOTAL_TIMING(s, #s); CLEAR_AVERAGE_TIMING(s);     } while (0)

inline int roundFloat(float f);

IplImage* drawFloatGraph(const float *arraySrc, int nArrayLength, IplImage *imageDst DEFAULT(0), float minV DEFAULT(0.0), float maxV DEFAULT(0.0), int width DEFAULT(0), int height DEFAULT(0), char *graphLabel DEFAULT(0), bool showScale DEFAULT(true));

IplImage* drawIntGraph(const int *arraySrc, int nArrayLength, IplImage *imageDst DEFAULT(0), int minV DEFAULT(0), int maxV DEFAULT(0), int width DEFAULT(0), int height DEFAULT(0), char *graphLabel DEFAULT(0), bool showScale DEFAULT(true));

IplImage* drawUCharGraph(const uchar *arraySrc, int nArrayLength, IplImage *imageDst DEFAULT(0), int minV DEFAULT(0), int maxV DEFAULT(0), int width DEFAULT(0), int height DEFAULT(0), char *graphLabel DEFAULT(0), bool showScale DEFAULT(true));

void showFloatGraph(const char *name, const float *arraySrc, int nArrayLength, int delay_ms DEFAULT(500), IplImage *background DEFAULT(0));

void showIntGraph(const char *name, const int *arraySrc, int nArrayLength, int delay_ms DEFAULT(500), IplImage *background DEFAULT(0));

void showUCharGraph(const char *name, const uchar *arraySrc, int nArrayLength, int delay_ms DEFAULT(500), IplImage *background DEFAULT(0));

void showImage(const IplImage *img, int delay_ms DEFAULT(0), char *name DEFAULT(0));

void setGraphColor(int index DEFAULT(0));

void setCustomGraphColor(int R, int B, int G);


IplImage* convertImageToGreyscale(const IplImage *imageSrc);

IplImage* convertImageYIQtoRGB(const IplImage *imageYIQ);

IplImage* convertImageRGBtoYIQ(const IplImage *imageRGB);

IplImage* convertImageHSVtoRGB(const IplImage *imageHSV);

IplImage* convertImageRGBtoHSV(const IplImage *imageRGB);

inline void convertPixelRGBtoHSV_256(int bR, int bG, int bB, int &bH, int &bS, int &bV);

inline void convertPixelHSVtoRGB_256(int bH, int bS, int bV, int &bR, int &bG, int &bB);

void convertPixelRGBtoHSV_180(int bR, int bG, int bB, int &bH, int &bS, int &bV);

void convertPixelHSVtoRGB_180(int bH, int bS, int bV, int &bR, int &bG, int &bB);


CvPoint2D32f addPointF(const CvPoint2D32f pointA, const CvPoint2D32f pointB);

CvPoint2D32f subtractPointF(const CvPoint2D32f pointA, const CvPoint2D32f pointB);

CvPoint2D32f scalePointF(const CvPoint2D32f point, float scale);

CvPoint2D32f scalePointAroundPointF(const CvPoint2D32f point, const CvPoint2D32f origin, float scale);


float scaleValueF(float p, float s, float maxVal);
#define scaleValueFI(p, s, maxVal) scaleValueF(p, s, (float)maxVal)

int scaleValueI(int p, float s, int maxVal);


CvPoint2D32f rotatePointF(const CvPoint2D32f point, float angleDegrees);

CvPoint2D32f rotatePointAroundPointF(const CvPoint2D32f point, const CvPoint2D32f origin, float angleDegrees);


float findDistanceBetweenPointsF(const CvPoint2D32f p1, const CvPoint2D32f p2);

float findDistanceBetweenPointsI(const CvPoint p1, const CvPoint p2);

float findAngleBetweenPointsF(const CvPoint2D32f p1, const CvPoint2D32f p2);

float findAngleBetweenPointsI(const CvPoint p1, const CvPoint p2);

void drawCross(IplImage *img, const CvPoint pt, int radius, const CvScalar color );


void printPoint(const CvPoint pt, const char *label);

void printPointF(const CvPoint2D32f pt, const char *label);


CvRect scaleRect(const CvRect rectIn, float scaleX, float scaleY, int w DEFAULT(0), int h DEFAULT(0));

CvRect scaleRectInPlace(const CvRect rectIn, float scaleX, float scaleY, float borderX DEFAULT(0.0f), float borderY DEFAULT(0.0f), int w DEFAULT(0), int h DEFAULT(0));

CvRect offsetRect(const CvRect rectA, const CvRect rectB);

CvRect offsetRectPt(const CvRect rectA, const CvPoint pt);

void drawRect(IplImage *img, const CvRect rect, const CvScalar color DEFAULT(CV_RGB(220,0,0)));

void drawRectFilled(IplImage *img, const CvRect rect, const CvScalar color DEFAULT(CV_RGB(220,0,0)));

void printRect(const CvRect rect, const char *label DEFAULT(0));

CvRect cropRect(const CvRect rectIn, int w, int h);


IplImage* cropImage(const IplImage *img, const CvRect region);

IplImage* resizeImage(const IplImage *origImg, int newWidth, int newHeight, bool keepAspectRatio);

IplImage *rotateImage(const IplImage *src, float angleDegrees, float scale DEFAULT(1.0f));

CvPoint2D32f mapRotatedImagePoint(const CvPoint2D32f pointOrig, const IplImage *image, float angleRadians, float scale DEFAULT(1.0f));

IplImage* combineImagesResized(int nArgs, ...);
IplImage* combineImages(int nArgs, ...);

IplImage* smoothImageBilateral(const IplImage *src, float smoothness DEFAULT(30));

IplImage* blendImage(const IplImage* image1, const IplImage* image2, const IplImage* imageAlphaMask);

IplImage* convertMatrixToUcharImage(const CvMat *srcMat);

IplImage* convertFloatImageToUcharImage(const IplImage *srcImg);

int saveImage(const char *filename, const IplImage *image);

void saveFloatMat(const char *filename, const CvMat *src);

void saveFloatImage(const char *filename, const IplImage *srcImg);

void drawText(IplImage *img, CvPoint position, CvScalar color, char *fmt, ...);


void printImageInfo(const IplImage *image_tile, const char *label DEFAULT(0));

void printImagePixels(const IplImage *image, const char *label DEFAULT(0), int maxElements DEFAULT(300));

void printMat(const cv::Mat M, const char *label DEFAULT(0), int maxElements DEFAULT(300));

void printMatrix(const CvMat *M, const char *label DEFAULT(0), int maxElements DEFAULT(300));

void printMatInfo(const cv::Mat M, const char *label DEFAULT(0));

void printPoint32f(const CvPoint2D32f pt, const char *label DEFAULT(0));

void printLine(const CvPoint ptA, const CvPoint ptB, const char *label DEFAULT(0));

void printDataRange(const CvArr *src, const char *msg);



#if defined (__cplusplus)
}
#endif


#endif 
