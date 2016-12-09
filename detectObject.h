#pragma once


#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

void detectLargestObject(const Mat &img, CascadeClassifier &cascade, Rect &largestObject, int scaledWidth = 320);
void detectManyObjects(const Mat &img, CascadeClassifier &cascade, vector<Rect> &objects, int scaledWidth = 320);
