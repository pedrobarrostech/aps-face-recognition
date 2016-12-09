#pragma once


#include <stdio.h>
#include <iostream>
#include <vector>
#include "opencv2/opencv.hpp"


using namespace cv;
using namespace std;

Ptr<FaceRecognizer> learnCollectedFaces(const vector<Mat> preprocessedFaces, const vector<int> faceLabels, const string facerecAlgorithm = "FaceRecognizer.Eigenfaces");

void showTrainingDebugData(const Ptr<FaceRecognizer> model, const int faceWidth, const int faceHeight);

Mat reconstructFace(const Ptr<FaceRecognizer> model, const Mat preprocessedFace);

double getSimilarity(const Mat A, const Mat B);
