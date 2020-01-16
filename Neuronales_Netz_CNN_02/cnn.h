#include <QCoreApplication>
#include <iostream>
#include <algorithm>
#include <math.h>

#ifndef CNN_H
#define CNN_H
#endif // CNN_H

class Channel
{
private:

    int ranNum;
    float weight;  // Kernel-Gewicht
    //weight = rand() / float(RAND_MAX);  //Generates random number from 0 to 1
    float maxLReLu;
    float matrixLayer;


public:

    void conv3D();
    void getChannel();
    void setChannel();
    void addChannel();
    void clearChannel();

    float lReLu();
};

class Kernell
{
private:

    int n;

    float weight;  // Kernel-Gewicht
    float matrix;  // 20x20x20 x 28 Matrix
    float kernell[3][3][3];

public:
    float reset();
};

