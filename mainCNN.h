#include <iostream>
#include <vector>
#ifndef MAINCNN_H
#define MAINCNN_H

#endif // MAINCNN_H

class Channel{
public:
    Channel(){}
    ~Channel(){}

    // Parameter:
    std::vector<std::vector<int>> imageM; //Channel Matrix

    std::vector<std::vector<int>> kernelM;  //Kernel matrix

    std::vector<int> trace;    //Backtrace Vector
    std::vector<int> mask;    // Backtrace Vector

    std::vector<std::vector<int>> kernelW; //Kernel W' rotierte Kernel
    int kernelWeight;

    //Methoden der Channel Klasse:
    std::vector<std::vector<int>> get(); // Die Werte werden in die Input Matrix übertragen.
    void add();
    void set(std::vector<std::vector<int>> image, std::vector<std::vector<int>> kernelMatrix);
    void Imageclear();

    std::vector<std::vector<int>> activationLReLU();
    std::vector<std::vector<int>> actDervateLeakyReLU();
    void printMatrix();
    void showKernel();

    ///Convolution Neural Network:
    void do_Kernel();   // Channel 28
    void do_Kernel32(); // Channel 32
    void do_Kernel2();  // MaxPooling
    int maxPooling32(std::vector<std::vector<int>> image);

    void do_Kernel3();
    int maxPooling64(std::vector<std::vector<int>> image);

    std::vector<std::vector<int>> transition(std::vector<std::vector<int>> inputMatrix, size_t rowIndex,
                   size_t colIndex, size_t rowSize, size_t colSize);

    // Image * Kernel = Feature Map:
    int convolution(std::vector<std::vector<int>> image, std::vector<std::vector<int>> kernel);

    void connect();
    int conv64(std::vector<std::vector<int>> iMatrix, std::vector<std::vector<int> > kernel);

    void back_propConv64();
    void back_propConv32();
    std::vector<std::vector<int>> rotationKernel();
};

class Kernel : public Channel{
public:
    //Parameter:
    std::vector<std::vector<int> > kernelMatrix;

    //Methoden der Kernel Klasse:
    void setKernel(std::vector<std::vector<int> > kernel);
    std::vector<std::vector<int>> resetKernel();
    void showKernel();
};

