#include <iostream>
#include <vector>
#ifndef MAINCNN_H
#define MAINCNN_H

#endif // MAINCNN_H

class Channel
{
public:

    Channel(){}
    ~Channel(){}

    std::vector<int> indexCount;
    int read_in();

    std::vector<std::vector<std::vector<int>>> input_vec;

    std::vector<std::vector<std::vector<int>>> imageM;  // Channel-Matrix
    std::vector<std::vector<std::vector<int>>> kernelM; // Kernel-MAtrix


    std::vector<int> trace;    //Backtrace Vector 10x10x10
    std::vector<std::vector<std::vector<int>>> mask;    // Backtrace Vector 5x5x5

    std::vector<std::vector<std::vector<int>>> kernelW; //Kernel W' rotierte Kernel

    int kernelWeight;

    std::vector<int> change_w;    // changed weights
    std::vector<int> x_change;
    //
    std::vector<std::vector<std::vector<int>>> out_value;
    std::vector<std::vector<std::vector<int>>> o_matrix; // Original Matrix

    double bias;

    //int zeile, spalte, k2, l2;

    int setMatrix(std::vector<std::vector<std::vector<int>>> vec, std::vector<std::vector<std::vector<int>>> kernel);

    void printMatrix();

    void showKernel();

    void imageclear();

    std::vector<std::vector<std::vector<int>>> get();
    //
    void add();
//////////////////////////////////////////////// Convolution //////////////////////////////////////////////////////////

    std::vector<std::vector<std::vector<int>>> transition(std::vector<std::vector<std::vector<int>>> inputMatrix,
                                             size_t rowIndex, size_t colIndex, size_t kIndex, size_t rowSize, size_t colSize, size_t kSize);

    int convolution(std::vector<std::vector<std::vector<int>>> image,
                    std::vector<std::vector<std::vector<int>>> kernell);

    std::vector<std::vector<std::vector<int>>> activationLReLU(); // Leaky ReLU
    std::vector<std::vector<std::vector<int>>> actDervateLeakyReLU();


    void do_Kernel();
    void do_Kernel2();  // MaxPooling
    int maxPooling32(std::vector<std::vector<std::vector<int>>> image);
////////////////////////////////////////////////////////////////////////////

    void connect();
    int conv64(std::vector<std::vector<std::vector<int>>> image,
               std::vector<std::vector<std::vector<int>>> kernell);

    void do_Kernel3(); // Conv32 10x10x10

    void do_Kernel32(); // Channel 32 x 10x10x10

    int maxPooling64(std::vector<std::vector<std::vector<int>>> image); // MaxPooling 64 x 10x10x10

    void weight_change();

    void back_propConv32();
    void back_propConv64();
    void backpro_Maxpool32();
    void backpro_Maxpool64();

    int backMax64(std::vector<std::vector<std::vector<int> > > image);

    int rotationKernel();

    double biasUpdate();
    void backAdd();
    void printMask();
    //std::vector<std::vector<std::vector<int>>> actDervateLeakyReLU();
};

class Kernel : public Channel{
public:
    //Parameter:
    std::vector<std::vector<std::vector<int>>> kernelMatrix;

    //Methoden der Kernel Klasse:
    void setKernel(std::vector<std::vector<std::vector<int>>> kernel);
    std::vector<std::vector<std::vector<int>>> resetKernel();
    void showKernel();
};
