#include <QCoreApplication>
#include <iostream>
#include "mainCNN.h"



void Channel::set(std::vector<std::vector<int> > image, std::vector<std::vector<int>> kernelMatrix){
    imageM = image; // Setze die Werte in die Matrix.
    kernelM = kernelMatrix;
}

std::vector<std::vector<int>> Channel::get(){
    return imageM; // Gebe die Image-Matrx zurück.
    return kernelM;
}

void Channel::add(){


    // Hier wird das Ergebnis der Convolution berechnet, dass mithilfe einer
    // Zwischen-Parameter gespeichert wird und diese auf die Image-Matrix 32x nxnxn
    // Übertagen wird.

    std::vector<std::vector<int>> tempMatrix(imageM.size() + 2);
    // Die Matrix wird außerhalb des Randes mit 0 durch +2 initialisiert und mit 0 gefüllt
    std::vector<int> tempVector(imageM[0].size() + 2, 0);

    //vector<vector<vector<int>>> temM3D(m3D_.size() + 2);
    //vector<int> temVec(m3D_[0].size() + 2, 0);

    tempMatrix[0] = tempVector;
    //temM3D[0] = temVec;

    for (size_t i = 1; i < imageM.size() + 1; ++i)
    {
        tempMatrix[i] = tempVector;
        for (size_t j = 1; j < imageM[0].size() + 1; ++j)
        {
            tempMatrix[i][j] = imageM[i - 1][j - 1];
            //std::cout << "|" << tempMatrix[i][j];

        }
        //std::cout << "\n";
    }
    tempMatrix[imageM.size() + 1] = tempVector;
    imageM = tempMatrix; // Ergebnis-Matrix
}

void Channel::Imageclear(){
    imageM.clear(); // Leere die Image-Matrix.
    kernelM.clear();
    std::cout << "\nDie Channel- und Kernel-Matrix wurde geleert!!!\n";
}

void Channel::printMatrix(){

    for(int i = 0; i < imageM.size(); i++)
    {
        for(int j = 0; j < imageM.size(); j++)
        {
            std::cout << "|" << imageM[i][j];
        }
        std::cout << "\n";
    }
}

void Channel::showKernel(){
    for(int i = 0; i < kernelM.size(); i++){
        for(int j = 0; j < kernelM.size(); j++){
            std::cout << "|" << kernelM[i][j];
        }
        std::cout << "\n";
    }
}

/////////////////////////////////////////////////// Convolution //////////////////////////////////////////////////////////////////////////7


void Channel::do_Kernel()
{
    std::vector<std::vector<int>> featureMap(imageM.size() - kernelM.size() + 1);
    std::cout << "featureMap-Matrix size: "<< featureMap.size() << "\n";
    //Row index
    for (size_t rowSize = 0; rowSize < featureMap.size(); ++rowSize)
    {
        std::vector<int> tVector(imageM[0].size() - kernelM[0].size() + 1);
        featureMap[rowSize] = tVector;
        // Column index
        for (size_t columnSize = 0; columnSize < tVector.size(); ++columnSize)
        {

            featureMap[rowSize][columnSize] = convolution(transition(imageM, rowSize, columnSize, kernelM.size(), kernelM[0].size()), kernelM);
            //std::cout << featureMap[rowSize][columnSize] << " ";
        }
        //std::cout << "\n";
    }
    imageM = featureMap;

}


std::vector<std::vector<int>> Channel::transition(std::vector<std::vector<int>> inputMatrix, size_t rowIndex, size_t colIndex, size_t rowSize, size_t colSize){

    std::vector<std::vector<int>> partOfMatrix(rowSize);
    //std::cout << "rowSize: " << rowSize << ", colSize: " << colSize << "\n";

    for (size_t i = 0; i < rowSize; ++i) // Zeilen Größe des Vectors
    {
         std::vector<int> vecTemp(colSize);
         partOfMatrix[i] = vecTemp;

         for (size_t j = 0; j < colSize; ++j) // Spalten Größe der Matrix
         {
              partOfMatrix[i][j] = inputMatrix[rowIndex + i][colIndex + j];

         }
    }
    //std::cout << "Input Groesse: " << partOfMatrix.size() << "\n";
    return partOfMatrix;
}


std::vector<std::vector<int>> Channel::activationLReLU()
{
    for(int i = 0; i < imageM.size(); i++){
        for(int j = 0; j < imageM.size(); j++){
            // Leaky ReLU activation -> max(0.01*x, x)
            //Falls die Matrix größer als 0.01 * matrix ist:

            //imageM[i][j] = (image[i][j] > int(0.01 * image[i][j]) ? image[i][j] : int(0.01 * image[i][j]));

            if(imageM[i][j] > 0) /// Falls x > 0 ist dann bleibt der Wert von der Matrix erhalten.
            {
                imageM[i][j] = imageM[i][j];
            }
            else /// Sonst wird max(x * 0.01, x) verrechnet und das Maximum benutzt.
            {
                imageM[i][j] = std::max(int(0.01 * imageM[i][j]), imageM[i][j]);
            }
            //imageM[i][j] = (0.01 * imageM[i][j]);
        }
    }
    return imageM;

    ///wenn der Wert vor dem : nicht gleich 0 ist, dann ist es der Wert nach dem :

}

int Channel::convolution(std::vector<std::vector<int>> image, std::vector<std::vector<int>> kernel)
{

    std::vector<std::vector<int>> featureMap(image.size() - kernel.size() + 1);

    int bias = 0;
    int s = 0;

    for(int i = 0; i < image.size(); i++)
    {
        for(int j = 0; j < image[0].size(); j++)
        {
            /// Image * Kernel + Bias = Y-Matrix 32 x 20x20x20
            //std::cout << image[i][j] << " ";
            s += image[i][j] * kernel[i][j] + bias;

            //std::cout << "|" << ss;
        }
        //std::cout << "\n";
     }

    return s;

}

/////////////////////////////////////////////// Max Pooling ///////////////////////////////////////////////////////////

void Channel::do_Kernel2()
{

    int pool = 2;
    int stride = 2;

    std::cout << "Matrixgroesse von ImageM: " << imageM.size() << "\n";

    std::vector<std::vector<int>> featureMap(int(imageM.size() / pool));

    std::cout << "Matrixgroesse von featureMap: " << featureMap.size() << "\n";

    for (size_t i = 0; i < featureMap.size(); ++i)
    {
        std::vector<int> tVector(int(imageM.size() / stride));
        //std::cout << "Groesse von tVector: " << tVector.size() << "\n";
        featureMap[i] = tVector;  // Zwischenvektor

        for (size_t j = 0; j < tVector.size(); ++j)
        {
            featureMap[i][j] = maxPooling32(transition(imageM, (kernelM.size() - 1) * i, (kernelM[0].size() - 1) * j,
                                          kernelM.size() - 1, kernelM[0].size() - 1));
        }
    }
    imageM = featureMap; //Die maximalen Werte werden in die feature Matrix gespeichert.
}


int Channel::maxPooling32(std::vector<std::vector<int>> imageM)
{
    int s = 0;
    // Test:
    //std::cout << "Max Image hat die Groesse: " << imageM.size() << "\n";

    for (size_t i = 0; i < imageM.size()  ; ++i) // Zeile
    {
        for (size_t j = 0; j < imageM[0].size() ; ++j) // Spalte
        {
            // kernell[i][j] std::vector<std::vector<int>> kernell
            s = std::max(s, imageM[i][j]); // Das maximale Element wird ausgefiltert

        }
    }
    trace.push_back(s);
    return s;
}

void Channel::connect()
{
    /* D
     * Hier wird die 2. Convolution an der 10x10x10-Matrix durchgeführt, indem
     * die Image-Matrix erweitert mit Zeilen und Spalten mit Nullen wird,
     *  so dass die Kernel-Matrix zusätzlich multipliziert wird.
     */

    std::vector<std::vector<int>> featMap(imageM.size() - kernelM.size() + 1);

    //Row index
    for (size_t rowSize = 0; rowSize < featMap.size(); ++rowSize)
    {
        std::vector<int> tVector(imageM[0].size() - kernelM[0].size() + 1);
        featMap[rowSize] = tVector;
        // Column index
        for (size_t columnSize = 0; columnSize < tVector.size(); ++columnSize)
        {
            featMap[rowSize][columnSize] = conv64(transition(imageM, rowSize, columnSize, kernelM.size(), kernelM[0].size()), kernelM);
            //std::cout << featureMap[rowSize][columnSize] << " ";
        }
    }
    imageM = featMap;
}

int Channel::conv64(std::vector<std::vector<int>> iMatrix, std::vector<std::vector<int>> kernel)
{

    int bias = 0;
    int con = 0;

    for(int i = 0; i < iMatrix.size(); i++)
    {
        for(int j = 0; j < iMatrix[0].size(); j++)
        {
            /// Image * Kernel + Bias = Y-Matrix 32 x 20x20x20
            con += iMatrix[i][j] * kernel[i][j] + bias;
            //std::cout << "|" << s;
        }
        //std::cout << "\n";
     }
    return con;

}

void Channel::do_Kernel3()
{
    /* Diese Funktion wandelt die Matrix in einem Vektor um, so dass diese anhand eines Zwischenvektor's,
     * den Maxpool berechnet werden kann mithilfe der Größen der Matrizen von der Image-Matrix und Kernel- Matrix.
     */

    int pool = 2;
    int stride = 2;

    std::cout << "Matrixgroesse von ImageM: " << imageM.size() << "\n";

    std::vector<std::vector<int>> featureMap(int(imageM.size() / pool));

    std::cout << "Matrixgroesse von featureMap: " << featureMap.size() << "\n";

    for (size_t i = 0; i < featureMap.size(); ++i)
    {
        std::vector<int> tVector(int(imageM.size() / stride));
        //std::cout << "Groesse von tVector: " << tVector.size() << "\n";
        featureMap[i] = tVector;  // Zwischenvektor

        for (size_t j = 0; j < tVector.size(); ++j)
        {
            featureMap[i][j] = maxPooling64(transition(imageM, (kernelM.size() - 1) * i, (kernelM[0].size() - 1) * j,
                                          kernelM.size() - 1, kernelM[0].size() - 1));
        }
    }
    imageM = featureMap; //Die maximalen Werte werden in die feature Matrix gespeichert.
}


int Channel::maxPooling64(std::vector<std::vector<int>> imageM)
{
    /* Diese Funktion filtert die maximalen Werte der 32 x 20x20x20- Matrix, um diese Image-Matrix in
     * eine 32 x 10x10x10 umzuwandeln.
     * Die gefilterten maximalen Werte werden in ein Vektor gespeichert, so dass man die Werte wieder
     * Für die Backpropagation nutzen kann.
     */

    int s = 0;
    // Test:
    //std::cout << "Max Image hat die Groesse: " << imageM.size() << "\n";

    for (size_t i = 0; i < imageM.size()  ; ++i) // Zeile
    {
        for (size_t j = 0; j < imageM[0].size() ; ++j) // Spalte
        {
            // kernell[i][j] std::vector<std::vector<int>> kernell
            s = std::max(s, imageM[i][j]); // Das maximale Element wird ausgefiltert

        }
    }
    mask.push_back(s);
    return s;
}

void Channel::back_propConv64()
{

    std::cout << "\ntrace-Matrix size: " << trace.size() << "\n";
    for(int i = trace.size()-1; i > 0; i--)
    {
        std::cout << "|" << trace[i];
    }
    std::cout << "\n";
}


std::vector<std::vector<int>> Channel::rotationKernel()
{
    /* Hier wird die Kernel-Matrix rotiert
     */

    //kernelWeight = 0;
    std::cout << "\nKernel-Matrix Groesse: " << kernelM.size() << "\n";


    for(int row = 0; row < kernelM.size(); row++){
        for(int col = 0; col < kernelM.size(); col++)
        {
            std::reverse(kernelM.begin(), kernelM.end());
            kernelM[row][col] = kernelM[row][col];

            //kernelW[row].push_back(kW);
            std::cout << "|" << kernelWeight;
        }
        std::cout << "\n";
    }
    //kernelW.push_back(kW);
    return kernelM;
}

std::vector<std::vector<int>> Channel::actDervateLeakyReLU()
{
    for(int i = 0; i < imageM.size(); i++){
        for(int j = 0; j < imageM.size(); j++){
            // Leaky ReLU activation -> max(0.01*x, x)
            //Falls die Matrix größer als 0.01 * matrix ist:

            //imageM[i][j] = (image[i][j] > int(0.01 * image[i][j]) ? image[i][j] : int(0.01 * image[i][j]));

            if(imageM[i][j] > 0) /// Falls x > 0 ist dann bleibt der Wert von der Matrix erhalten.
            {
                imageM[i][j] = 1;
            }
            else /// Sonst wird max(x * 0.01, x) verrechnet und das Maximum benutzt.
            {
                imageM[i][j] = 0.01;
            }
            //imageM[i][j] = (0.01 * imageM[i][j]);
        }
    }
    return imageM;
}



/////////////////////////////////////////////////// Kernel /////////////////////////////////////////////////////////////

void Kernel::setKernel(std::vector<std::vector<int> > kernel){
    kernelMatrix = kernel;
}

std::vector<std::vector<int>> Kernel::resetKernel(){
    return kernelMatrix;
}


void Kernel::showKernel(){
    for(int i = 0; i < kernelMatrix.size(); i++){
        for(int j = 0; j < kernelMatrix.size(); j++){
            std::cout << "|" << kernelMatrix[i][j];
        }
        std::cout << "\n";
    }

}

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    // Aufruf der Klassen:
    srand(1);
    Channel conv;
    Kernel kernll;

    const size_t r1 = 20, c1 = 20; // Festlegung der Matrix Größe.
    int v1;
    std::vector<std::vector<int>> imageMatrix(r1);
    //vector<vector<vector<int>>> imageM3D(r1, vector<vector <int>>(c1, vector <int>(h1)));

    for (size_t i = 0; i < r1; ++i) {
        std::vector<int> vecTemp(c1);
        imageMatrix[i] = vecTemp;
        for (size_t j = 0; j < c1; ++j)
        {
            v1 = rand() % 100;//element always changes
            imageMatrix[i][j] = v1;

            //for(int k = 0; k < h1; k++){
                //imageM3D[i][j][k] = v1;
            //}
        }
    }
    // Calculation from the Kernel matrix
        const size_t r2 = 3, c2 = 3; // size von der matrix

        std::vector<std::vector<int>> kernelMatrix(r2);
        //vector<vector<vector<int>>> kernelM3D(r2, vector<vector <int>>(c2, vector <int>(h2)));

        for (size_t i = 0; i < r2; ++i)
        {
            std::vector<int> vecTemp(c2);
            kernelMatrix[i] = vecTemp;

            for (size_t j = 0; j < c2; ++j)
            {
                v1 = rand() % 3 - 1; //element always changes number from -1 to 1
                kernelMatrix[i][j] = v1;
                //for(size_t k = 0; k < h2; k++){
                    //kernelM3D[i][j][k] = v1; //Generate 3D-Kernel-matrix 3x3x3
                //}
            }
        }


    // Channel Image:
    conv.set(imageMatrix, kernelMatrix);
    std::cout << "\n28 x 20x20-Image Matrix:\n";
    conv.printMatrix();
    std::cout << "\n";

    // Kernel:
    kernll.setKernel(kernelMatrix);
    std::cout << "\n 28 x 3x3-Kernel Matrix:\n";
    kernll.showKernel();
    std::cout << "\n";

    // Transform Matrix in Vector:
    conv.add();
    conv.do_Kernel();

    // Convolution + Activation Leaky ReLU:
    std::cout << "\n...........................Convolution: Matrix ist im prozess...................\n";
    std::cout << "\n...................................Activation: Leaky ReLU..........................:\n";
    //conv.printMatrix();
    conv.activationLReLU();
    conv.printMatrix();

    std::cout << "\n.................................Max Pooling..........................................\n";
    std::cout << "\nChannel: 32 x 10x10\n";
    conv.do_Kernel2();
    conv.printMatrix();
    std::cout << "\n";

    //conv.Kernelclear();
    /////////////////////////////////////// Channel 64: /////////////////////////////////////////////////////
    std::cout << "..................................Channel 32 x 10x10...............................\n";
    kernll.setKernel(kernelMatrix);
    kernll.resetKernel();
    std::cout << "Kernel Matrix:\n";
    kernll.showKernel();

    std::cout << "\nConvolution 64 x 10x10x10:\n";
    conv.add();
    conv.connect();

    std::cout << "Activation and Leaky ReLU:\n";
    conv.activationLReLU();
    std::cout << "\n";

    conv.printMatrix();
    std::cout << "\n";

    std::cout << "\n.................................Max Pooling 64..........................................\n";
    std::cout << "\nChannel: 64 x 5x5\n";
    conv.do_Kernel3();
    conv.printMatrix();

    //conv.Imageclear();
    std::cout << "\n";
    std::cout << "\n...................................Backpropagation.......................................\n";
    conv.back_propConv64();
    std::cout << "\n";

    std::cout << "\n.....................Kernel W'............................................\n";
    conv.rotationKernel();
    return 0;
}
