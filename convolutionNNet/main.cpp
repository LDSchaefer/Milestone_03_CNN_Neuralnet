#include <QCoreApplication>
#include <iostream>
#include "mainCNN.h"





inline int Channel::setMatrix(std::vector<std::vector<std::vector<int> > > image, std::vector<std::vector<std::vector<int> > > kernel){

        imageM = image;
        kernelM = kernel;
        kernelW = image;

}

double Channel::biasUpdate()
{
    bias = 0;
    bias = rand() % 3 - 1;

    return bias;

}
inline void Channel::printMatrix()
{
    //Gebe die Image-Matrix aus:

    std::cout << "Image-Matrix size: " << imageM.size() << "\n";
    for(int i = 0; i < imageM.size(); i++){
        for(int j = 0; j < imageM.size(); j++){
            for(int k = 0; k < imageM.size(); k++){
                //
                std::cout << "|" << imageM[i][j][k];
                }
            }
        std::cout << "\n";
    }
}

inline void Channel::showKernel()
{
    // Ausgabe der Kernel-Matrix:

    std::cout << "\nKernel-Matrix size: " << kernelM.size() << "\n";
    for(int i = 0; i < kernelM.size(); i++){
        for(int j = 0; j < kernelM.size(); j++){
            for(int k = 0; k < kernelM.size(); k++){
                //
                std::cout << "|" << kernelM[i][j][k];
            }
        }
        std::cout << "\n";
    }
}

inline std::vector<std::vector<std::vector<int>>> Channel::get()
{
    return imageM; // Gebe die Image-Matrx zurück.
}

inline void Channel::imageclear()
{
    //imageM.clear();
    //kernelM.clear();
    for(size_t r1 = 0; r1 < imageM.size(); r1++){
        for(size_t c1 = 0; c1 < imageM.size(); c1++){
            for(size_t k1 = 0; k1 < imageM.size(); k1++){
                imageM[r1][c1][k1] = 0;
            }
        }
    }

    for(size_t r2 = 0; r2 < kernelM.size(); r2++){
        for(size_t c2 = 0; c2 < kernelM.size(); c2++){
            for(size_t k2 = 0; k2 < kernelM.size(); k2++){

                kernelM[r2][c2][k2] = 0;
            }
        }
    }
    std::cout << "\nDie Channel- und Kernel-Matrix wurde geleert!!!\n";
}

inline void Channel::add()
{
    //std::cout << "It works!!\n";
    // Hier wird das Ergebnis der Convolution berechnet, dass mithilfe einer
    // Zwischen-Parameter gespeichert wird und diese auf die Image-Matrix 32x nxnxn
    // Übertagen wird.

    std::vector<std::vector<std::vector<int>>> tempMatrix(imageM.size() + 2);
    // Die Matrix wird außerhalb des Randes mit 0 durch +2 initialisiert und mit 0 gefüllt
    std::vector<std::vector<int>> tempVector(imageM.size() + 2, std::vector<int>(imageM[0].size() + 2, 0));

    //std::vector<int> temp(imageM[0].size() + 2, 0)
   //(std::vector<int>(imageM[0].size() + 2, 0))

    //std::cout << "Image-Matrix size: " << imageM.size() << "\n";
    tempMatrix[0] = tempVector;

    for (size_t i = 1; i < imageM.size() + 1; ++i)
    {
        tempMatrix[i] = tempVector;

        for (size_t j = 1; j < imageM.size() + 1; ++j)
        {
            //tempMatrix[i][j] = tempVector;

            for(size_t k = 1; k < imageM.size() + 1; ++k)
            {
                tempMatrix[i][j][k] = imageM[i - 1][j - 1][k - 1];
                //std::cout << "|" << tempMatrix[i][j][k];
            }
        }
        //std::cout << "\n";
    }
    tempMatrix[imageM.size() + 1] = tempVector;
    imageM = tempMatrix; // Ergebnis-Matrix
}

//////////////////////////////////////////////// Convolution //////////////////////////////////////////////////////////


inline std::vector<std::vector<std::vector<int>>> Channel::transition(std::vector<std::vector<std::vector<int>>> inputMatrix,
                                         size_t rowIndex, size_t colIndex, size_t kIndex, size_t rowSize, size_t colSize, size_t kSize)
{

    /* Diese Funktion ist der Zwischenvektor, welches die Image-Matrix durch 0en
     * gefüllt und erweitert wird, sodass man mit diesem Vektor bzw.
     * Matrix die Convolution berechnen kann.
     */

    std::vector<std::vector<std::vector<int>>> partOfMatrix(rowSize);
    //std::cout << "rowSize: " << rowSize << ", colSize: " << colSize << ", kSize: " << kSize << "\n";

    for (size_t i = 0; i < rowSize; ++i) // Zeilen Größe des Vectors
    {
         std::vector<std::vector<int>> vecTemp(colSize, std::vector<int>(kSize));
         partOfMatrix[i] = vecTemp;

         for (size_t j = 0; j < colSize; ++j) // Spalten Größe der Matrix
         {
             //std::vector<int> tem(kSize);
             //partOfMatrix[i][j] = tem;

             for(size_t k = 0; k < kSize; k++)
             {
                 //std::cout << partOfMatrix[i][j][k] << " ";
                 partOfMatrix[i][j][k] = inputMatrix[rowIndex + i][colIndex + j][kIndex + k];
                 //std::cout << partOfMatrix[i][j][k] << " ";
             }
              //partOfMatrix[i][j][k][l] = inputMatrix[rowIndex + i][colIndex + j][kIndex + 1][lIndex + 1];
         }
         //std::cout << "\n";
    }
    //std::cout << "Input Groesse: " << partOfMatrix.size() << "\n";
    return partOfMatrix;
}

inline int Channel::convolution(std::vector<std::vector<std::vector<int>>> image, std::vector<std::vector<std::vector<int>>> kernell)
{

    /* Hier wird Convolution berechnet von Kernel und Channel.
     * Dabei hat die Channel-Matrix eine Grösse von 28 x 20x20x20 und die Kernel-Matrix
     * eine Grösse von 28/32 x 3x3x3.
     * Die Kernel-Matrix wird bei der Multiplikation der Channel-Matrix jeweils um 1 verschoben,
     * da ausserhalb des Channel-matrix Bereiches noch 0en hizugefügt wurden.
     */

    int bias = 0;
    int s = 0;

    for(int i = 0; i < image.size(); i++)
    {
        for(int j = 0; j < image.size(); j++)
        {
            for(int k = 0; k < image[0].size(); k++)
            {

                //std::cout << "|" << image[i][j][k];
                /// Image * Kernel + Bias = Y-Matrix 32 x 20x20x20
                s += image[i][j][k] * kernell[i][j][k] + biasUpdate();
                //std::cout << "|" << s;

            }

        }
        //std::cout << "\n";
     }
    return s;
}

inline std::vector<std::vector<std::vector<int>>> Channel::activationLReLU()
{

    /* Diese Funktion ist eine Aktivierungsfunktion, indem es jeden x-Wert überprüft, ob es
     * grösser 0 ist, dann bleibt der Wert vorhanden, aber wenn der Wert kleiner als 0 ist
     * dann wird x mit 0.01 multipliziert und auf 0 aufgerundet.
     */

    for(int i = 0; i < imageM.size(); i++)
    {
        for(int j = 0; j < imageM.size(); j++)
        {
            for(int k = 0; k < imageM.size(); k++)
            {
                // Leaky ReLU activation -> max(0.01*x, x)
                //Falls die Matrix größer als 0.01 * matrix ist:

                //imageM[i][j] = (image[i][j] > int(0.01 * image[i][j]) ? image[i][j] : int(0.01 * image[i][j]));

                if(imageM[i][j][k] > 0) /// Falls x > 0 ist dann bleibt der Wert von der Matrix erhalten.
                {
                    imageM[i][j][k] = imageM[i][j][k];
                }
                else /// Sonst wird max(x * 0.01, x) verrechnet und das Maximum benutzt.
                {
                    imageM[i][j][k] = std::max(int(0.01 * imageM[i][j][k]), imageM[i][j][k]);
                }
                    //imageM[i][j] = (0.01 * imageM[i][j]);
                }
            }
        }

    return imageM;
    ///wenn der Wert vor dem : nicht gleich 0 ist, dann ist es der Wert nach dem :
}

inline int Channel::maxPooling32(std::vector<std::vector<std::vector<int>>> imageM)
{
    /* Hier wird jeweils der maximale Wert gefiltert, so dass die Matrix
     * reduziert wird, z.B. eine 10x10x10-Matrix entsteht.
     */

    int s = 0;
    // Test:
    //std::cout << "Max Image hat die Groesse: " << imageM.size() << "\n";

    for (size_t i = 0; i < imageM.size()  ; ++i) // Zeile
    {
        for (size_t j = 0; j < imageM.size() ; ++j) // Spalte
        {
            for(size_t k = 0; k < imageM[0].size(); k++){
                // kernell[i][j] std::vector<std::vector<int>> kernell
                s = std::max(s, imageM[i][j][k]); // Das maximale Element wird ausgefiltert
            }
        }
    }
    trace.push_back(s);
    return s;
}

inline void Channel::do_Kernel()
{
    /* Diese Funktion berechnet die Goessen der Kernel- und Image-Matrix aus,
     * so dass diese für die Convolution-Berechnung verwendet werden.
     * Da die Image-Matrix und die Kernel-Matrix verschiedene Grössen haben,
     * wird die Image-Matrix mit Kernel-Matrix Grösse angepasst, damit
     * Multiplikation des Kernels sich immer um 1 verschieben kann.
     */

    //std::cout << "do_Kenrel() funktioniert auch!!\n";
    std::vector<std::vector<std::vector<int>>> featureMap(imageM.size() - kernelM.size() + 1);
    //std::cout << "featureMap-Matrix size: "<< featureMap.size() << "\n";

    //Row index
    for (size_t rowSize = 0; rowSize < featureMap.size(); ++rowSize)
    {
        //std::vector<int>(imageM[0].size() - kernelM[0].size() + 1)
        std::vector<std::vector<int>> tVector(imageM.size() - kernelM.size() + 1);
        featureMap[rowSize] = tVector;
        // Column index

        for (size_t columnSize = 0; columnSize < featureMap.size(); ++columnSize)
        {
            std::vector<int> tVec(imageM.size() - kernelM.size() + 1);
            featureMap[rowSize][columnSize] = tVec;

            for(size_t kSize = 0; kSize < featureMap.size(); kSize++)
            {
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
                featureMap[rowSize][columnSize][kSize] = convolution(transition(imageM, rowSize, columnSize, kSize, kernelM.size(), kernelM.size(), kernelM.size()), kernelM);
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
            }
        }
        //std::cout << "\n";
    }
    imageM = featureMap;
}

inline void Channel::do_Kernel2()
{

    int pool = 2;
    int stride = 2;

    std::cout << "Matrixgroesse von ImageM: " << imageM.size() << "\n";

    std::vector<std::vector<std::vector<int>>> featureMap(int(imageM.size() / pool));

    std::cout << "Matrixgroesse von featureMap: " << featureMap.size() << "\n";

    for (size_t i = 0; i < featureMap.size(); ++i)
    {
        std::vector<std::vector<int>> cVector(int(imageM.size() / stride));
        featureMap[i] = cVector;
        //std::cout << "Groesse von cVector: " << cVector.size() << "\n";

        for (size_t j = 0; j < cVector.size(); ++j)
        {
            std::vector<int> tVector(int(imageM.size() / stride));
            //std::cout << "Groesse von tVector: " << tVector.size() << "\n";
            featureMap[i][j] = tVector;  // Zwischenvektor

            for(size_t k = 0; k < tVector.size(); k++)
            {
                //std::cout << "| " << featureMap[i][j][k];
                featureMap[i][j][k] = maxPooling32(transition(imageM, (kernelM.size() - 1) * i, (kernelM.size() - 1) * j, (kernelM.size() - 1) * k,
                                              kernelM.size() - 1, kernelM.size() - 1, kernelM.size() - 1));

                //std::cout << "|" << featureMap[i][j][k];
            }
        }
        //std::cout << "\n";
    }
    imageM = featureMap; //Die maximalen Werte werden in die feature Matrix gespeichert.
}

//////////////////////////////////// Convolution und MaxPooling Teil 2: ////////////////////////////////////////////////

inline int Channel::conv64(std::vector<std::vector<std::vector<int> > > image, std::vector<std::vector<std::vector<int> > > kernell)
{
    /* Hier wird Convolution berechnet von Kernel und Channel.
     * Dabei hat die Channel-Matrix eine Grösse von 32x10x10x10 und die Kernel-Matrix
     * eine Grösse von 32x3x3x3.
     * Die Kernel-Matrix wird bei der Multiplikation der Channel-Matrix jeweils um 1 verschoben,
     * da ausserhalb des Channel-matrix Bereiches noch 0en hizugefügt wurden.
     */

    //int bias = 0;
    int sum64 = 0;

    for(int i = 0; i < image.size(); i++)
    {
        for(int j = 0; j < image.size(); j++)
        {
            for(int k = 0; k < image[0].size(); k++)
            {

                //std::cout << "|" << image[i][j][k];
                /// Image * Kernel + Bias = Y-Matrix 32 x 10x10x10
                sum64 += image[i][j][k] * kernell[i][j][k] + biasUpdate();
                //std::cout << "|" << s;

            }

        }
        //std::cout << "\n";
     }
    return sum64;
}

inline int Channel::maxPooling64(std::vector<std::vector<std::vector<int> > > image)
{

    /* Hier wird jeweils der maximale Wert gefiltert, so dass die Matrix
     * reduziert wird, z.B. eine 10x10x10-Matrix wird in eine 5x5x5-Matrix
     * umgewandelt.
     */

    int s = 0;
    // Test:
    //std::cout << "Max Image hat die Groesse: " << imageM.size() << "\n";

    for (size_t i = 0; i < imageM.size()  ; ++i) // Zeile
    {
        for (size_t j = 0; j < imageM.size() ; ++j) // Spalte
        {
            for(size_t k = 0; k < imageM[0].size(); k++){
                // kernell[i][j] std::vector<std::vector<int>> kernell
                s = std::max(s, imageM[i][j][k]); // Das maximale Element wird ausgefiltert
                mask[i][j].push_back(s);
            }
        }
    }

    return s;
}

inline void Channel::connect()
{
    /* D
     * Hier wird die 2. Convolution an der 10x10x10-Matrix durchgeführt, indem
     * die Image-Matrix erweitert mit Zeilen und Spalten mit Nullen wird,
     *  so dass die Kernel-Matrix zusätzlich multipliziert wird.
     */

    //std::cout << "do_Kenrel() funktioniert auch!!\n";
    std::vector<std::vector<std::vector<int>>> featureMap(imageM.size() - kernelM.size() + 1);
    //std::cout << "featureMap-Matrix size: "<< featureMap.size() << "\n";

    //Row index
    for (size_t rowSize = 0; rowSize < featureMap.size(); ++rowSize)
    {
        //std::vector<int>(imageM[0].size() - kernelM[0].size() + 1)
        std::vector<std::vector<int>> tVector(imageM.size() - kernelM.size() + 1);
        featureMap[rowSize] = tVector;
        // Column index

        for (size_t columnSize = 0; columnSize < featureMap.size(); ++columnSize)
        {
            std::vector<int> tVec(imageM.size() - kernelM.size() + 1);
            featureMap[rowSize][columnSize] = tVec;

            for(size_t kSize = 0; kSize < featureMap.size(); kSize++)
            {
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
                featureMap[rowSize][columnSize][kSize] = conv64(transition(imageM, rowSize, columnSize, kSize, kernelM.size(), kernelM.size(), kernelM.size()), kernelM);
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
            }
        }
        //std::cout << "\n";
    }
    imageM = featureMap;
}

inline void Channel::do_Kernel3()
{
    /* Hier werden jeweils die Groessen der Kernel- und Channel-Matrix gespeichert,
     * so dass die Image-Matrix in einem Vektor umgewandelt ist, so dass man
     * jeweils aus 2x2x2 das Maximum filtert.
     * Diese Funktion wandelt die Matrix in einem Vektor um, so dass diese anhand eines Zwischenvektor's,
     * den Maxpool berechnet werden kann mithilfe der Größen der Matrizen von der Image-Matrix und Kernel- Matrix.
     */

    int maxPool = 2;
    int stride = 2;

    std::cout << "Matrixgroesse von ImageM: " << imageM.size() << "\n";

    std::vector<std::vector<std::vector<int>>> featureMap(int(imageM.size() / maxPool));

    std::cout << "Matrixgroesse von featureMap: " << featureMap.size() << "\n";

    for (size_t i = 0; i < featureMap.size(); ++i)
    {
        std::vector<std::vector<int>> cVector(int(imageM.size() / stride));
        featureMap[i] = cVector;
        //std::cout << "Groesse von cVector: " << cVector.size() << "\n";

        for (size_t j = 0; j < cVector.size(); ++j)
        {
            std::vector<int> tVector(int(imageM.size() / stride));
            //std::cout << "Groesse von tVector: " << tVector.size() << "\n";
            featureMap[i][j] = tVector;  // Zwischenvektor

            for(size_t k = 0; k < tVector.size(); k++)
            {
                //std::cout << "| " << featureMap[i][j][k];
                featureMap[i][j][k] = maxPooling32(transition(imageM, (kernelM.size() - 1) * i, (kernelM.size() - 1) * j, (kernelM.size() - 1) * k,
                                              kernelM.size() - 1, kernelM.size() - 1, kernelM.size() - 1));

                //std::cout << "|" << featureMap[i][j][k];
            }
        }
        //std::cout << "\n";
    }
    imageM = featureMap; //Die maximalen Werte werden in die feature Matrix gespeichert.

}


inline std::vector<std::vector<std::vector<int>>> Channel::actDervateLeakyReLU()
{
    /* Diese Funktion ist die Leaky ReLU Ableitungsfunktion, welches für die
     * Backpropagation verwendet wird.
     */

    for(int i = 0; i < imageM.size(); i++)
    {
        for(int j = 0; j < imageM.size(); j++)
        {
            for(int k = 0; k < imageM.size(); k++)
            {
                // Leaky ReLU activation -> max(0.01*x, x)
                //Falls die Matrix größer als 0.01 * matrix ist:

                //imageM[i][j] = (image[i][j] > int(0.01 * image[i][j]) ? image[i][j] : int(0.01 * image[i][j]));

                if(imageM[i][j][k] > 0) /// Falls x > 0 ist dann bleibt der Wert von der Matrix erhalten.
                {
                    imageM[i][j][k] = 1;
                }
                else /// Sonst wird max(x * 0.01, x) verrechnet und das Maximum benutzt.
                {
                    imageM[i][j][k] = 0.01;
                }
            //imageM[i][j] = (0.01 * imageM[i][j]);
            }
        }
    }
    return imageM;
}

inline void Channel::weight_change(){
    for(unsigned int i = 0; i < o_matrix.size(); i++){
        for(unsigned int j = 0; j < o_matrix.size(); j++){
            for(unsigned int k = 0;k < o_matrix.size(); k++){

                change_w.push_back(o_matrix[i + 1][j + 1][k + 1] * imageM[i][j][k]); //+1 bei o_matrix wegen 0 rand
                std::reverse(change_w.begin(), change_w.end());
                }
            }
        }
    //std::reverse(change_w.begin(), change_w.end());
}

inline int Channel::rotationKernel()
{
    /* Hier wird die Kernel-Matrix rotiert
     */

    kernelWeight = 0;
    kernelW = kernelM;
    for(unsigned int row = 0; row < kernelM.size(); row++)
    {
        for(unsigned int col = 0; col < kernelM.size(); col++)
        {
            for(unsigned int k = 0; k < kernelM.size(); k++)
            {
                std::reverse(kernelW.begin(), kernelW.end());

                kernelWeight = kernelW[row][col][k];
                kernelW[row][col][k] = kernelW[row][col][k];
                std::cout << "|" << kernelW[row][col][k];
            }


            //kernelW[row].push_back(kW);
        }
        std::cout << "\n";
    }
    //kernelW.push_back(kW);
    return kernelWeight;
}

int Channel::backMax64(std::vector<std::vector<std::vector<int> > > image)
{
    for(int i = 0; i < image.size(); i++)
    {
        for(int j = 0; j < image.size(); j++)
        {
            for(int k = 0; k < image[0].size(); k++)
            {

                //std::cout << "|" << image[i][j][k];
            }
        }
        //std::cout << "\n";
    }


}

void Channel::back_propConv32(){

}

void Channel::back_propConv64()
{

    int sum = 0;
    actDervateLeakyReLU(); // Channel Matrix hat abhier ableitungs werte (1,0.01)
    weight_change(); // veränderung der weights wird in vec change_w gespeichert

    for(unsigned int i = 0; i < 64; i++) // für 64 Channel
    {
        for(unsigned int j = 0; j < imageM.size(); j++)
        {
            for(unsigned int k = 0; k < imageM.size(); k++)
            {
                for(unsigned int l = 0; l < imageM.size(); l++) // jkl für Dimension des Channels
                {
                    for(unsigned int x = 0; x < kernelM.size(); x++)
                    {
                        for(unsigned int y = 0; y < kernelM.size(); y++)
                        {
                            for(unsigned int z = 0; z < kernelM.size(); z++) // xyz für Dimension des Kernels
                            {
                            sum += imageM[j][k][l] * kernelW[x][y][z]; //summen Zeichen der dX_f berechnung
                            //std::cout << "|" << sum;
                            }
                        }
                    }
                }
            }
        }
        x_change.push_back(sum); // veränderung von X wird in x_change gespeichert evtl update analog zu weights
        sum = 0;
    }
    for(unsigned int x = 0; x < kernelM.size(); x++)
    {
        for(unsigned int y = 0; y < kernelM.size(); y++)
        {
            for(unsigned int z = 0; z <kernelM.size(); z++)  // jeder Kernel wert wird mithilfe von vektor change_w geupdated
            {
                kernelM[x][y][z] =  change_w[x] + kernelM[x][y][z];
            }

        }
        change_w.pop_back();

    }

    int num = 0;
    for(unsigned int i = 0; i < 64;i++){
        for(unsigned int j = 0; j < imageM.size(); j++)
        {
            for(unsigned int k = 0; k < imageM.size(); k++)
            {
                for(unsigned int l = 0; l < imageM.size(); l++) // summe der einzelnen Channel werte für bias update
                {
                    num += imageM[j][k][l];
                }
            }
        }
        // bias muss noch in das Net gemoved werden da man pro Filter 1 Bias hat und nicht pro kernel 1 bias 1 Filter besteht
        // besteht aus mehreren Kernels. auf channel ebene
        bias = bias + num;
        num = 0;
    }
}

void Channel::backpro_Maxpool64()
{
    // 5x5x5 => 10x10x10
    //sigma => sigma * dY

    //std::cout << "do_Kenrel() funktioniert auch!!\n";
    std::vector<std::vector<std::vector<int>>> featureMap(imageM.size() - kernelM.size() + 1);
    //std::cout << "featureMap-Matrix size: "<< featureMap.size() << "\n";

    //Row index
    for (size_t rowSize = 0; rowSize < featureMap.size(); ++rowSize)
    {
        //std::vector<int>(imageM[0].size() - kernelM[0].size() + 1)
        std::vector<std::vector<int>> tVector(imageM.size() - kernelM.size() + 1);
        featureMap[rowSize] = tVector;
        // Column index

        for (size_t columnSize = 0; columnSize < featureMap.size(); ++columnSize)
        {
            std::vector<int> tVec(imageM.size() - kernelM.size() + 1);
            featureMap[rowSize][columnSize] = tVec;

            for(size_t kSize = 0; kSize < featureMap.size(); kSize++)
            {
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
                featureMap[rowSize][columnSize][kSize] = backMax64(transition(imageM, rowSize, columnSize, kSize, kernelM.size(), kernelM.size(), kernelM.size()));
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
            }
        }
        //std::cout << "\n";
    }
    imageM = featureMap;

}

void Channel::backpro_Maxpool32(){

}

inline void Channel::backAdd(){

    //std::cout << "It works!!\n";
    // Hier wird das Ergebnis der Convolution berechnet, dass mithilfe einer
    // Zwischen-Parameter gespeichert wird und diese auf die Image-Matrix 32x nxnxn
    // Übertagen wird.

    std::vector<std::vector<std::vector<int>>> tempMatrix(imageM.size() + 2);
    // Die Matrix wird außerhalb des Randes mit 0 durch +2 initialisiert und mit 0 gefüllt
    std::vector<std::vector<int>> tempVector(imageM.size() + 2, std::vector<int>(imageM[0].size() + 2, 0));

    //std::vector<int> temp(imageM[0].size() + 2, 0)
   //(std::vector<int>(imageM[0].size() + 2, 0))

    //std::cout << "Image-Matrix size: " << imageM.size() << "\n";
    tempMatrix[0] = tempVector;

    for (size_t i = 1; i < imageM.size() + 1; ++i)
    {
        tempMatrix[i] = tempVector;

        for (size_t j = 1; j < imageM.size() + 1; ++j)
        {
            //tempMatrix[i][j] = tempVector;

            for(size_t k = 1; k < imageM[0].size() + 1; ++k)
            {
                tempMatrix[i][j][k] = imageM[i - 1][j - 1][k - 1];
                //std::cout << "|" << tempMatrix[i][j][k];
            }
        }
        //std::cout << "\n";
    }
    tempMatrix[imageM.size() + 1] = tempVector;
    imageM = tempMatrix; // Ergebnis-Matrix
}

inline void Channel::printMask()
{
    //Gebe die Image-Matrix aus:

    std::cout << "Image-Matrix size: " << mask.size() << "\n";
    for(int i = 0; i < mask.size(); i++){
        for(int j = 0; j < mask.size(); j++){
            for(int k = 0; k < mask.size(); k++){
                //
                std::cout << "|" << mask[i][j][k];
                }
            }
        std::cout << "\n";
    }
}
//////////////////////////////////////////// Kernel: ////////////////////////////////////////////////////////////

inline void Kernel::setKernel(std::vector<std::vector<std::vector<int> > > kernel)
{
    kernelM = kernel;
}

inline std::vector<std::vector<std::vector<int>>> Kernel::resetKernel()
{

    return kernelM;
}

inline void Kernel::showKernel(){
    // Ausgabe der Kernel-Matrix:

    std::cout << "\nKernel-Matrix size: " << kernelM.size() << "\n";
    for(int i = 0; i < kernelM.size(); i++){
        for(int j = 0; j < kernelM.size(); j++){
            for(int k = 0; k < kernelM.size(); k++){
                //
                std::cout << "|" << kernelM[i][j][k];
            }
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
    Kernel kern;

    std::cout << "\n...........................................Funktionseingang: .....................................\n";

    const int row = 20, col = 20, k1 = 20;
    const int r2 = 3, c2 = 3, k2 = 3;
    int random, ran;

    std::vector<std::vector<std::vector<int>>> imageMatrix(row, std::vector<std::vector<int>>(col, std::vector<int>(k1)));

    std::vector<std::vector<std::vector<int>>> kernelMatrix(r2, (c2, std::vector<std::vector<int>>(k2)));

    //std::cout << "\nKernel: " << row << "x" << col << "x" << k1 << "-Matrix:\n";

    for(int size = 0; size < 28; size++)
    {
        for(int i = 0; i < row; i++)
        {
            std::vector<std::vector<int>> cTemp(col, std::vector<int>(k1));
            imageMatrix[i] = cTemp;

            for(int j = 0; j < col; j++)
            {
                for(int k = 0; k < k1; k++)
                {
                    random = rand() % 10 - 1;
                     imageMatrix[i][j][k] = random;

                        //std::cout << vec[i][j][k][l] << " ";

                }
            }
        }
    }

    for(int size = 0; size < 28; size++)
    {
        for(int i = 0; i < r2; i++)
        {
            std::vector<std::vector<int>> cTemp(col, std::vector<int>(k1));
            kernelMatrix[i] = cTemp;

            for(int j = 0; j < c2; j++)
            {
                for(int k = 0; k < k2; k++)
                {
                    ran = rand() % 3 - 1;
                    kernelMatrix[i][j][k] = ran;
                    //std::cout << vec[i][j][k][l] << " ";

                    }
            }
        }
    }


    // Channel Image:
    conv.setMatrix(imageMatrix, kernelMatrix);
    std::cout << "\n28 x 20x20x20-Image Matrix:\n";
    conv.printMatrix();
    std::cout << "\n";

    // Kernel:
    kern.setKernel(kernelMatrix);
    std::cout << "\n 28 x 3x3-Kernel Matrix:\n";
    kern.showKernel();
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
    std::cout << "\nChannel: 32 x 10x10x10\n";
    conv.do_Kernel2();
    conv.printMatrix();
    std::cout << "\n";

    //conv.Kernelclear();
    /////////////////////////////////////// Channel 64: /////////////////////////////////////////////////////
    std::cout << "..................................Channel 32 x 10x10x10...............................\n";
    kern.setKernel(kernelMatrix);
    kern.resetKernel();
    std::cout << "Kernel Matrix:\n";
    kern.showKernel();

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
    //conv.backpro_Maxpool64();
    conv.backAdd();
    conv.backpro_Maxpool64();
    conv.printMatrix();
    //std::cout << "\n";

    //std::cout << "\n.....................Kernel W'............................................\n";
    //conv.rotationKernel();
    return 0;
}
