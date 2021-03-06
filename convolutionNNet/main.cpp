#include <QCoreApplication>
#include <iostream>
#include "mainCNN.h"
#include <string>
#include <stdlib.h>
#include <fstream>

inline int Channel::read_in()
{
    //
    std::vector<std::vector<std::vector<int>>> readin_val;

    Channel con;
    int channel;
    int inp = 0;
    int count = 0;
    int numb= 0;
    int batch_num = 0;
    int flip;
    unsigned int num = 0;
    std::string u_input;
    std::ifstream file;
    std::string fileid;
    std::cout << "\nsingle file [1] oder batch[0] oder kein Import [2]:";
    std::cin >> inp;
    if(inp == 0)
    {
        //
        std::cout<< "\nGeben Sie ein Zahl (1 bis 2) ein: " ;
        std::cin >> flip;
        std::cout << "" << std::endl;

        while(flip < 80)
        {
            std::cout << ".....................................Prozess ist im Gang.......................................\n;";
            if(flip == 1)
            {

                fileid = "qgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr."+ std::to_string(8000) + "_event.dat";
                std::cout << "Daten werden in file gefuellt!!!...................count[" << count << "\n";
                //break;
                count++;

            }



            else
            {
                fileid = "nqgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr."+ std::to_string(numb) + "_event.dat";
            }


            file.open(fileid);
            std::cout << "\n.........................................Oeffne File.....................................\n";
            while (file >> num){ // solage ein Wert aus der  file gelesen werden kann ist num /= 0. Sind alle Werte eingelesen wird num 0 while bricht ab

                num = file.get();
                for(size_t channel = 0; channel < 28; channel++){
                    for(size_t i = 0; i < 20; i++){
                        for(size_t j = 0; j < 20; j++){
                            for(size_t k = 0; k < 20; k++){
                                readin_val[i][j].push_back(num);
                            }
                        }
                    }
                 }


                 input_vec.swap(readin_val); // swap hier falsch pushback evtl.
                 readin_val.clear();
                 file.close();
                 numb++;
            }

        }

    }

    else if(inp == 1){
        std::cout<< "Geben Sie den Pfad an (z.B. C:\\Users\\sade2\\OneDrive\\Desktop\\Uni3.Semester\\PRGPraktikum\\Milenstone_3):";
        std::cin >> fileid;
        std::cout << "" << std::endl;
        file.open(fileid);
        while (file >> num){
                num = file.get();
                for(size_t channel = 0; channel < 28; channel++){
                    for(size_t i = 0; i < 20; i++){
                        for(size_t j = 0; j < 20; j++){
                            for(size_t k = 0; k < 20; k++){
                                readin_val[i][j].push_back(num);
                            }
                        }
                    }
                }
        }
        input_vec.swap(readin_val);
        readin_val.clear();
        file.close();
    }

    else if(inp == 2){
            return 0;
    }
    int random, chan = 0;
    for(chan = 0; chan < 28; chan++){
        for(int i = 0; i < 3; i++){
            for(int j = 0; j < 3; j++){
                for(int k = 0; k < 3; k++){
                    random = rand() % 3 -1 ;
                    kernelll[i][j][k] = random;
                }
            }
        }
    }
    std::cout << "Möchten Sie gefuellt Daten in CNN berechnen? yes[0] or no[1]: " ;
    std::cin >> channel;
    if(channel == 0){
        std::cout << "\n..................Prozess zum CNN ist im gang................................\n";
        con.setMatrix(chan, input_vec, kernelll);
        con.printMatrix();
        con.add();
        std::cout << "\n";
        con.do_Kernel();

        // Convolution + Activation Leaky ReLU:
        std::cout << "\n...........................Convolution: Matrix ist im prozess...................\n";
        std::cout << "\n...................................Activation: Leaky ReLU..........................:\n";
        //conv.printMatrix();
        con.activationLReLU();
        con.printMatrix();

        std::cout << "\n.................................Max Pooling..........................................\n";
        std::cout << "\nChannel: 32 x 10x10x10\n";
        con.do_Kernel2();
        con.printMatrix();
        std::cout << "\n";

        //conv.Kernelclear();
        /////////////////////////////////////// Channel 64: /////////////////////////////////////////////////////
        std::cout << "\nConvolution 64 x 10x10x10:\n";
        con.add();
        con.connect();

        std::cout << "Activation and Leaky ReLU:\n";
        con.activationLReLU();
        std::cout << "\n";

        con.printMatrix();
        std::cout << "\n";

        std::cout << "\n.................................Max Pooling 64..........................................\n";
        std::cout << "\nChannel: 64 x 5x5\n";
        con.do_Kernel3();
        con.printMatrix();

        //conv.Imageclear();
        std::cout << "\n";
        std::cout << "\n...................................Backpropagation backpro_Maxpool64().......................................\n";
        //
        //conv.printMask();
        con.backpro_Maxpool64();
        std::cout << "\n";
        con.printMatrix();

        std::cout << "\nActivation and Leaky ReLU:\n";
        con.actDervateLeakyReLU();
        std::cout << "\n";
        con.printMatrix();
        std::cout << "\n";

    }

    // Hier kommt Read In einer Datei hin siehe Milestone 1
    // Funktion soll sowohl Dateien als auch Batches eingelesen können -> überladen der Funktion für jeweils Batches und Dateien



}



inline int Channel::setMatrix(int channelcnt, std::vector<std::vector<std::vector<int> > > image, std::vector<std::vector<std::vector<int> > > kernel){

        imageM = image;
        kernelM = kernel;
        kernelW = image;
        std::cout << "\n..................Channel couter:[" << channelcnt << "]................................\n";
        channelcnt++;

        return channelcnt;

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
    //std::cout << "rowIndex: " << rowIndex << ", colIndex: " << colIndex << ", kIndex: " << kIndex << "\n";

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
            for(size_t k = 0; k < imageM[0].size(); k++)
            {
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

    //std::cout << "Matrixgroesse von ImageM: " << imageM.size() << "\n";

    std::vector<std::vector<std::vector<int>>> featureMap(int(imageM.size() / maxPool));

    //std::cout << "Matrixgroesse von featureMap: " << featureMap.size() << "\n";

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
                    float(imageM[i][j][k] = 0.01);
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
/*
int Channel::backMax64(std::vector<std::vector<std::vector<int>>> image)
{
    //(2, std::vector<std::vector<int>>(2, std::vector<int>(2, (0,1))))
    //std::cout << "Image-Matrix size: " << image.size() << "\n";
    //std::vector<std::vector<std::vector<int>>> tempVector[2][2][2];
    int backtrace = 0;
    for(size_t row = 0; row < image.size(); row++){
        for(size_t col = 0; col < image.size(); col++){
            for(size_t k = 0; k < image.size(); k++)
            {
                backtrace = imageM[row][col][k];
                std::cout << "| " << imageM[row][col][k];
            }
        }
        std::cout << "\n";
    }
    return backtrace;

}

void Channel::backpro_Maxpool64()
{
    //std::cout << "do_Kenrel() funktioniert auch!!\n";
    std::vector<std::vector<std::vector<int>>> featureMap(imageM.size() - kernelM.size() + 3);
    //std::cout << "featureMap-Matrix size: "<< featureMap.size() << "\n";

    //Row index
    for (size_t rowSize = 0; rowSize < featureMap.size(); ++rowSize)
    {
        //std::vector<int>(imageM[0].size() - kernelM[0].size() + 1)
        std::vector<std::vector<int>> tVector(imageM.size() - kernelM.size() + 3);
        featureMap[rowSize] = tVector;
        //std::cout << "featureMap-Matrix size: "<< tVector.size() << "\n";
        // Column index

        for (size_t columnSize = 0; columnSize < featureMap.size(); ++columnSize)
        {
            std::vector<int> tVec(imageM.size() - kernelM.size() + 3);
            featureMap[rowSize][columnSize] = tVec;
            //std::cout << "featureMap-Matrix size: "<< tVec.size() << "\n";

            for(size_t kSize = 0; kSize < featureMap.size(); kSize++)
            {
                //std::cout << "featureMap["<<rowSize<<"]["<<columnSize<<"]["<<kSize<<"]" << "\n";
                featureMap[rowSize][columnSize][kSize] = backMax64(transition(imageM, (kernelM.size() + 1) * rowSize, (kernelM.size() + 1) * columnSize, (kernelM.size() + 1) * kSize,
                                                                              kernelM.size() - 1, kernelM.size() - 1, kernelM.size() - 1));
                //std::cout << featureMap[rowSize][columnSize][kSize] << " ";
            }
        }
        //std::cout << "\n";
    }
    imageM = featureMap;
}
*/
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


void Channel::backpro_Maxpool32()
{
    /// Du kannst die Funktion einfach aufrufen.
    /* Diese Funktion filtert quasi ebenfalls Sigma von der 10x10x10-Matrix und überträgt die Werte
     * die Zeile & Spalte auf die 20x20x20-Matrix, so dass es wieder seine Usprüngliche Matrix
     * berechnet wird.
     * Es ist keine Garantie, ob es richtig wieder in seine richtigen Reihenfolge gelangt.
     */

    std::vector<std::vector<std::vector<int>>> backMatrix32(imageM.size() + 10);
    // Die Matrix wird außerhalb des Randes mit 0 durch + 10 initialisiert und mit 0 gefüllt
    std::vector<std::vector<int>> backVec32(imageM.size() + 10, std::vector<int>(imageM[0].size() + 10, 0));


    backMatrix32[0] = backVec32;

    for (size_t i = 1; i < imageM.size() + 10; ++i)
    {
        backMatrix32[i] = backVec32;

        for (size_t j = 1; j < imageM.size() + 10; ++j)
        {
            //tempMatrix[i][j] = tempVector;

            for(size_t k = 1; k < imageM[0].size() + 10; ++k)
            {
                backMatrix32[i][j][k] = imageM[i - 1][j - 1][k - 1];
                //std::cout << "|" << tempMatrix[i][j][k];
            }
        }
        //std::cout << "\n";
    }
    backMatrix32[imageM.size() + 1] = backVec32;
    imageM = backMatrix32; // Ergebnis-Matrix
    std::cout << "Image-Matrix size: " << imageM.size() << "\n";
}

inline void Channel::backpro_Maxpool64(){


    // Hier wird das Ergebnis der Backpropagation der MaxPooling berechnet, dass mithilfe einer
    // Zwischen-Vektos gespeichert wird und diese auf die Image-Matrix 32x nxnxn
    // Übertagen wird.
    // Die Y_Image 5x5x5-Matrix wird zu einer 10x10x10-Matrix umgewandelt.

    std::vector<std::vector<std::vector<int>>> backMatrix(imageM.size() + 5);
    // Die Matrix wird außerhalb des Randes mit 0 durch +2 initialisiert und mit 0 gefüllt
    std::vector<std::vector<int>> backVector(imageM.size() + 5, std::vector<int>(imageM[0].size() + 5, 0));

    //std::vector<int> temp(imageM[0].size() + 2, 0)
   //(std::vector<int>(imageM[0].size() + 2, 0))

    //std::cout << "Image-Matrix size: " << tempMatrix.size() << "\n";
    backMatrix[0] = backVector;

    for (size_t i = 1; i < imageM.size() + 5; ++i)
    {
        backMatrix[i] = backVector;

        for (size_t j = 1; j < imageM.size() + 5; ++j)
        {
            //tempMatrix[i][j] = tempVector;

            for(size_t k = 1; k < imageM[0].size() + 5; ++k)
            {
                backMatrix[i][j][k] = imageM[i - 1][j - 1][k - 1];
                //std::cout << "|" << tempMatrix[i][j][k];
            }
        }
        //std::cout << "\n";
    }
    backMatrix[imageM.size() + 1] = backVector;
    imageM = backMatrix; // Ergebnis-Matrix
}

////// Testfunktion:

inline void Channel::printMask()
{
    //Gebe die Image-Matrix aus:
    std::vector<std::vector<std::vector<int>>> tempVector(2, std::vector<std::vector<int>>(2, std::vector<int>(2,0)));
    //std::cout << "Image-Matrix size: " << mask.size() << "\n";
    for(int i = 0; i < tempVector.size(); i++)
    {
        for(int j = 0; j < tempVector.size(); j++){
            for(int k = 0; k < tempVector.size(); k++){


                std::cout << tempVector[i][j][k] << " ";
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

    for(int kernelsize = 0; kernelsize < 28; kernelsize++)
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

    for(int chanelsize = 0; chanelsize < 28; chanelsize++)
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
        // Channel Image:
        conv.setMatrix(chanelsize, imageMatrix, kernelMatrix);
    }





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
    std::cout << "\n...................................Backpropagation backpro_Maxpool64().......................................\n";
    //
    //conv.printMask();
    conv.backpro_Maxpool64();
    std::cout << "\n";
    conv.printMatrix();

    std::cout << "\nActivation and Leaky ReLU:\n";
    conv.actDervateLeakyReLU();
    std::cout << "\n";
    conv.printMatrix();

    //conv.back_propConv64();
    //conv.backpro_Maxpool32();
    //std::cout << "\n";
    //conv.printMatrix();
    conv.read_in();
    return 0;
}
