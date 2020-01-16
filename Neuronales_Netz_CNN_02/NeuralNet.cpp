#include <iostream>
#include <vector>
#include <fstream>
#include "NeuralNet.h"
#include <string>
#include <stdlib.h>


NeuralNet::NeuralNet(const std::vector<unsigned> &topology){
    read_in();
    std::vector<double> results;
    int inp;
    unsigned long long num_layer = topology.size(); // Anzahl der Eingegebenen Werte für die Topologie (da frei wählbar)

    for(unsigned i = 0; i < num_layer ; i++){
        net_struct.push_back(Layer()); // Erstellt einen Layer(Vektor) in dem Objekt net_struct, welches ein Vektor von einem Vektor ist
        for(unsigned j = 0; j <= topology[i]; j++){ // Zugriff auf den eingegeben Wert um größe der Topologie abzufragen;
                                                    //<= um bias Neuron zu erstellen
            ///Wenn der Output Layer Anzahl des höchsten Output Layer ist
            ///unsigned out_Num = num_layer == topology.size() - 1 ?  0 : topology[num_layer + 1];
            ///out_Num
            net_struct.back().push_back(Neuron());  // Mit .back greifen wir auf das zuletzt erstellte Element von net_struct zu
                                                    // .push_back speichert dann in dieser Position genau eine Instanz von Neuron ab
            std::cout << " Neuron erstellt "<<std::endl;
        }
    }
    std::cout << "Wollen Sie Die Ausgabe fuer eine spezifische Eingabe in das Neurale Netz? Ja [1] Nein [0]";
    std::cin >> inp;
    std::cout << "" << std::endl;
    if(inp == 1){
        output_spec();
    }
    inp = 0;
    std::cout << "Wollen sie alle Outputs des Neuralen Netzwerks ausgeben? ja = 1, nein = 0" << std::endl;
    std::cin >> inp;

    if(inp == 1){
        output_all(results);
    }

    else if(inp == 0) {
        exit(0); //Falls der Benutzer nichts ausgeben möchte
    }
}

void NeuralNet::output_all(std::vector<double> &res_vals){ // Nimmt einen Vektor entgegen in dem die Resultate gespeichert werden sollen
    std::vector<float> res;
    for(unsigned int i=0;i<net_struct.size();i++){ // für jedes Layer
        for(unsigned int j=0;j<net_struct.at(i).size();j++){ // für jedes Neuron
            if(i==0){
                net_struct.at(i).at(j).first_step(input_vec);
            }
            else{
                net_struct.at(i).at(j).softMax(hidden_vec);
            }
            res.push_back(net_struct.at(i).at(j).sigma);
        }
        hidden_vec.swap(res);
        res.clear();
    }
    for(unsigned int i = 0; i < net_struct.back().size(); i++){ // wir gehen für jede Neurone des letzten Layers durch (Outputlayer)
        res_vals.push_back(net_struct.back()[i].sigma); // wir pushen die Resultate jeweils in einen Vektor mit den Resulataten der Neuronen
        std::cout << net_struct.back()[i].sigma << std::endl;
    }
}


void NeuralNet::output_spec(){

    int inp;

    std::vector<double> spec_inp;

    std::vector<unsigned int> temp_val;

    for(unsigned int i=0;i<net_struct.front().size();i++){
        std::cout << "geben Sie eine Wert fuer die Input Neurone "<< i+1 << " ein:";
        std::cin >> inp;
        input_vec.push_back(inp);
        std::cout << "" << std::endl;
    }
    output_all(spec_inp);
}

void NeuralNet::read_in(){
    std::vector<int> readin_val;
    int inp = 0;
    int numb= 0;
    int batch_num = 0;
    int flip;
    unsigned int num = 0;
    std::string u_input;
    std::ifstream file;
    std::string fileid;
    std::cout<<"single file [1] oder batch[0] oder kein Import [2] oder um die CNN [3] zu gehen: ";
    std::cin >> inp;
    if(inp == 0){
        std::cout<< "Wie viele Dateien sind in dem Batch?" ;
        std::cin >> batch_num;
        std::cout <<"" << std::endl;
        while(numb<batch_num){
            flip=  rand() % 2 + 1;
            if(flip == 1){
                fileid = "qgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr."+ std::to_string(numb) + "_event.dat";
            }
            else{
                fileid = "nqgp\\phsd50csr.auau.31.2gev.centr.0000phsd50csr.auau.31.2gev.centr."+ std::to_string(numb) + "_event.dat";
            }
            file.open(fileid);
            while (file >> num){
                    num = file.get();
                    readin_val.push_back(num);
            }
            input_vec.swap(readin_val);
            readin_val.clear();
            file.close();
            numb++;
        }
    }
    else if(inp == 1){
        std::cout<< "Geben Sie den Pfad an:";
        std::cin >> fileid;
        std::cout << "" << std::endl;
        file.open(fileid);
        while (file >> num){
                num = file.get();
                readin_val.push_back(num);
        }
        input_vec.swap(readin_val);
        readin_val.clear();
        file.close();
    }
    else if(inp == 2){
            return;
    }
    else if (inp == 3) {
        return NeuralNet::conv3D();
    }
    //if(net_struct.front().size()!= num){
    //    std::cout << "Topologie passt nicht zum Datenset";
    //    return;
    //}

    // Hier kommt Read In einer Datei hin siehe Milestone 1
    // Funktion soll sowohl Dateien als auch Batches eingelesen können -> überladen der Funktion für jeweils Batches und Dateien

}

void NeuralNet::backpropagation(){
    // Hier werden die erhaltenen Gewichtungen an die Neuronen/softmaxfunktion zurückgegeben

}


void NeuralNet::math_dist(){
    // Hier werden Winkel und Momentum für 4c berechnet (genaueres verständniss von softmax nötig)

}

// evtl. muss neuron_res in Neural Net berechnet werden

//Man benötig einen Kernel- und Channel-Matrix um diese zu multiplizieren
//Matrizen: kernel[3][3][3] * channel[22][22][22] = convY
void NeuralNet::conv3D()
{
    // Hier wird die Matrixmultiplikation berechnet, indem Y = X * W für i,j,k
    float matrix_X[5][5] = { 0 };
    int weight_m[3][3][3] = { 0 };
    float matrix_Y[5][5] = { 0 };


    //Eigentliche matrix:
    int n = 22;
    float conv_Y[22][22][22] = { 0 };

    for(int i = 0; i < 3; i++) // l = i - 1
    {
        for(int j = 0; j < 3; j++) // m = j - 1
        {
            matrix_X[i][j] = rand() % 9;
            //weight_m[i][j] = -1 + rand() % 2;

            for(int k = 0; k < 3; k++) // n = k - 1
            {
                //matrix_Y[i][j] += matrix_X[i][k] * weight_m[k][j];
                weight_m[i][j][k] = -1 + rand() % 3;
                std::cout << weight_m[i][j][k] << " ";

            }
            //std::cout << matrix_x[i][j][k] << " ";
        }
        std::cout << "\n";
    }






}
