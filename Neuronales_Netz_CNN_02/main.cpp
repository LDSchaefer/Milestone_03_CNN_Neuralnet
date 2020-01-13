/*
Neuronales Netzwerk, welches aus mehreren Neuronen Besteht

Berechnung, Ausgabe und import von Dateien sind möglich.
*/
#include <iostream>
#include <vector>
#include "NeuralNet.h"


int main()
{
    srand(0);

    std::vector<unsigned> topology;
    int inp = 1;
    int count = 1;
    int neuron_count;
    std::cout << "Geben Sie die Topologie an:"<<std::endl;
    while(inp == 1){
        std::cout<< "Wie viele Neuronen soll der Layer " << count << " haben?"<<std::endl;
        std::cin >> neuron_count;
        topology.push_back(neuron_count - 1);
        std::cout <<"Soll ein weiterer Layer der Topologie hinzugefuegt werden? ja [1], nein [0]"<<std::endl;
        std::cin >> inp;
        count ++;
    }
    NeuralNet net(topology); //Verbindet die Topologie der Neuronen mit dem neuronalem Netzwerk

    return 0;
}

// Als Referenz: https://www.youtube.com/watch?v=KkwX7FkLfug
