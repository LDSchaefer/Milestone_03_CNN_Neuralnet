#ifndef NEUROALNET_H

#define NEUROALNET_H

#endif // NEURALNET_H

#include <vector>

#include <iostream>

#include "Neuron.h"

class NeuralNet{



public:

    std::vector<int> input_vec;

    std::vector<float> hidden_vec;



    // 2a



    NeuralNet(const std::vector<unsigned> &topology);



    // 2b



    void output_all(std::vector<double> &res_vals);



    // 2c



    void output_spec();



    // 2d



    void read_in();



    // 2e



    void backpropagation();



    // 2f



    void init_cnn(const std::vector<unsigned> &topology);



private:



    std::vector<Layer> net_struct; // Die Struktur unseres Netzes,



                                   //Zugriff auf einzelne Neuronen/Vektoren: net_struct[layer_num][neuron_num]







};
