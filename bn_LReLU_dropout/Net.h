#include <stdlib.h>
#include <vector>
#include <math.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <algorithm>

#include "Neuron.h"
#include "cnn.h"

class Net // Class declaration
{
public:
    Channel con; // Channel Klasse
    Kernel kern; // Kenrel Klasse
    std::vector<int> x_change;

  Net(const std::vector <size_t> &topology)
  {
    size_t layers_number = topology.size();
    for (size_t layer_num = 0; layer_num != layers_number; layer_num++) {
      net_layers.push_back(Layer());
      size_t outputs_number;

      if (layer_num == topology.size() - 1) {
        outputs_number = 0;
      }
      else {
        outputs_number = topology[layer_num + 1];
      }

      for (size_t neuron_num = 0; neuron_num != topology[layer_num]; neuron_num++) {
        net_layers.back().push_back(Neuron(outputs_number, neuron_num, topology[layer_num]));
      }
      net_layers.back().back().set_output_value(1);
    }

    batch_size = 1;
    batch_input = new float[batch_size*net_layers[0].size()];
    batch_qgp = new bool[batch_size];
    batch_nonzero = new bool[net_layers[0].size()];
    batch_active_neurons = new int[1];

    clear_gradients();

    i_event = 0;

    // stat
    clear_stat();
  }

  // Declare prototypes of functions
  void analyze();
  void back_prop(const std::vector <double> &target_values, size_t test, double epoch, int topology_type);
  void back_propConv64();
  void get_results(std::vector <double> &results_values) const;
  void loadQGP(std::string filename);
  void loadNQGP(std::string filename);
  void print_weights(std::string filename);

  void loadNQGPbatch(std::string filename, int batch_index, bool qgp);

  void fillNeurons(int index);
  void batchNormalization();

  double loss();
  void cnn(std::vector<std::vector<std::vector<int> > > input);

  void clear_gradients();

  void clear_stat()
  {
      // stat
      for( int i = 0; i < 28; i++ ) {
          for( int j = 0; j < 20; j++ ) {
              for( int k = 0; k < 20; k++ ) {
                  for( int l = 0; l < 20; l++ ) {
                      input_data_stat[i][j][k][l] = 0;
                      input_data_stat_n[i][j][k][l] = 0;
                  }
              }
          }
      }

      for( int i = 0; i < 20; i++ ) {
          ang_phi_stat[i] = 0;
          ang_teta_stat[i] = 0;
          momentum_stat[i] = 0;

          ang_phi_stat_n[i] = 0;
          ang_teta_stat_n[i] = 0;
          momentum_stat_n[i] = 0;
      }
  }

  void setBatch(int number) { batch_size = number; delete[] batch_input; batch_input = new float[batch_size*net_layers[0].size()];  delete[] batch_qgp; batch_qgp = new bool[batch_size];
                              delete [] batch_nonzero; batch_nonzero = new bool[net_layers[0].size()]; std::fill(batch_nonzero,batch_nonzero+net_layers[0].size(),0);}
  int getBatchSize() { return batch_size; }
  bool getBatch_qgp(size_t index) { return batch_qgp[index]; }
  float getBatch_input(size_t index) {return batch_input[index];}
  bool getBatch_nonzero(size_t index) {return batch_nonzero[index];}
  void setBatch_active_neurons() {n_active_neurons = 0; for(int i=0; i<net_layers[0].size(); i++) {if(batch_nonzero[i]) n_active_neurons++;}  delete[] batch_active_neurons; batch_active_neurons = new int[n_active_neurons];
                                  int index = 0; for(int i=0; i<net_layers[0].size(); i++){ if(batch_nonzero[i]) {batch_active_neurons[index] = i; index++; /*qDebug() << "batch_active_neurons #" << index-1<< "   " << batch_active_neurons[index-1];*/}} /*qDebug() << "N active neurons " << n_active_neurons;*/}

  void analyze_1hidden_layer();
  void change_topology(const std::vector <size_t> &topology);
  void print() {qDebug()<<"net_layers size: "<<net_layers.size(); for(int i=0;i<net_layers.size();i++){qDebug()<<"layer #"<<i<<" size: "<<net_layers[i].size();}}

  // stat
  void PrintStat() {
      for( int i = 0; i < 20; i++ ) {
          qDebug()<<" - i: "<<i<<";   phi: "<<ang_phi_stat[i]<<";   teta: "<<ang_teta_stat[i]<<";   momentum: "<<momentum_stat[i];
      }
  }

  int* getPhiStat() { return &ang_phi_stat[0]; }
  int* getTetStat() { return &ang_teta_stat[0]; }
  int* getMomStat() { return &momentum_stat[0]; }

  int* getPhiStat_n() { return &ang_phi_stat_n[0]; }
  int* getTetStat_n() { return &ang_teta_stat_n[0]; }
  int* getMomStat_n() { return &momentum_stat_n[0]; }
  //

  double current_speed;
  double total_net_error = 0;
  double net_error;

  float *batch_input;
  bool *batch_qgp;
  bool *batch_nonzero;
  int *batch_active_neurons;

  int batch_size;
  int n_active_neurons;

  int i_event;

  // stat
  int input_data_stat[28][20][20][20];
  int ang_phi_stat[20];
  int ang_teta_stat[20];
  int momentum_stat[20];

  int input_data_stat_n[28][20][20][20];
  int ang_phi_stat_n[20];
  int ang_teta_stat_n[20];
  int momentum_stat_n[20];

  std::vector<std::vector<std::vector<int>>> kernell;


  ~Net() // Destructor
  {delete [] batch_input; delete [] batch_qgp; delete[] batch_active_neurons;}
private:
  // Class members
  std::vector <Layer> net_layers;
};

//
// Direct mode of neural network
// copy input values from map/MNIST
// for-loop "analyze" for every neuron layers
//
inline void Net::analyze()
{
    qDebug() << "<analyze>";
  for (size_t layer_num = 1; layer_num != net_layers.size(); layer_num++) {
    Layer &prev_layer = net_layers[layer_num - 1];
    Layer &this_layer = net_layers[layer_num]; //----
    for (size_t n = 0; n != net_layers[layer_num].size(); n++) {
        qDebug() << " >>>>> analize(prev_layer) layer_num: " << layer_num << ";   n: " << n;
      net_layers[layer_num][n].analize(prev_layer);
    }
    qDebug() << " >>> layer_num: " << layer_num;
    for (size_t n = 0; n != net_layers[layer_num].size(); n++) {
        qDebug() << " >>>>> analizeSoftMax(this_layer) layer_num: " << layer_num << ";   n: " << n;
      net_layers[layer_num][n].analizeSoftMax(this_layer); //----
    }
  }
}

inline void Net::analyze_1hidden_layer()
{
    qDebug() << " --- Net::analyze_1hidden_layer() ---";
//  for (size_t n = 0; n != net_layers[1].size(); n++) {
//      net_layers[1][n].analize(net_layers[0]);
//  }
  for (size_t n = 0; n != net_layers[1].size(); n++) {
      net_layers[1][n].analizeBatch(net_layers[0], batch_active_neurons, n_active_neurons);
  }
  for (size_t n = 0; n != net_layers[1].size(); n++) {
    net_layers[1][n].analizeLReLU(); //----
    net_layers[1][n].dropout(0.5);
//    qDebug()<< " neuron " << n << "     " << net_layers[1][n].get_output_value();
  }
  for (size_t n = 0; n != net_layers[2].size(); n++) {
    net_layers[2][n].analize(net_layers[1]);
  }
  for (size_t n = 0; n != net_layers[2].size(); n++) {
    net_layers[2][n].analizeSoftMax(net_layers[2]); //----
  }
}

inline double Net::loss()
{
    double Loss = 0;
    for (size_t layer_num = 1; layer_num != net_layers.size(); layer_num++) {
      for (size_t n = 0; n != net_layers[layer_num].size(); n++) {
        Loss += net_layers[layer_num][n].get_loss_value();
      }
    }
    Loss = Loss/2;
    return Loss;
}

inline void Net::clear_gradients()
{
  for (size_t layer_num = 0; layer_num != net_layers.size()-1; layer_num++) {
    for (size_t n = 0; n != net_layers[layer_num].size(); n++) {
      Neuron &neuron = net_layers[layer_num][n];
//      neuron.set_gradient( 0 );
      for(size_t nn = 0; nn < net_layers[layer_num+1].size(); nn++) {
        neuron.set_loc_gradient( 0, nn );
      }
    }
  }

//  qDebug() << " --- clear last layer ---";
//  qDebug() << " -|- net_layers.size()-1: " << net_layers.size()-1;
//  qDebug() << " -|- net_layers[net_layers.size()-1].size(): " << net_layers[net_layers.size()-1].size();

  for (size_t n = 0; n != net_layers[net_layers.size()-1].size(); n++) {
//      qDebug() <<" ---|--- n: " << n;
    Neuron &neuron = net_layers[net_layers.size()-1][n];
    neuron.set_gradient( 0 );
  }
}


//
// back propagation
// calculate output gradients
// update input weights of neurons
//
inline void Net::back_prop(const std::vector <double> &target_values, size_t test, double epoch, int topology_type)
{
    qDebug() << "<back_prop>";
//  clear_gradients();
  Layer &output_layer = net_layers.back();

  for (size_t n = 0; n != output_layer.size(); n++) {
//    output_layer[n].calc_output_gradients(target_values[n]);
      qDebug() <<" ------- calc_output_gradientsSoftMax: n: "<<n;
    output_layer[n].calc_output_gradientsSoftMax(target_values[n], net_layers[net_layers.size()-1]); //---
  }
  for (size_t layer_num = net_layers.size() - 2; layer_num != 0; layer_num--) {
    Layer &hidden_layer = net_layers[layer_num];
    Layer &next_layer = net_layers[layer_num + 1];
    for (size_t n = 0; n != hidden_layer.size(); n++) {
      hidden_layer[n].calc_hidden_gradients(next_layer);
    }
  }

  if(!topology_type){
  for (size_t layer_num = net_layers.size() - 1; layer_num != 0; layer_num--) {
    Layer &layer = net_layers[layer_num];
    Layer &prev_layer = net_layers[layer_num - 1];
    for (size_t n = 0; n != layer.size(); n++) {
      layer[n].recalc_gradients(prev_layer);
      if( int(i_event+1) % batch_size == 0 ) {
          qDebug() << " --- --- --- --- --- correct weights";
//          layer[n].recalc_gradients(prev_layer);
          layer[n].update_input_weights(prev_layer,epoch);
          clear_gradients();
      }
    }
  }
  }
  else{
    for (size_t n = 0; n != net_layers[2].size(); n++) {
      net_layers[2][n].recalc_gradients(net_layers[1]);
          net_layers[2][n].update_input_weights(net_layers[1],epoch);
          clear_gradients();
    }
    for (size_t n = 0; n != net_layers[1].size(); n++) {
      net_layers[1][n].recalc_gradientsBatch(net_layers[0], batch_active_neurons, n_active_neurons);
          net_layers[1][n].update_input_weightsBatch(net_layers[0],epoch, batch_active_neurons, n_active_neurons);
          clear_gradients();
    }
  }

  i_event++;
}

//
// copy output values from net layers to result vector
//
inline void Net::get_results(std::vector <double> &results_values) const
{
  results_values.clear();
  for (size_t n = 0; n != net_layers.back().size(); n++) {
    results_values.push_back(net_layers.back()[n].get_output_value());
  }
}

//
// load QGP Data from file
//
inline void Net::loadQGP(std::string filename)
{
  std::ifstream QGP_file(filename.c_str());
//  int ss = 0;
  if (!QGP_file.eof()) {
      double input;
      for (size_t neuron_num = 0; neuron_num != net_layers[0].size(); neuron_num++) {
        Neuron &neuron = net_layers[0][neuron_num];
        QGP_file >> input;
        neuron.set_output_value(input);
//        if(input) ss++;//qDebug() << input << " " ;
      }
    }
//  qDebug() << " --- ss: " << ss;
  QGP_file.close();
}

//
// load NQGP Data from file
//
inline void Net::loadNQGP(std::string filename)
{
  std::ifstream NQGP_file(filename.c_str());
  if (!NQGP_file.eof()) {
      double input;
      for (size_t neuron_num = 0; neuron_num != net_layers[0].size(); neuron_num++) {
        Neuron &neuron = net_layers[0][neuron_num];
        NQGP_file >> input;
        neuron.set_output_value(input);
      }
    }
  NQGP_file.close();
}

inline void Net::loadNQGPbatch(std::string filename, int batch_index, bool qgp)
{
    qDebug() << " --- Net::loadNQGPbatch --- batch_index: " << batch_index;
  std::ifstream NQGP_file(filename.c_str());
  // stat
  int l_ind(0), k_ind(0), j_ind(0), i_ind(0);
  //
  if (!NQGP_file.eof()) {
      float input;
      for (size_t neuron_num = 0; neuron_num != net_layers[0].size(); neuron_num++) {
        NQGP_file >> input;
        // stat
        if( qgp ) {
          input_data_stat[i_ind][j_ind][k_ind][l_ind] += input;
          ang_teta_stat[l_ind] += input;
          ang_phi_stat[k_ind] += input;
          momentum_stat[j_ind] += input;
        } else {
          input_data_stat_n[i_ind][j_ind][k_ind][l_ind] += input;
          ang_teta_stat_n[l_ind] += input;
          ang_phi_stat_n[k_ind] += input;
          momentum_stat_n[j_ind] += input;
        }
//        if(input)qDebug()<<" - set input_data_stat["<<i_ind<<"]["<<j_ind<<"]["<<k_ind<<"]["<<l_ind<<"] += "<<input;
        l_ind++;
        if( l_ind == 20 ) {
          l_ind = 0;
          k_ind++;
          if( k_ind == 20 ) {
            k_ind = 0;
            j_ind++;
            if( j_ind == 20 ) {
              j_ind = 0;
              i_ind++;
            }
          }
        }
        //
//        batch_input[batch_index*batch_size+neuron_num] = short(input);
        batch_input[batch_index*net_layers[0].size()+neuron_num] = input;
//        qDebug() << " batch_input " << batch_input[batch_index*batch_size+neuron_num] << " double batch_input " << double(batch_input[batch_index*batch_size+neuron_num]);
        if(input) batch_nonzero[neuron_num] = 1;
//        if(input) qDebug()<<"       input: "<<input;
      }
    }
  NQGP_file.close();
  batch_qgp[batch_index] = qgp;
}

inline void Net::fillNeurons(int index)
{
    qDebug()<<" --- Net::fillNeurons --- index: "<<index;
    for (size_t neuron_num = 0; neuron_num != net_layers[0].size(); neuron_num++) {
        Neuron &neuron = net_layers[0][neuron_num];
        neuron.set_output_value(batch_input[index*net_layers[0].size() + neuron_num]);
//        if(batch_input[index*net_layers[0].size() + neuron_num]) qDebug()<<"       n: "<<neuron_num<<";   neuron.set_output_value: "<<batch_input[index*net_layers[0].size() + neuron_num]<<";   batch_ind: "<<index*net_layers[0].size() + neuron_num;
    }
}

inline void Net::batchNormalization()
{
    float epsilon = 0.001;
    for(size_t i = 0; i < net_layers[0].size(); i++){
        float summ = 0;
        for(int j = 0; j < batch_size; j++){

//            qDebug() << " batch_input " << batch_input[j*net_layers[0].size() + i] << " double batch_input " << double(batch_input[j*net_layers[0].size() + i]);

            summ += batch_input[j*net_layers[0].size() + i];
        }
        summ = summ / batch_size;

        float sigma = 0;
        for(int j = 0; j < batch_size; j++){
            sigma += pow((batch_input[j*net_layers[0].size() + i] - summ), 2);
        }
        sigma = sigma / batch_size;

        for(int j = 0; j < batch_size; j++){
            float temp = batch_input[j*net_layers[0].size() + i];
            batch_input[j*net_layers[0].size() + i] = (batch_input[j*net_layers[0].size() + i] - summ) / pow(sigma + epsilon, 0.5);
//            if(temp)
//                qDebug() << "j " << j << " temp " << temp << " batch_input " << batch_input[j*net_layers[0].size() + i];
        }

    }
}

inline void Net::change_topology(const std::vector<size_t> &topology)
{
  size_t layers_number = topology.size();
  net_layers.clear();

  for (size_t layer_num = 0; layer_num != layers_number; layer_num++) {
    net_layers.push_back(Layer());
    size_t outputs_number;

    if (layer_num == topology.size() - 1) {
      outputs_number = 0;
    }
    else {
      outputs_number = topology[layer_num + 1];
    }


    for (size_t neuron_num = 0; neuron_num != topology[layer_num]; neuron_num++) {
      net_layers.back().push_back(Neuron(outputs_number, neuron_num, topology[layer_num]));
//      if(layer_num < layers_number - 1){
//          net_layers.back().back().set_weights( net_layers.back().back().get_weights() * sqrt(2/double(topology[layer_num])) );
//          net_layers.back().back().set_prev_weights( net_layers.back().back().get_weights() );
////          qDebug() << " > neuron_num: "<<neuron_num<<";   set_weights: "<<net_layers.back().back().get_weights() * pow(1/double(topology[layer_num]), 0.5);
////          qDebug()<<" weight " << net_layers.back().back().get_weights();
////          qDebug()<<" 1/topology[layer_num+1] " << 1/double(topology[layer_num+1]);
//      }
    }

    net_layers.back().back().set_output_value(1);
  }
}

inline void Net::cnn(std::vector<std::vector<std::vector<int>>> input)
{
    /* Hier werden die Inputs-Daten mithilfe des Neuronalem Netzwerk in die
     * Convolution Neurales Netzwerk gefüllt.
     */



    int random, ran;
    //std::vector<std::vector<int>> kernell;
    qDebug() << ".....Net::cnn: Convolution Neural Net.......\n";
    for(size_t r = 0; r < input.size(); r++)
    {
        for(size_t c = 0; c < input.size(); c++)
        {
            for(size_t k = 0; k < input.size(); k++)
            {
                random = rand() % 100;
                input[r][c][k] = random;
            }
        }
    }

    for(int i = 0; i < 3; i++)
    {
        std::vector<std::vector<int>> cTemp(3, std::vector<int>(3));
        kernell[i] = cTemp;

        for(int j = 0; j < 3; j++)
        {
            for(int k = 0; k < 3; k++)
            {
                ran = rand() % 3 - 1;
                kernell[i][j][k] = ran;
                //std::cout << vec[i][j][k][l] << " ";

                }
        }
    }

    // Daten werden in die CNN Klasse übertragen.
    kern.setKernel(kernell);
    con.setMatrix(input, kernell);
}

inline void Net::back_propConv64()
{
    int sum = 0;
    con.actDervateLeakyReLU(); // Channel Matrix hat abhier ableitungs werte (1,0.01)
    con.weight_change(); // veränderung der weights wird in vec change_w gespeichert
    for(unsigned int i = 0; i < 64; i++) // für 64 Channel
    {
        for(unsigned int j = 0; j < con.imageM.size(); j++)
        {
            for(unsigned int k = 0; k < con.imageM.size(); k++)
            {
                for(unsigned int l = 0; l < con.imageM.size(); l++) // jkl für Dimension des Channels
                {
                    for(unsigned int x = 0; x < con.kernelM.size(); x++)
                    {
                        for(unsigned int y = 0; y < con.kernelM.size(); y++)
                        {
                            for(unsigned int z = 0; z < con.kernelM.size(); z++) // xyz für Dimension des Kernels
                            {
                            sum += con.imageM[j][k][l]*con.kernelW[x][y][z]; //summen Zeichen der dX_f berechnung
                            }
                        }
                    }
                }
            }
        }
        x_change.push_back(sum); // veränderung von X wird in x_change gespeichert evtl update analog zu weights
        sum=0;
    }
    for(unsigned int x = 0; x < con.kernelM.size(); x++)
    {
        for(unsigned int y = 0; y < con.kernelM.size(); y++)
        {
            for(unsigned int z = 0; z < con.kernelM.size(); z++)  // jeder Kernel wert wird mithilfe von vektor change_w geupdated
            {
                con.kernelM[x][y][z] =  con.change_w.pop_back() + con.kernelM[x][y][z];
            }

        }

    }
    int num = 0;
    for(unsigned int i = 0; i <64;i++){
        for(unsigned int j = 0; j < con.imageM.size(); j++)
        {
            for(unsigned int k = 0; k < con.imageM.size(); k++)
            {
                for(unsigned int l = 0; l < con.imageM.size(); l++) // summe der einzelnen Channel werte für bias update
                {
                    num += con.imageM[j][k][l];
                }
            }
        }
        // bias muss noch in das Net gemoved werden da man pro Filter 1 Bias hat und nicht pro kernel 1 bias 1 Filter besteht
        // besteht aus mehreren Kernels. auf channel ebene
        con.bias = con.bias + num;
        num = 0;
    }

}
