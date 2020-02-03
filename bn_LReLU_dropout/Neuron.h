#include <stdlib.h>
#include <vector>
#include <math.h>
#include <QDebug>


using std::vector;

class Neuron;

typedef std::vector<Neuron> Layer;

class Neuron // Class declaration
{
public:
  struct connection {
    double prev_weight;
    double weight;
    double change_weight = 0;
    double vdw = 0;
    double sdw = 0;
  };

  double eta = 0.15;
  std::vector <connection> output_weights;

  std::vector <double> loc_grad;

  Neuron(size_t outputs_num, size_t number) // Parameterized Constructor
  {
    for (size_t c = 0; c != outputs_num; c++) {
      output_weights.push_back(connection());
      output_weights.back().weight =/*0;*/ random_weight();
      output_weights.back().prev_weight = output_weights.back().weight;

      loc_grad.push_back(0);
    }
    neuron_number = number;
  }

  Neuron(size_t outputs_num, size_t number , size_t n_neurons) // Parameterized Constructor
  {
    for (size_t c = 0; c != outputs_num; c++) {
      output_weights.push_back(connection());
      output_weights.back().weight = random_weight() * sqrt(2/double(n_neurons));
      output_weights.back().prev_weight = output_weights.back().weight;

      loc_grad.push_back(0);
    }
    neuron_number = number;
  }

  void set_output_value(double value) { output_value = value; }
  double get_output_value() const { return output_value; }
  void set_sum_value(double value) { sum_value = value; } //----
  double get_sum_value() const { return sum_value; } //----
  double get_loss_value() const { return loss_value; } //----

  void analize(const Layer &prev_layer);
  void calc_output_gradients(double target_value);
  void calc_hidden_gradients(const Layer &next_layer);
  void update_input_weights(Layer &prev_layer, double epoch);
  void get_neuron_weights(std::vector <double> &neuron_weights);
  void analizeSoftMax(Layer &this_layer); //----
  void calc_output_gradientsSoftMax(double target_value, const Layer &this_layer); //---
  void analizeLReLU(); //----
  void dropout(double probability);
  double get_weights() { return output_weights.back().weight; };
  double set_weights(double w) { output_weights.back().weight = w; };
  double get_prev_weights() { return output_weights.back().prev_weight; };
  double set_prev_weights(double w) { output_weights.back().prev_weight = w; };

  void recalc_gradients(Layer &prev_layer);
  void set_gradient(double g) { gradient = g; };
  void add_gradient(double g) { gradient += g; };
  double get_gradient() { return gradient; };

  void set_loc_gradient(double g, int i) { loc_grad[i] = g; };
  void add_loc_gradient(double g, int i) { loc_grad[i] += g; };
  double get_loc_gradient(int i) { return loc_grad[i]; };

  void analizeBatch(const Layer &prev_layer, int *batch_active_neurons, int n_active_neurons);
  void recalc_gradientsBatch(Layer &prev_layer, int *batch_active_neurons, int n_active_neurons);
  void update_input_weightsBatch(Layer &prev_layer, double epoch, int *batch_active_neurons, int n_active_neurons);



  ~Neuron()
  {}
private:
  size_t neuron_number;
  double output_value;
  double gradient;
  double sum_value; //----
  double loss_value;

  static double random_weight() { return rand() / double(RAND_MAX)/* - 0.5*/; }
  static double activation_function(double x) { return 1 / (1 + exp(-x)); }
  static double activation_function_derivative(double x) { return activation_function(x) * (1 - activation_function(x)); }

  double activation_functionLReLU_derivative(double x) { if(x < 0) return negativeSlop; return 1;}


  double sum_weight_gradients(const Layer &next_layer) const;
  double beta1 = 0.9;
  double beta2 = 0.999;
  double epsilon = 1e-8;
  double learningRate = 0.01;
  double negativeSlop = 1e-2;
//  double learningRate = 0.0001;
//  double learningRate = 0.00001;

};


inline void Neuron::analize(const Layer &prev_layer)
{
  double sum = 0;
  for (size_t n = 0; n != prev_layer.size(); n++) {
    sum += prev_layer[n].get_output_value() * prev_layer[n].output_weights[neuron_number].weight;
//    qDebug() <<" ------- n: "<<n<<";   get_output_value: "<<prev_layer[n].get_output_value()<<";   neuron_number: "<<neuron_number<<";   output_weights: "<<prev_layer[n].output_weights[neuron_number].weight;
  }
//  output_value = activation_function(sum);
  sum_value = sum; //----
  qDebug() << "<Neuron::analize> sum_value: " << sum_value;
}

inline void Neuron::analizeBatch(const Layer &prev_layer, int *batch_active_neurons, int n_active_neurons)
{
    double sum = 0;
    for (size_t n = 0; n != n_active_neurons; n++) {
      sum += prev_layer[batch_active_neurons[n]].get_output_value() * prev_layer[batch_active_neurons[n]].output_weights[neuron_number].weight;
//      if(prev_layer[batch_active_neurons[n]].get_output_value()) qDebug()<<"> sum: "<<sum<<";   get_output_value: "<<prev_layer[batch_active_neurons[n]].get_output_value()<<";   output_weights: "<<prev_layer[batch_active_neurons[n]].output_weights[neuron_number].weight;
    }
    sum_value = sum; //----
}
inline void Neuron::analizeSoftMax(Layer &this_layer) //----
{
    qDebug() << "<analizeSoftMax>";
    // ---
    double min_v = this_layer[0].get_sum_value();
    for (size_t n = 0; n != this_layer.size(); n++) {
      if( this_layer[n].get_sum_value() < min_v ) min_v = this_layer[n].get_sum_value();
    }
    for (size_t n = 0; n != this_layer.size(); n++) {
      this_layer[n].set_sum_value(this_layer[n].get_sum_value()-min_v);
    }
    // ---
  double sum = 0;
  for (size_t n = 0; n != this_layer.size(); n++) {
    sum += exp(this_layer[n].get_sum_value());
    qDebug() << " this_layer[n].get_sum_value() " << this_layer[n].get_sum_value();

  }
  output_value = exp(sum_value)/sum;
  qDebug()<<" ----- sum_value: "<<sum_value<<";   sum: "<<sum<<";   exp: "<<exp(sum_value);
}

inline void Neuron::analizeLReLU() //----
{
//  qDebug() << "sum_value  " << sum_value<< "  negativeSlop "<<negativeSlop;

  if( sum_value < 0 )
      output_value = negativeSlop * sum_value;
  else
      output_value = sum_value;
}

inline void Neuron::calc_output_gradients(double target_value)
{
  double difference = target_value - output_value;
  gradient = difference * activation_function_derivative(output_value);
}

inline void Neuron::calc_output_gradientsSoftMax(double target_value, const Layer &this_layer) //---
{
  double sum = 0;
  for (size_t n = 0; n != this_layer.size(); n++) {
    sum += exp(this_layer[n].get_output_value());
    qDebug() <<" ------- n: "<<n<<";   get_output_value: "<<this_layer[n].get_output_value()<<";   exp: "<<exp(this_layer[n].get_output_value());
  }

  double difference = target_value - output_value;
  double activation_functionSoftMax = exp(output_value)/sum;
//  if(1-abs(difference) > 0.00001)
//      gradient = -log2(1-abs(difference)) * activation_functionSoftMax * (1-activation_functionSoftMax);
//  else
//      gradient = 10 * activation_functionSoftMax * (1-activation_functionSoftMax);

  gradient = difference * activation_functionSoftMax * (1-activation_functionSoftMax);

  qDebug() <<"           activation_functionSoftMax: "<<activation_functionSoftMax<<";   difference: "<<difference<<";   gradient: "<<gradient<<";   sum: "<<sum;

//  if(target_value < 1)
//      gradient += (activation_functionSoftMax - 1)/2;
//  else
//      gradient += activation_functionSoftMax/2;

//  if(target_value < 1)
//      gradient = difference * (1-activation_functionSoftMax );
//  else
//      gradient = difference * activation_functionSoftMax;

//  if(-log(1-abs(difference)) < 0.01)
//      gradient = 0;

//  if(-log(1-abs(difference)) > 10)
//{
//      if(target_value < 1)
//          gradient = activation_functionSoftMax - 1;
//      else
//          gradient = activation_functionSoftMax;

//  }

//  qDebug() << " loss_0 " << loss_value;
//  double loss_0 = loss_value;
  loss_value = -log(1-abs(difference));
//  if( loss_value > loss_0 ) gradient = -gradient;
//  qDebug() << " gradient " << gradient;
//  qDebug() << " loss_1 " << loss_value;

}

inline double Neuron::sum_weight_gradients(const Layer &next_layer) const
{
  double sum = 0;
//  qDebug()<< "next_layer.size() " << next_layer.size();
  for (size_t n = 0; n != next_layer.size(); n++) {
    sum += output_weights[n].weight * next_layer[n].gradient;
//    qDebug() << "next_layer[n].gradient  " << next_layer[n].gradient;
//    qDebug() << "output_weights[n].weight  " << output_weights[n].weight;


  }
  return sum;
}

inline void Neuron::calc_hidden_gradients(const Layer &next_layer)
{
  double  derivative_difference = sum_weight_gradients(next_layer);
  gradient = derivative_difference * activation_functionLReLU_derivative(output_value);
//  qDebug() << "gradient  " << gradient<< "  output_value "<<output_value;

}

inline void Neuron::recalc_gradients(Layer &prev_layer)
{
    qDebug() << "<Neuron::recalc_gradients>";
  for (size_t n = 0; n != prev_layer.size(); n++) {
    Neuron &neuron = prev_layer[n];
    neuron.add_loc_gradient( gradient * neuron.get_output_value(), neuron_number );
//    qDebug() << " ---------- n: "<<n<<";   gradient: "<<gradient<<";   neuron.get_output_value(): "<<neuron.get_output_value();
  }
}

inline void Neuron::recalc_gradientsBatch(Layer &prev_layer, int *batch_active_neurons, int n_active_neurons)
{
//    qDebug()<<"recalc ";
    for (int n = 0; n != n_active_neurons; n++) {
      Neuron &neuron = prev_layer[batch_active_neurons[n]];
      neuron.add_loc_gradient( gradient * neuron.get_output_value(), neuron_number );
//      if(neuron.get_output_value())
//        qDebug()<<"neuron.get_output_value() " << neuron.get_output_value();
    }
}


inline void Neuron::update_input_weights(Layer &prev_layer, double epoch)
{
//    qDebug() << " --- gradient: "<<gradient;
  for(size_t n = 0; n != prev_layer.size(); n++) {
    Neuron &neuron = prev_layer[n];
//    if( !neuron.output_value ) continue;
//    double old_change_weight = neuron.output_weights[neuron_number].change_weight;
//    double new_change_weight = (1 - neuron.alpha) * neuron.eta * neuron.get_output_value() * gradient + neuron.alpha * old_change_weight;
//    qDebug() << " ------- gradient: " << gradient;
    double new_change_weight;
    neuron.output_weights[neuron_number].vdw = beta1 * neuron.output_weights[neuron_number].vdw + (1 - beta1) * neuron.get_loc_gradient(neuron_number);//gradient;
    neuron.output_weights[neuron_number].sdw = beta2 * neuron.output_weights[neuron_number].sdw + (1 - beta2) * pow(neuron.get_loc_gradient(neuron_number),2);//pow(gradient,2);
//    neuron.output_weights[neuron_number].vdw = beta1 * neuron.output_weights[neuron_number].vdw + (1 - beta1) * gradient;
//    neuron.output_weights[neuron_number].sdw = beta2 * neuron.output_weights[neuron_number].sdw + (1 - beta2) * pow(gradient,2);

    double vdwCorrected = neuron.output_weights[neuron_number].vdw / (1-pow(beta1,epoch+1));
    double sdwCorrected = neuron.output_weights[neuron_number].sdw / (1-pow(beta2,epoch+1));
    new_change_weight = vdwCorrected / (sqrt(sdwCorrected) + epsilon );
    neuron.output_weights[neuron_number].change_weight = new_change_weight;
//    neuron.output_weights[neuron_number].weight += new_change_weight;
    neuron.output_weights[neuron_number].weight += new_change_weight * learningRate;
//    qDebug() << " --- new weight: " << neuron.output_weights[neuron_number].weight;
  }
  gradient = 0;
}

inline void Neuron::update_input_weightsBatch(Layer &prev_layer, double epoch, int *batch_active_neurons, int n_active_neurons)
{
//        qDebug() << " --- gradient: "<<gradient;
      for(size_t n = 0; n != n_active_neurons; n++) {
        Neuron &neuron = prev_layer[batch_active_neurons[n]];
    //    if( !neuron.output_value ) continue;
    //    double old_change_weight = neuron.output_weights[neuron_number].change_weight;
    //    double new_change_weight = (1 - neuron.alpha) * neuron.eta * neuron.get_output_value() * gradient + neuron.alpha * old_change_weight;
    //    qDebug() << " ------- gradient: " << gradient;
//        qDebug()<<"loc gradient "<<neuron.get_loc_gradient(neuron_number);
        double new_change_weight;
        neuron.output_weights[neuron_number].vdw = beta1 * neuron.output_weights[neuron_number].vdw + (1 - beta1) * neuron.get_loc_gradient(neuron_number);//gradient;
        neuron.output_weights[neuron_number].sdw = beta2 * neuron.output_weights[neuron_number].sdw + (1 - beta2) * pow(neuron.get_loc_gradient(neuron_number),2);//pow(gradient,2);
    //    neuron.output_weights[neuron_number].vdw = beta1 * neuron.output_weights[neuron_number].vdw + (1 - beta1) * gradient;
    //    neuron.output_weights[neuron_number].sdw = beta2 * neuron.output_weights[neuron_number].sdw + (1 - beta2) * pow(gradient,2);

        double vdwCorrected = neuron.output_weights[neuron_number].vdw / (1-pow(beta1,epoch+1));
        double sdwCorrected = neuron.output_weights[neuron_number].sdw / (1-pow(beta2,epoch+1));
        new_change_weight = vdwCorrected / (sqrt(sdwCorrected) + epsilon );
        neuron.output_weights[neuron_number].change_weight = new_change_weight;
    //    neuron.output_weights[neuron_number].weight += new_change_weight;
        neuron.output_weights[neuron_number].weight += new_change_weight * learningRate;
    //    qDebug() << " --- new weight: " << neuron.output_weights[neuron_number].weight;
      }
      gradient = 0;
}


inline void Neuron::get_neuron_weights(std::vector <double> &neuron_weights)
{
  for (size_t i = 0; i != output_weights.size(); i++) {
    neuron_weights.push_back(output_weights[i].weight);
  }
}

inline void Neuron::dropout(double probability)
{
//    qDebug() << "dropout START " << output_value;
    if( rand() / double(RAND_MAX) < probability)
        output_value = 0;
    else
        output_value = output_value / (1 - probability);

//    qDebug() << "dropout END " << output_value;
}
