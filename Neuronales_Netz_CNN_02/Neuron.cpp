#include <iostream>
#include <vector>
#include <math.h>
#include "Neuron.h"
#include <cstdlib>

Neuron::Neuron() ///unsigned out_num
{
    //Aufgabe a):
    //testHidden();
    //val.push_back(1);
    //val.push_back(2);
    //val.push_back(3);
    //std::cout << "\n";

    //softMax(val); //Aktivierungsfunktion
    sumHid();
    derivation();
    //outputGradient();
}

float Neuron::sum_ex(){
    //Aufgabe b):
    ///Hier wird die Summe der Werte zusammen gerechnet und zurückgegeben
    float sum = 0;
    //float sum_hid = 0;
    return sum; // die Summe wird zurückgegeben
}

float Neuron::learningRate(){
    //Die Funktion berechnet die learning rates, welches die Kontroll der Geschwindigkeit
    //auf Konvergenzen zeigt.
    //Differnenzialgleichung:
    //w_t+1 = w_t + eta / wurzel(v + epsilon)
    double beta1 = 0.9;
    double m;
    m = beta1 + (1 - beta1) * out_m;
    eta = 0.0;          //Counter für die learning rates
    double e = pow(10, -8);
    learning = 0;
    for(int i = 0; i < deltaWeight.size(); i++){
        eta += 1;
        learning = - eta * (e / deltaWeight[i]) + m;
        weight_t.push_back(learning);
    }
    std::cout << "\n";

    for(int i = 0; i < weight_t.size(); i++){
        std::cout << "Learning rates: " << weight_t[i] << "\n";
    }

    return eta;
}

float Neuron::weightUpdate(){
    //Diese Funktion updatet die Gewichte

    //float t = 0;


    weight = rand() / double(RAND_MAX);
    deltaWeight.push_back(weight); //Die weights werden in diesem Vector zurückgegeben
    return weight;
}


float Neuron::first_step(std::vector<int> input){
    //Aufgabe b):
    ///Hier wird Softmax berechnet die s(x) = e^x / sum(e^x)
    ///Aktivierungsfnktion für den Output

    float sum = 0;     //Für die erst Summe der Hidden-Layers


    //deltaWeight.push_back(weight); //Die Gewichte werden in dieser Variable gespeichert und zurückgegeben.

    for(unsigned int i = 0; i < input.size(); i++) // für jeden wert des eingegebenen vektors wird der e hoch i berechnet
    {
        //std::cout << "Der Vektor hat folgende Elemente: " << input[i] << std::endl;
        //Hier werden die Inputs mit dem Gewicht (Kanten) multipliziert

        std::cout << " Inputs x: " << i << std::endl;
        std::cout << "Zufaellige weights w" << i << ": " << weightUpdate() << std::endl;

        vec.push_back(input[i] * weightUpdate()); //Speichert die Inputs * Weights

        sum += input[i] * weightUpdate();            //Die Summe der Gewichte und Inputs

        ///Zusätzlich werden zufällige Zahlen multipliziert, die die Weights sind
        ///w[l] = rand(0:1)
        /// p1 * w1 -->
        /// p2 * w2 -->
        /// p3 * w3 -->
        /// ...
        /// pn * wn -->
        /// Siehe Plenartfolie_03 Seite 5
    }
    sigma = sum;
    std::cout << "\n";

    // Der Wert an der gegebenen Position (i auf Blatt) wird durch sum geteilt
    hidden.push_back(sigma);                            //Hier werden die Werte der Hidden-Layer gespeichert.
}

float Neuron::softMax(std::vector<float> input){
    //Aufgabe b):
    ///Hier wird Softmax berechnet die s(x) = e^x / sum(e^x)
    ///Aktivierungsfnktion für den Output

    float sum = 0;     //Für die erst Summe der Hidden-Layers
    float sum_out = 0; //Für den Output der summierten Hidden-Layer
    float sigmoid = 0; //Für den endgültigen Output


    //deltaWeight.push_back(weight); //Die Gewichte werden in dieser Variable gespeichert und zurückgegeben.

    for(unsigned int i = 0; i < input.size(); i++) // für jeden wert des eingegebenen vektors wird der e hoch i berechnet
    {
        //std::cout << "Der Vektor hat folgende Elemente: " << input[i] << std::endl;
        //Hier werden die Inputs mit dem Gewicht (Kanten) multipliziert

        std::cout << " Inputs x: " << i << std::endl;
        std::cout << "Zufaellige weights w" << i << ": " << weightUpdate() << std::endl;

        vec.push_back(input[i] * weightUpdate()); //Speichert die Inputs * Weights

        sum += input[i] * weightUpdate();            //Die Summe der Gewichte und Inputs

        ///Zusätzlich werden zufällige Zahlen multipliziert, die die Weights sind
        ///w[l] = rand(0:1)
        /// p1 * w1 -->
        /// p2 * w2 -->
        /// p3 * w3 -->
        /// ...
        /// pn * wn -->
        /// Siehe Plenartfolie_03 Seite 5
    }
    std::cout << "\n";

    // Der Wert an der gegebenen Position (i auf Blatt) wird durch sum geteilt
    float sigma = (1 / (1 + std::exp(- sum)));           //Aktivierungsfunktion

    sum_out += sigma * weightUpdate();
    sigmoid = (1 / (1 + std::exp(- sum_out)));           //Aktivierungsfunktion Softmax
    std::cout << "Output der Softmax(" << sum_out << "): " << sigmoid << std::endl;

    hidden.push_back(sigma);                            //Hier werden die Werte der Hidden-Layer gespeichert.
    outVal.push_back(sigmoid);                          //Die Softmax-Endergebnisse werden in diesem Vector zurückgegeben.

}

///Diese Funktion sorgt dafür, dass eine Verbindung zwischen Neuronen von Layer i und Layer i + 1,
///mitsamt einer Möglichkeit die Gewichte zu hinterlegen.

float Neuron::derivation(){
    //Aufgabe b):
    ///Die Ableitungsfunktion sigmoid'(x) = 1 / 1 + e^-x dinet für das Lernen des Netzwerkes

    for(int i = 0; i < outVal.size(); i++){
        //exp_deriv = 1 / 1 + std::exp(-outVal[i]); ///Die Softmax für die Backpropagation
        exp_deriv = (outVal[i] * (1 - outVal[i])); //Die wahre Ableitung der Softmax Funktion: f'(x) = S * (1 - S)
    }
    std::cout << "Die Ableitung der Softmax(): " << exp_deriv << std::endl;
    return exp_deriv;
}

float Neuron::sumHid()
{
    //Aufgabe c):
    ///Die Hidden Funktion berechnet die jeweiligen Inputs mit ihren Gewichtungen
    /// berechnet werden:
    ///
    /// input1 * weights + input2 * weights = hidden
    /// 1 * 0.8 + 1 * 0.2 = 1
    ///1 * 0.4 + 1 * 0.9 = 1.3
    ///1 * 0.3 + 1 * 0.5 = 0.8
    double sum_hid = 0;  //Die Summe der Hidden-Layer wird berechnet
    for(int i = 0; i < vec.size(); i++){

        std::cout << "Vektor (Inputs * Weights) Element(" << i << "): " << vec[i] << "\n";
        sum_hid += vec[i]; //Die Layer werden zusammen mit den weights multipliziert und addiert.
    }
    std::cout << "Die Summe der Hidden-Layers L: " << sum_hid << std::endl; //Ausgabe der Summe Layer
    //std::cout << "--------------------------------------------------------------------------------------------------------------"<<"\n";


}


float Neuron::outputGradient(std::vector<unsigned int> input_val){
    ///hidden_1 = 10 * (layer_1) + 0 * (layer_2) + 2 * (layer_3)
    /// Hier soll die Output-Gradient für output-Layer für andere Hidden-Gradient für alle Hidden-Layer
    //float sig_outn; //Für den Output Layer Gradient Berechnung:

    double t = 0;

    out_m = t - sigma * exp_deriv;
    //om = (tm - f(xm)) * f'(xm):
    for(int i = 0; i < outVal.size(); i++){
        std::cout << "Output Sigma(): " << outVal[i] << "\n";
    }
    std::cout << "Output Gradient: " << out_m << "\n";
    return out_m;

}


/*
float Neuron::testHidden(){
    float s (0);
    float s2 (0);
    float f;
    float f2;
    std::vector<float> myvec;
    myvec.push_back(1);
    myvec.push_back(2);
    myvec.push_back(3);

    for(int i = 0; i < myvec.size(); i++){
        //std::cout << "Zufaellige weights w" << i << ": " << weight << std::endl;
        //std::cout << "Die Input : x" << i << ": " << myvec.at(i) << std::endl;

        s += weight * myvec.at(i); //Jedes Element wird mit seiner zufälligen Gewichtung multipliziert.
    }

    std::cout << "S(x): " << s << std::endl;
    f = (1/(1+exp(s)));
    std::cout << "Die Hidden-Layer sind: f(" << s << ") = " << f << std::endl;
    std::cout << "\n";
    s2 += s * weight;
    std::cout << "Hidden-Layer(x): " << s2 << std::endl;
    f2 = (1 / (1 + std::exp(s2)));
    std::cout << "Die Softmax-Funktion: f2(" << s << ") = " << f2 << std::endl;
    myvec.pop_back();
}
*/
