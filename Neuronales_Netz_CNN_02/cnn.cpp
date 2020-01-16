#include "cnn.h"
#include <iostream>
#include <vector>
#include <fstream>
#include "NeuralNet.h"
#include <string>
#include <stdlib.h>



float Channel::lReLu(){

    // Lineare Aktivierungsfunktion
    // f(x) = max(0.01*x, x) Leaky ReLu:
    //  ^
    //  |              /
    //  |             /
    //  |            /
    //  |           /
    //  |          /
    //  |         /
    //  |        /
    //  |       /
    //--|------/------------------------->
    //  |_____/
      //maxLReLu = std::max_element(0, x);
      //std::cout << std::max(0.0, x);

}


float Kernell::reset()
{
    n = 3; // Größe der Matrix.
    // Weight 3x3x3 matrix
    //   ________________
    // /                /|
   // /________________/ |
   // |w0,0| w0,1| w0,2| |
   // |w1,0| w1,1| w1,2| |
   // |w2,0| w2,1| w2,2|/
   // ------------------
    kernell[n][n][n] = 0; // 3x3x3-Matrix

    // Fülle die Matrix mit Randomzahlen auf:
    for(int i = 0; i < sizeof (kernell); i++)
    {
        weight = -1 + rand() % 3;
        for(int j = 0; j < n; j++)
        {
            for(int k = 0; k < n; k++)
            {
                kernell[i][j][j] =  weight;
                // Printe die 3x3x3-Matrix:
                //std::cout << kernell[i][j][k] << " ";
            }
        }
        //std::cout << "\n";
    }
    return kernell[n][n][n];
}
