#include <iostream>
#include "src/digit_recognition.h"

#define IMAGE_SIZE 28*28

int main() {
    //by defualt:
    //4 layers
    //30 neurons for each hidden layer
    //1000 training batches
    //10 training imsges per batch
    //learning rate = 1


    DigitRecognition application(4,IMAGE_SIZE,30,1000);
    return 0;
}

//neural network theory resources
//http://neuralnetworksanddeeplearning.com