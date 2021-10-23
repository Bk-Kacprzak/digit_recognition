#include "layer.h"
#include <cmath>
#include <iostream>

Layer::Layer(int neuronCount_, int weightCount_) :
        neuronCount(neuronCount_), weightCount(weightCount_) {

    weight.resize(neuronCount); //30 neurons

    for(int i=0; i < neuronCount; i++)
        weight[i].resize(weightCount);

    bias.resize(neuronCount);
    activation.resize(neuronCount);
    weightedSum.resize(neuronCount);

    error.resize(neuronCount);
    weightDerivative.resize(neuronCount);

    for(int i = 0; i < neuronCount; i++) {
        weightDerivative[i].resize(weightCount);
    }

    biasDerivative.resize(neuronCount);

    reset();
    randomize();
}

void Layer::randomize() {
    for(int i = 0; i < neuronCount; i++) {
        bias[i] = 2 * ((static_cast <float> (rand())) / (static_cast <float> (RAND_MAX))) - 1;

        for(int j = 0; j< weightCount; j++) {
            weight[i][j] = 2 * ((static_cast <float> (rand())) / (static_cast <float> (RAND_MAX))) - 1;
        }
    }
}

void Layer::reset() {
    for(int i = 0; i < neuronCount; i++) {
        error[i] = 0.0f;
        biasDerivative[i] = 0.0f;
        for (int j = 0; j < weightCount; j++)
        {
            weightDerivative[i][j] = 0.0f;
        }
    }
}

//setters

void Layer::retrieveActivation(Layer const *previousLayer) {
    for(int i =0; i < neuronCount; i++) {
        //sum (previous_neuron * weights) + bias
        weightedSum[i] = 0.0f;
        for(int j=0; j< weightCount; j++) {
            weightedSum[i] += previousLayer->getActivation(j) * weight[i][j];
        }
        weightedSum[i] += bias[i];
        //get Sigmoid of it and set the activation of each neuron
        activation[i] = getSigmoid(weightedSum[i]);
    }
}

float Layer::getSigmoid(float argument) {
    return 1.0f/(1 + exp(-argument));
}

void Layer::activeInput(float value, int index) {
    activation[index] = value;
}

void Layer::applyTraining() {
    //stochastic gradient descent
    //weight[neruon][weight] - deriative (weight)
    //bias[neuron] - deriative (bias)

    for (int j = 0; j < neuronCount; j++)
    {
        for (int k = 0; k < weightCount; k++)
        {
            weight[j][k] += weightDerivative[j][k];
            weightDerivative[j][k] = 0.0f;
        }
        bias[j] += biasDerivative[j];
        biasDerivative[j] = 0.0f;
    }
}

void Layer::setErrorFunction(float value, int neuron) {
    error[neuron] = value;
}

void Layer::setWeightDerivative(float value, int neuron, int weight) {
    weightDerivative[neuron][weight] += value;
}

void Layer::setBiasDerivative(float value, int neuron) {
    biasDerivative[neuron] += value;
}

void Layer::setPreviousLayerError(float value, int neuron) {
    error[neuron] += value;
}

//getters

float Layer::getActivation(int index) const {
    return activation[index];
}

int Layer::getNeuronCount() const {
    return neuronCount;
}

float Layer::getWeightedSum(int neuron) const {
    return weightedSum[neuron];
}

float Layer::getWeight(int neuronIndex, int weightIndex) const {
    return weight[neuronIndex][weightIndex];
}

float Layer::getError(int neuron) const {
    return error[neuron];
}
