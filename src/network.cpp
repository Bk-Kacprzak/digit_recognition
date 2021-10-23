#include "network.h"
#include <iostream>
Network::Network(int layersCount_, int inputLayersCount_, int hiddenLayersCount_, int outputLayersCount_) :
        layersCount(layersCount_),
        inputLayerNeuronCount(inputLayersCount_),
        hiddenLayerNeuronCount(hiddenLayersCount_),
        outputLayerNeuronCount(outputLayersCount_) {

    if(layersCount < 3 || inputLayerNeuronCount < 1 || hiddenLayerNeuronCount < 1 || outputLayerNeuronCount < 1)
        throw std::invalid_argument("Invalid parameters");

    correctAnswers.resize(outputLayerNeuronCount); //10
    layers.resize(layersCount);

    //input layer
    layers[0] = new Layer(inputLayerNeuronCount, 0);
    //first hidden layer, outside the loop to connect weightCount with input layer
    layers[1] = new Layer(hiddenLayerNeuronCount, inputLayerNeuronCount);

    //hidden layers
    for(int i = 2; i < layers.size() - 1; i++)
        layers[i] = new Layer(hiddenLayerNeuronCount, hiddenLayerNeuronCount);

    //output layer
    layers[layersCount - 1] = new Layer(outputLayerNeuronCount, hiddenLayerNeuronCount);
}

Network::~Network() {
    for(int i=0; i<layersCount; i++)
        delete layers[i];
}

void Network::loadInput(float inputValue, int index) {
    layers[0]->activeInput(inputValue, index);
}

void Network::activate() {
    for(int i=1; i<layersCount; i++) {
        Layer const * previousLayer = layers[i - 1];
        layers[i]->retrieveActivation(previousLayer);
    }
}

void Network::estimateCost(int realAnswer) {
    answerCost = 0.0f;
    int networkAnswerIndex = 0;

    for (int i = 0; i < outputLayerNeuronCount; i++)
    {
        if (layers[layersCount -1]->getActivation(networkAnswerIndex) < layers[layersCount - 1]->getActivation(i))
        {
            networkAnswerIndex = i;
        }

        if (i == realAnswer)
        {
            correctAnswers[i] = 1.0f;
        }

        else
        {
            correctAnswers[i] = 0.0f;
        }

        //cost function
        answerCost += pow((correctAnswers[i] - layers[layersCount-1]->getActivation(i)), 2) * 0.5f;

        correct = 0;
        if (correctAnswers[networkAnswerIndex] == 1.0f)
        {
            correct = 1;
        }

        answer = networkAnswerIndex;
    }
}

void Network::learn(float learningRate, int batchSize) {

    Layer *&outputLayer = layers[layersCount -1];

    //output layer
    for (int i = 0; i < outputLayerNeuronCount; i++)
    {
        //gradient for hidden to output weights
        //error function[i] = d(cost)/d(activation) * sigmoid'(weighted sum)
        //d(cost)/d(activation) - how fast the cost is changing
        //sigmoid'(weighted sum) - how fast the activation is changing at  weighted sum
        outputLayer->setErrorFunction(
                (outputLayer->getActivation(i) - correctAnswers[i]) * sigmoidDerivative(outputLayer->getWeightedSum(i)), i);


        //gradient of the cost function

        for (int j = 0; j < hiddenLayerNeuronCount; j++)
        {
            //rate of change of the cost with respect to any weight in the network
            outputLayer->setWeightDerivative(-(learningRate / (float) batchSize) * outputLayer->getError(i) *
                                             layers[layersCount - 2]->getActivation(j), i, j);
        }

        outputLayer->setBiasDerivative(-(learningRate / (float) batchSize) * outputLayer->getError(i), i);
    }

    // calculate hidden layers
    // error functions for the next layers
    for (int i = (layersCount - 2); i > 0; i--)
    {
        for (int j = 0; j < layers[i]->getNeuronCount(); j++)
        {
            layers[i]->setErrorFunction(0.0f, j);

            for (int k = 0; k < layers[i + 1]->getNeuronCount(); k++)
            {
                layers[i]->setPreviousLayerError(
                        layers[i + 1]->getWeight(k, j) * layers[i + 1]->getError(k) * sigmoidDerivative(
                                layers[i]->getWeightedSum(j)), j);
            }

            //gradient of the cost function

            for (int k = 0; k < layers[i - 1]->getNeuronCount(); k++)
            {
                layers[i]->setWeightDerivative(
                        -(learningRate / (float) batchSize) * layers[i]->getError(j) * layers[i - 1]->getActivation(k),
                        j, k);
            }

            //dc/db = error function
            layers[i]->setBiasDerivative(-(learningRate / (float) batchSize) * layers[i - 1]->getError(j), j);
        }
    }
}

void Network::applyLearned() {
    for (int i = 1; i < layersCount; i++)
    {
        layers[i]->applyTraining();
    }
}

float Network::sigmoidDerivative(float inputValue) {
    return getSigmoid(inputValue) * (1 - getSigmoid(inputValue));
}

float Network::getSigmoid(float argument) {
    return 1.0f/(1 + exp(-argument));
}

float Network::getCost() const {
    return answerCost;
}

int Network::isCorrect() const {
    return correct;
}

int Network::getAnswer() const {
    return answer;
}