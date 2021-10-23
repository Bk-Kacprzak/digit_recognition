#ifndef DIGIT_RECOGNITION_NETWORK_H
#define DIGIT_RECOGNITION_NETWORK_H
#include <cmath>
#include <vector>
#include "layer.h"
class Network {
public:
    Network(int layersCount_, int inputLayersCount_, int hiddenLayersCount_, int outputLayersCount_);
    ~Network();

    //setters
    void loadInput(float inputValue, int index);
    void activate();
    void estimateCost(int realAnswer);
    void learn(float learningRate, int batchSize);

    //getters
    float sigmoidDerivative(float inputValue);
    float getSigmoid(float inputValue);
    void applyLearned();
    float getCost() const;
    int isCorrect() const;
    int getAnswer() const;

private:
    std::vector<Layer*> layers;
    std::vector<float> correctAnswers;

    int inputLayerNeuronCount;
    int hiddenLayerNeuronCount;
    int outputLayerNeuronCount;
    int layersCount;
    float answerCost;
    int correct;
    int answer;
};


#endif //DIGIT_RECOGNITION_NETWORK_H
