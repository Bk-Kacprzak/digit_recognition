#ifndef DIGIT_RECOGNITION_LAYER_H
#define DIGIT_RECOGNITION_LAYER_H
#include <vector>

class Layer {
public:
    Layer(int neuronCount_, int weightNum);
    void reset();
    void randomize();

    //setters
    void retrieveActivation(Layer const*previousLayer);
    void activeInput(float value, int index);
    void setPreviousLayerError(float value, int neuron);
    void setWeightDerivative(float value, int neuron, int weight);
    void setBiasDerivative(float value, int neuron);
    void setErrorFunction(float value, int neuron);
    void applyTraining();

    //getters
    float getSigmoid(float argument);
    float getActivation(int index) const;
    int getNeuronCount() const;
    float getError(int neuron) const;
    float getWeightedSum(int neuron) const;
    float getWeight(int neuronIndex, int weightIndex) const;

private:
    int neuronCount;
    int weightCount;

    std::vector<float> activation;
    std::vector<float> weightedSum;
    std::vector<std::vector<float>> weight;
    std::vector<float> bias;
    std::vector<float> error;
    std::vector<std::vector<float>> weightDerivative;
    std::vector<float> biasDerivative;
};


#endif //DIGIT_RECOGNITION_LAYER_H
