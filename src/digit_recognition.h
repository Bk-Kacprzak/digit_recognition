#ifndef DIGIT_RECOGNITION_DIGIT_RECOGNITION_H
#define DIGIT_RECOGNITION_DIGIT_RECOGNITION_H
#include "file_reader.h"
#include "network.h"

class DigitRecognition {
public:
    DigitRecognition(int layersCount_, int imageSize_, int hiddenLayersNeuronCount_, int trainingBatches_);

private:
    void printDigit(const FileReader& file, int index) const;
    void train();
    void test();
    FileReader trainingImages;
    FileReader trainingLabels;
    FileReader testImages;
    FileReader testLabels;

    Network network;

    int trainingImagesPerBatch = 10;
    int trainingBatches;
    float learningRate = 1;
    float batchAccuracy;
};


#endif //DIGIT_RECOGNITION_DIGIT_RECOGNITION_H
