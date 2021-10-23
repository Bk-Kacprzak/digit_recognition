#include "digit_recognition.h"
#include <iostream>
#include <chrono>
#include <thread>

DigitRecognition::DigitRecognition(int layersCount_, int imageSize_, int hiddenLayersNeuronCount_, int trainingBatches_) :
        trainingImages(std::filesystem::current_path().parent_path().string()+ "/MNIST/train-images-idx3-ubyte"),
        trainingLabels(std::filesystem::current_path().parent_path().string()+ "/MNIST/train-labels-idx1-ubyte"),
        testImages(std::filesystem::current_path().parent_path().string()+ "/MNIST/t10k-images-idx3-ubyte"),
        testLabels(std::filesystem::current_path().parent_path().string()+ "/MNIST/t10k-labels-idx1-ubyte"),
        trainingBatches(trainingBatches_),
        network(layersCount_, imageSize_, hiddenLayersNeuronCount_, 10) {
    std::cout<<"Press any key to train the network";
    std::cin.get();
    train();
    std::cout<<"Press any key to test\n";
    std::cin.get();
    test();
}

void DigitRecognition::printDigit(const FileReader& file, int index) const{
    for (int i = 0; i < 28; i++)
    {
        for (int j = 0; j < 28; j++)
        {
            if ((int)(10 * (file.getPixel(index * 784 + i * 28 + j))) > 3) //edge
            {
                std::cout<<"* ";
            }
            else
            {
                std::cout<<"  ";
            }
        }
        std::cout<<std::endl;
    }
}

void DigitRecognition::train() {
    float batchCost = 0.0f;
    int imageIndex = 0;
    for (int b = 0; b < trainingBatches; b++)
    {
        batchCost = 0.0f;
        batchAccuracy = 0.0f;
        for (int m = 0; m < trainingImagesPerBatch; m++)
        {
            imageIndex = b * trainingImagesPerBatch + m;
            for (int i = 0; i < 784; i++)
            {
                network.loadInput(trainingImages.getPixel(imageIndex * 784 + i), i);
            }

            network.activate();
            network.estimateCost(trainingLabels.getLabel(imageIndex));
            network.learn(learningRate, trainingImagesPerBatch);
            batchCost += network.getCost() / trainingImagesPerBatch;
            batchAccuracy += ((float)network.isCorrect() / (float)trainingImagesPerBatch);
        }
        network.applyLearned();
        std::cout<<"Batch : "<<b + 1<<" / "<<trainingBatches<<" Accuracy : "<<batchAccuracy * 100<<"% Cost : "<<batchCost<<std::endl;
    }

    std::cout<<std::endl<<"Done Training!"<<std::endl<<std::endl;

}

void DigitRecognition::test() {
    int correct = 0;
    float accuracy = 0.0f;

    for (int i = 0; i < testImages.getNumOfItems(); i++)
    {
        for (int j = 0; j < 784; j++)
        {
            network.loadInput(testImages.getPixel(i * 784 + j), j);
        }
        network.activate();
        network.estimateCost(testLabels.getLabel(i));
        correct += network.isCorrect();
        accuracy = 100 * (float)(correct) / (i + 1);
        printDigit(testImages, i);
        std::cout<<std::endl<<"Answer : " <<network.getAnswer()<<std::endl;
        std::cout<<std::endl<<"Average Accuracy : "<<accuracy<< "% "<<std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
    }
}
