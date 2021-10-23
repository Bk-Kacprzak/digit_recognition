#ifndef DIGIT_RECOGNITION_FILE_READER_H
#define DIGIT_RECOGNITION_FILE_READER_H

#include <vector>
#include <fstream>
#include <filesystem>

class FileReader {
public:
    explicit FileReader(const std::string& source);

    //getters
    int32_t getNumOfItems() const;
    float getPixel(int) const;
    int getLabel(int) const;

private:
    int getFileSize(const std::string&);
    void read();
    int32_t toLittleEndian(int) const;
    int fileSize;
    int32_t magicNum;
    int32_t dataIndex; //begin index of data
    int32_t numOfItems;
    std::vector<unsigned char> fileMap;
};

#endif //DIGIT_RECOGNITION_FILE_READER_H
