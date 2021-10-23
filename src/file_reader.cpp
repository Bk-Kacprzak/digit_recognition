//
// Created by Bartłomiej Kacprzak on 21/10/2021.
//

#include "file_reader.h"
#include <iostream>

FileReader::FileReader(const std::string& source) {
    char byte;
    std::ifstream file;
    file.open(source, std::ios::binary | std::ios::in);
    if(!file.good())
        throw std::runtime_error("Unable to open the file\n");

    fileSize = (int)getFileSize(source);
    fileMap.resize(fileSize);

    for(int i =0; i < fileSize; i++) {
        file.read(&byte, 1);
        fileMap[i] = byte;
    }
    file.close();

    read();
}

int FileReader::getFileSize(const std::string &source) {
    std::ifstream fileSize(source, std::ifstream::ate | std::ifstream::binary);
    return (int)fileSize.tellg();
}

void FileReader::read() {
    //information about magic number, dataIndex etc.
    //http://yann.lecun.com/exdb/mnist/

        magicNum = toLittleEndian(0);
        //magic number for images: 2051
        //magic number for labels: 2049
        if((magicNum != 2051) && (magicNum != 2049))
            throw std::runtime_error("File read unproperly\n");

        numOfItems = toLittleEndian(4);
        //image
        if(magicNum == 2051)
            dataIndex = 16;
        else
            dataIndex = 8;
}

int32_t FileReader::toLittleEndian(int index) const {
    //bytes are written in big endian, translate it
    if(LITTLE_ENDIAN)
        return fileMap[index + 3] | (fileMap[index + 2] << 8) | (fileMap[index + 1] << 16) | (fileMap[index + 0] << 24);
    else
        return index;
}

int32_t FileReader::getNumOfItems() const {
    return numOfItems;
}

float FileReader::getPixel(int index) const {
    if(index + dataIndex >= fileMap.size())
        return 0.0f;

    int pixel = fileMap[index + dataIndex];
    return ((float)pixel/255);
}


int FileReader::getLabel(int index) const {
    return fileMap[index + dataIndex];
}
