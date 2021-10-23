cmake_minimum_required(VERSION 3.17)
project(digit_recognition)

set(CMAKE_CXX_STANDARD 17)

add_executable(digit_recognition main.cpp src/file_reader.cpp src/file_reader.h src/layer.cpp src/layer.h src/network.cpp src/network.h src/digit_recognition.cpp src/digit_recognition.h)