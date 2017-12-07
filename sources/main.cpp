#include <iostream>

#include "core/core.h"

int main(int argc, char** argv) {
    myduomlia::bann::Bann bann;
    
    bann.train("train.bann");
    
//    Eigen::MatrixXf input(4, 1);
//    input << 1, 0, 1, 1;
//    Eigen::MatrixXf output = bann.calc(input);
//    std::cout << output << std::endl;
    return 0;
}

