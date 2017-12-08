#include <iostream>

#include "core/core.h"

int main(int argc, char** argv) {
    myduomlia::bann::Bann bann("solve.bann");

    //    bann.train("train.bann");

    std::ifstream is("test.bann");
    if (!is.is_open()) {
        std::cerr << "Can't test file" << std::endl;
        exit(EXIT_FAILURE);
    }
    int count_samples, count_input_nodes, count_output_nodes;
    is >> count_samples >> count_input_nodes >> count_output_nodes;
    
    int success = 0;

    for (size_t j = 0; j < count_samples; j++) {
        Eigen::MatrixXf input(count_input_nodes, 1);
        for (size_t k = 0; k < count_input_nodes; k++) {
            float value;
            is >> value;
            input.row(k) << value;
        }
        int index_test;
        for (size_t k = 0; k < count_output_nodes; k++) {
            int value;
            is >> value;
            if(value == 1)
                index_test = k;
        }
        Eigen::MatrixXf res = bann.calc(input);
        int index_res = -1;
        float value = 0.1;
        for(size_t k = 0; k < count_output_nodes; k++){
            if(res(k, 0) > value){
                index_res = k;
                value = res(k, 0);
            }
        }
        if(index_test == index_res)
            success++;
    }
    is.close();
    std::cout << success * 1.0 / count_samples << std::endl;

    //    Eigen::MatrixXf input(4, 1);
    //    input << 1, 0, 1, 1;
    //    Eigen::MatrixXf output = bann.calc(input);
    //    std::cout << output << std::endl;
    return 0;
}

