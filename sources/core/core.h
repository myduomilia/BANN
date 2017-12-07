#ifndef CORE_H
#define CORE_H

#include <iostream>
#include <fstream>
#include <vector>
#include <boost/lexical_cast.hpp>
#include <eigen3/Eigen/Dense>

#include "../json/json.hpp"

using json = nlohmann::json;

static float sigmoid(float x) {
    return 1.0 / (1.0 + exp(-x));
}

namespace myduomlia {
    namespace bann {

        class Bann {
        private:

            int m_inputnodes;
            std::vector<int> m_hiddennodes;
            int m_outputnodes;
            float m_learningrate;

            std::vector<Eigen::MatrixXf> m_weights;

            std::string read_configuration_file(const char *file);
            json parse_settings(const std::string &str);
            
            float _train(Eigen::MatrixXf & input, Eigen::MatrixXf & outpit);

        public:
            Bann(const std::string & solve);
            void train(const std::string & data_set);
            Eigen::MatrixXf calc(Eigen::MatrixXf & input);
        };
    }
}

#endif

