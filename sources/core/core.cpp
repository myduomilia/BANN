#include "core.h"

myduomlia::bann::Bann::Bann(const std::string & solve) {
    json settings = parse_settings(read_configuration_file("settings.json"));
    if (settings.find("inputnodes") == settings.end()) {
        std::cerr << "Not found inputnodes" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (settings.find("hiddennodes") == settings.end()) {
        std::cerr << "Not found hiddennodes" << std::endl;
        exit(EXIT_FAILURE);
    }
    if (settings.find("outputnodes") == settings.end()) {
        std::cerr << "Not found outputnodes" << std::endl;
        exit(EXIT_FAILURE);
    }
    m_learningrate = 0.1;
    if (settings.find("learningrate") != settings.end())
        m_learningrate = boost::lexical_cast<float>(settings["learningrate"]);
    m_inputnodes = boost::lexical_cast<int>(settings["inputnodes"]);
    m_outputnodes = boost::lexical_cast<int>(settings["outputnodes"]);
    try {
        for (auto hiddennodes : settings["hiddennodes"])
            m_hiddennodes.push_back(boost::lexical_cast<int>(hiddennodes));
    } catch (...) {
        std::cerr << "Can't parse configuration file" << std::endl;
    }
    srand((unsigned int) time(0));
    Eigen::MatrixXf mat = Eigen::MatrixXf::Random(m_hiddennodes[0], m_inputnodes).cwiseAbs() * (1 / std::sqrt(m_inputnodes));
    m_weights.push_back(mat);
    for (size_t i = 1; i < m_hiddennodes.size(); i++) {
        mat = Eigen::MatrixXf::Random(m_hiddennodes[i], m_hiddennodes[i - 1]).cwiseAbs() * (1 / std::sqrt(m_hiddennodes[i]));
        m_weights.push_back(mat);
    }
    mat = Eigen::MatrixXf::Random(m_outputnodes, m_hiddennodes[m_hiddennodes.size() - 1]).cwiseAbs() * (1 / std::sqrt(m_hiddennodes[m_hiddennodes.size() - 1]));
    m_weights.push_back(mat);

}

Eigen::MatrixXf myduomlia::bann::Bann::calc(Eigen::MatrixXf & input) {
    Eigen::MatrixXf x = m_weights[0] * input;
    Eigen::MatrixXf final_outputs = x.unaryExpr(&sigmoid);
    for (size_t i = 1; i < m_hiddennodes.size(); i++) {
        x = m_weights[i] * final_outputs;
        final_outputs = x.unaryExpr(&sigmoid);
    }
    x = m_weights[m_weights.size() - 1] * final_outputs;
    final_outputs = x.unaryExpr(&sigmoid);
    return final_outputs;
}

float myduomlia::bann::Bann::_train(Eigen::MatrixXf & input, Eigen::MatrixXf & output) {
    Eigen::MatrixXf x = m_weights[0] * input;
    std::vector<Eigen::MatrixXf> vec_final_outputs;
    Eigen::MatrixXf final_outputs = x.unaryExpr(&sigmoid);
    vec_final_outputs.push_back(final_outputs);
    for (size_t i = 1; i < m_hiddennodes.size(); i++) {
        x = m_weights[i] * final_outputs;
        final_outputs = x.unaryExpr(&sigmoid);
        vec_final_outputs.push_back(final_outputs);
    }
    x = m_weights[m_weights.size() - 1] * final_outputs;
    final_outputs = x.unaryExpr(&sigmoid);
    vec_final_outputs.push_back(final_outputs);

    Eigen::MatrixXf output_errors = output - final_outputs;
    float error = output_errors.squaredNorm();
    m_weights[m_weights.size() - 1] += m_learningrate * (output_errors.array() * final_outputs.array() * (1 - final_outputs.array())).matrix() * vec_final_outputs[vec_final_outputs.size() - 2].transpose();

    for (size_t i = m_weights.size() - 2; i > 0; i--) {
        Eigen::MatrixXf hidden_errors = m_weights[i + 1].transpose() * output_errors;
        m_weights[i] += m_learningrate * (hidden_errors.array() * vec_final_outputs[i].array() * (1 - vec_final_outputs[i].array())).matrix() * vec_final_outputs[i - 1].transpose();
        output_errors = hidden_errors;
    }
    Eigen::MatrixXf input_errors = m_weights[1].transpose() * output_errors;
    m_weights[0] += m_learningrate * (input_errors.array() * vec_final_outputs[0].array() * (1 - vec_final_outputs[0].array())).matrix() * input.transpose();

    return error;
}

void myduomlia::bann::Bann::train(const std::string & data_set) {
    const int EPOCH = 100;
    for (size_t i = 0; i < EPOCH; i++) {
        std::ifstream is("train.bann");
        if (!is.is_open()) {
            std::cerr << "Can't train file" << std::endl;
            exit(EXIT_FAILURE);
        }
        int count_samples, count_input_nodes, count_output_nodes;
        is >> count_samples >> count_input_nodes >> count_output_nodes;
        float wrong = 0.0;
        for (size_t j = 0; j < count_samples; j++) {
            Eigen::MatrixXf input(count_input_nodes, 1);
            Eigen::MatrixXf output(count_output_nodes, 1);
            for (size_t k = 0; k < count_input_nodes; k++) {
                float value;
                is >> value;
                input.row(k) << value;
            }
            for (size_t k = 0; k < count_output_nodes; k++) {
                float value;
                is >> value;
                output.row(k) << value;
            }
            wrong += _train(input, output) / m_outputnodes;
        }
        std::cout << "EPOCH = " << i << " wrong = " << wrong / count_samples << std::endl;
        is.close();
    }
    std::ofstream os("solve.bann", std::ofstream::out);
    if (!os.is_open()) {
        std::cerr << "Can't output file" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (auto mat : m_weights) 
        os << mat << std::endl;
    os.close();

}

std::string myduomlia::bann::Bann::read_configuration_file(const char * file) {
    FILE* f = fopen(file, "r");
    if (f == NULL) {
        std::cerr << "Can't configuration file" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::string settings;
    char buffer[1024 * 16];
    memset(buffer, '\0', sizeof (buffer));
    while (fgets(buffer, sizeof (buffer), f) != NULL) {
        settings += std::string(buffer);
        memset(buffer, '\0', sizeof (buffer));
    }
    fclose(f);
    return settings;
}

json myduomlia::bann::Bann::parse_settings(const std::string &str) {
    json settings;
    try {
        settings = json::parse(str);
    } catch (...) {
        std::cerr << "Can't parse json configuration file" << std::endl;
        exit(EXIT_FAILURE);
    }
    return settings;
}