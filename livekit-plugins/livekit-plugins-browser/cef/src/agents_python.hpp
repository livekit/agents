#ifndef LKCEF_AGENT_PYTHON_HPP
#define LKCEF_AGENT_PYTHON_HPP

#include <pybind11/pybind11.h>


class BrowserImpl {
public:
    BrowserImpl();
    void start();
};

#endif // LKCEF_AGENT_PYTHON_HPP
