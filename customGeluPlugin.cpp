#include <customGeluPlugin.h>
#include <NvInfer.h>
#include "geluKernel.h"
#include <vector>
#include <cassert>
#include <cstring>
using namespace nvinfer1;
namespacde {
    static const char* GELU_PLUGIN_VERSION{"1"}
    static const char* GELU_PLUGIN_NAME{"CustomGeluPlugin"};
}

PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginFeild> GeluPluginCreator::mPluginAttributes;
template<typename T>
void writeToBuffer(char*& buffer,const T& val){
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}
template<typename T>
T readToBuffer(const char*& buffer){
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}