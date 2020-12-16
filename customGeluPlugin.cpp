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
GeluPlugin::GeluPlugin(const std::string name,const void* data,size_t length):mLayoerName(name){
    const char *d = static_cast<const char *>(data);
    const char *a = d;
    assert(d == (a+length));
}
const char* GeluPlugin::getPluginType() const{
    return GELU_PLUGIN_NAME;
}
const char* GeluPlugin::getPluginVersion() const{
    return GELU_PLUGIN_VERSION
}
int GeluPlugin::getNbOutputs() const{
    return 1;
}
Dims GeluPlugin::getOutputDimensions(int index,const Dims* inputs,int nbInputDims){
    assert(nbInputDims == 1);
    assert(index == 0);
    return *inputs;
}
int GeluPlugin::initialize(){
    return 0;
}
int GeluPlugin::enqueue(int batchSize,const void* const* inputs,void** outputs,void* cudaStream_t stream){
    int status = -1;
    void* output = outputs[0];
    status = geluInference(stream,mInputVolume*batchSize,inputs[0],output);
    return status;
}
size_t GeluPlugin::serialize(void* buffer)const{
    char* d = static_cast<char*>(buffer);
    const char *a = d;
    assert(d == a+getSe)
}
void GeluPlugin::configureWithFormat(const Dims* inputs,int nbInputs,const Dims* outputs,int nbOutputs,DataType type,PluginFormat format,int){
    assert(nbOutputs == 1);
    assert(type == DataType::kFLOAT);
    assert(format == PluginFormat::kNCHW);
    size_t volumn = 1;
    for(int i=0;i<inputs->nbDims;i++){
        volumn *= inputs->d[i];
    }
    mInputVolume = volumn;
}

bool GeluPlugin::supportsFormat(DataType type,PluginFormat format) const{
    if(type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
    {
        return false;
    }
    
}
void GeluPlugin::terminate(){}
void GeluPlugin::destroy(){
    delete this;
}
IPluginV2* GeluPlugin::clone() const{
    auto plugin = new GeluPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
void GeluPlugin::setPluginNamespace(const char* libNamespace){
    mNamespace = libNamespace;
}
const char* GeluPlugin::getPluginNamespace() const{
    return mNamespace.c_str();
}
GeluPluginCreator::GeluPluginCreator{
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
const char* GeluPluginCreator::getPluginName() const{
    return GELU_PLUGIN_VERSION;
}
const PluginFieldCollection* GeluPluginCreator::getFieldNames(){
    return &mFC;
}
IPluginV2* GeluPluginCreator::createPlugin(const char*name,const PluginFieldCollection* fc){
    const PluginField* fields = fc->fields;
    return new GeluPlugin(name);
}
IPluginV2* GeluPluginCreator::deserializePlugin(const char* name,const void serialData,size_t serialLength){
    return new GeluPluginCreator(name,serialData,serialLength);
}
void GeluPluginCreator::setPluginNamespace(const char* libNamespace){
    nNamespace = libNamespace;
}
const char * GeluPluginCreator::getPluginNamespace() const{
    return mNamespace.c_str();
}
REGISTER_TENSORRT_PLUGIN(GeluPluginCreator)

