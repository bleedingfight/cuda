#ifndef CUSTOM_GELU_PLUGIN_H
#define CUSTOM_GELU_PLUGIN_H
#include "NvInferPlugin.h"
#include <string>
#include <vector>
using namespace nvinfer1;
class GeluPlugin:public IPluginV2{
    public:
        GeluPlugin(const std::string name);
        GeluPlugin(const std::string name,const void* data,size_t length);
        GeluPlugin() = delete;
        int getNbOutputs() const override;
        int initialize() override;
        void terminate() override;
        size_t getWorkspaceSize(int) const override{return 0;};
        int enqueue(int batchSize,const void*  const* inputs,void** outputs,void* workspace,cudaStream_t stream) override;
        size_t getSerializationSize()const override;
        void serialize(void* buffer)const override;
        void configureWithFormat(const Dims* inputDims,int nbInputs,const Dims* outputDims,int nbOutputs,DataType type,PluginFormat format,int maxBatchSize)override;
        bool supportsFormat(DataType type,PluginFormat format) const override;
        const char* getPluginType()const override;
        const char* getPluginVersion()const override;
        void destroy()override;
        nvinfer1::IPluginV2* clone() const override;
        void setPluginNamespace(const char* pluginNamespace) override;
        const char* getPluginNamespace()const override;
    private:
        const std::string mLayoerName;
        size_t mInputVolume;
        std::string nNamespace;
}
class GeluPluginCreator:public IPluginCreator{
    public:
       GeluPluginCreator();
       const char* getPluginName() const override;
       const char* getPluginVersion() const override;
       const PluginFieldCollection* getFieldNames() override;
       IPluginV2* createPlugin(const char* name,const PluginFieldCollection* fc) override;
       IPluginV2* deserializePlugin(const char* name,const void* serialData,size_t serialLength) override;
       void setPluginNamespace(const char* pluginNamesoace) override;
       const char* getPluginNamespace()const override;
    private:
        static PluginFieldCollection mFC;
        static std::vector<PluginField> mPluginAttributes;
        std::string mNamespace;
};
#endif