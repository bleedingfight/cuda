/*
 * Copyright 1993-2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "customGeluPlugin.h"
#include "NvInfer.h"
#include "geluKernel.h"

#include <vector>
#include <cassert>
#include <cstring>

using namespace nvinfer1;

// Gelu plugin specific constants
namespace {
    static const char* GELU_PLUGIN_VERSION{"1"}; 
    static const char* GELU_PLUGIN_NAME{"CustomGeluPlugin"};
}

// Static class fields initialization
PluginFieldCollection GeluPluginCreator::mFC{};
std::vector<PluginField> GeluPluginCreator::mPluginAttributes;


// Helper function for serializing plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val)
{
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// 辅助插件序列化函数
template<typename T>
T readFromBuffer(const char*& buffer)
{
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}
// 插件构造函数，根据构造函数的名称构造插件
GeluPlugin::GeluPlugin(const std::string name)
    : mLayerName(name)
{
}
// 插件构造函数，根据构造函数的名称，构造函数的输入数据构造插件
GeluPlugin::GeluPlugin(const std::string name, const void* data, size_t length)
    : mLayerName(name)
{
    // Deserialize in the same order as serialization
    const char *d = static_cast<const char *>(data);
    const char *a = d;

    assert(d == (a + length));
}
// 获取插件的类型，插件名称
const char* GeluPlugin::getPluginType() const
{
    return GELU_PLUGIN_NAME;
}
// 获取插件的版本，用以优化插件
const char* GeluPlugin::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}
// 插件输出个数
int GeluPlugin::getNbOutputs() const
{
    return 1;
}
// 插件输出维度
Dims GeluPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    // 确定插件的输入个数为1
    assert(nbInputDims == 1); 
    assert(index == 0);

    // Gelu 并不改变插件的维度，输出维度和输入维度相同
    return *inputs;
}
// 插件初始化操作
int GeluPlugin::initialize()
{
    return 0;
}
// 插件的执行
int GeluPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream)
{
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // 启动CUDA和函数执行插件的计算操作，然今后将计算结果写入输出，同时返回计算是否成功的状态。
    status = geluInference(stream, mInputVolume * batchSize, inputs[0], output);

    return status;
}
// 序列化插件的大小，此插件不需要任何参数，序列化插件的大小为0
size_t GeluPlugin::getSerializationSize() const
{
    return 0;
}
// 序列化插件
void GeluPlugin::serialize(void* buffer) const 
{
    // 将buffer转化为bytes
    char *d = static_cast<char *>(buffer);
    const char *a = d;

    // 保证buffer的size的地址计算正确
    assert(d == a + getSerializationSize());
}

void GeluPlugin::configureWithFormat(const Dims* inputs, int nbInputs, const Dims* outputs, int nbOutputs, DataType type, PluginFormat format, int)
{
    // 确定输出的个数为1
    assert(nbOutputs == 1);
    // 输出的类型为float型
    assert(type == DataType::kFLOAT);
    // 输出数据的格式为WCHW
    assert(format == PluginFormat::kNCHW);

    // 获取输入的Size
    size_t volume = 1;
    for (int i = 0; i < inputs->nbDims; i++) {
        volume *= inputs->d[i];
    }
    mInputVolume = volume;
}
// 输入数据的格式
bool GeluPlugin::supportsFormat(DataType type, PluginFormat format) const
{
    // 这个插件仅仅支持数据类型为 floats 同时输入格式为 NCHW 的数据
    if (type == DataType::kFLOAT && format == PluginFormat::kNCHW)
        return true;
    else
        return false;
}

void GeluPlugin::terminate() {}

void GeluPlugin::destroy() {
    // 销毁插件的析构函数
    delete this;
}
// 创建构造函数
IPluginV2* GeluPlugin::clone() const
{
    auto plugin = new GeluPlugin(mLayerName);
    plugin->setPluginNamespace(mNamespace.c_str());
    return plugin;
}
// 设置插件的名称
void GeluPlugin::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}
// 获取插件的名称
const char* GeluPlugin::getPluginNamespace() const
{
    return mNamespace.c_str();
}

GeluPluginCreator::GeluPluginCreator()
{
    // Fill PluginFieldCollection with PluginField arguments metadata
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}
// 获取插件的名称
const char* GeluPluginCreator::getPluginName() const
{
    return GELU_PLUGIN_NAME;
}
// 获取插件的版本
const char* GeluPluginCreator::getPluginVersion() const
{
    return GELU_PLUGIN_VERSION;
}
// 获取插件的FieldCollection
const PluginFieldCollection* GeluPluginCreator::getFieldNames()
{
    return &mFC;
}
// 创建插件
IPluginV2* GeluPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    const PluginField* fields = fc->fields;

    return new GeluPlugin(name);
}
// 反序列化插件
IPluginV2* GeluPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    return new GeluPlugin(name, serialData, serialLength);
}
// 插值插件的namespace
void GeluPluginCreator::setPluginNamespace(const char* libNamespace) 
{
    mNamespace = libNamespace;
}
// 获取插件的namespace
const char* GeluPluginCreator::getPluginNamespace() const
{
    return mNamespace.c_str();
}

REGISTER_TENSORRT_PLUGIN(GeluPluginCreator);
