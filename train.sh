#/bin/bash
function checkfile(){
    filename=$1
    if [ ! -f ${filename} ];then
            printf "\033[30mm %s  not exists!\33[0m" ${frozen_file}
            exit
    fi
}
function message(){
    message=${1}
    printf "\033[34m ########################################################################################################################\n\033[0m"
    printf "\033[34m ############################################## [%s] ##########################################################\n\033[0m" ${message} 
    printf "\033[34m #########################################################################################################################\n\033[0m"

}
function compile(){
    build_path=${1}
    if [ ! -d ${build_path} ];then
            mkdir -p ${build_path}
    fi
    cd ${build_path}
    cmake ..
    make
}

current_path=`pwd`
build_path=build

frozen_file=${current_path}/models/frozen_graph.pb
uff_file=${current_path}/models/frozen_graph.uff
engine_file=${current_path}/models/sample.engine
plugin_so=${build_path}/libgeluplugin.so

rm -rf build
rm -rf models

message "training_model"
python model.py
if [ -f $engine_file ];then
        rm ${engine_file}
fi
checkfile ${frozen_file}
message "compile_so"
compile ${build_path}
cd ${current_path}
message "uff_and_engine"
python gelu_plugin.py
checkfile ${uff_file}
message "load_engine"
trtexec --uff=${uff_file} --output=Identity --uffInput=x,1,28,28 --plugins=${plugin_so}
message "Process_finished!"
docker run --gpus=1 --rm -p9000:9000 -p9001:9001 -p9002:9002 -v/tmp/mnist_models:/models -v/home/biomind/custom_plugin/build:/plugins bdb0cbe1c039 sh -c 'export LD_LIBRARY_PATH=/plugins:${LD_LIBRARY_PATH} && export LD_PRELOAD=libgeluplugin.so && tritonserver --model-repository=/models'
