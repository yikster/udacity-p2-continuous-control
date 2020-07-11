#!/bin/bash
source activate pytorch_p36

# CURR is my project repository
export CURR=`pwd`
echo $CURR
cd ../
git clone https://github.com/Unity-Technologies/ml-agents.git
cd ml-agents/ml-agents-envs
pip3 install -e .
cd ../ml-agents
pip3 install -e .
pip3 install unityagents
pip3 install gym torch torchsummary
cd $CURR
#wget https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip
#unzip Banana_Linux.zip

