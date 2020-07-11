#!/bin/bash
set -e

source /home/ec2-user/anaconda3/bin/activate "pytorch_p36"
jupyter nbconvert "Continuous_Control.ipynb" --ExecutePreprocessor.kernel_name=python --ExecutePreprocessor.timeout=100000 --to notebook --execute
source /home/ec2-user/anaconda3/bin/deactivate
