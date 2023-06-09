*********************************************************************
# Hyperparameter tuning for ResNet50 on ImageNet image classification
*********************************************************************

What is needed: 
- Nomad binaries (v4.3 or above).
- ImageNet dataset. Need to change the path to dataset in bb.py
- Mindspore v2.0
- GPU

How to run:
$NOMAD_HOME/bin/nomad param.txt

It is possible to perform a hot restart using a history file (created 
by Nomad)
