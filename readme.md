# peg-in-hole-visual-servoing-model

#### install dependencies
synth-ml:  
https://gitlab.com/sdurobotics/vision/synth-ml  
and other requirements:  
``pip3 install -r requirements.txt``

*(instructions assumes theres a gpu available)*

#### generate synthetic data
``./synth_data.sh``

#### train model
``python3 train_model.py``


#### usage
see [peg-in-hole-visual-servoing](https://github.com/RasmusHaugaard/peg-in-hole-visual-servoing).