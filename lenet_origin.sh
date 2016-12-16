#!/bin/bash

#draw the net
#python ../../caffe/python/draw_net.py lenet_train_test.prototxt lenet_train_test.png

#train lenet
echo "start trainning..."
../../caffe/build/tools/caffe train --solver=./lenet_solver.prototxt 2>&1 | tee log/lenet_origin.log

#store the log and plot the accuracy info 
echo "parsing the log and plot the figure"
cd log/
./plot_scripts/plot_training_log.py.example 0  lenet_origin.png lenet_origin.log
