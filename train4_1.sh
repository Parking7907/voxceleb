CUDA_VISIBLE_DEVICES=0 python ./trainSpeakerNet.py --model ResNetSE34P1 --log_input True --encoder_type SAP --trainfunc aamsoftmax --save_path exps/exp_P1_1 --nClasses 5994 --batch_size 400 --scale 30 --margin 0.3