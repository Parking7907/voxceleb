CUDA_VISIBLE_DEVICES=1 python ./trainSpeakerNet.py --model ResNetSE34P2 --log_input True --encoder_type SAP --trainfunc aamsoftmax --save_path exps/exp_P2_2 --nClasses 5994 --batch_size 400 --scale 30 --margin 0.1