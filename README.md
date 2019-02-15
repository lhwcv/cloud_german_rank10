# Alibaba Cloud German rank10 baseline
  Detail works please refer [tianchi forum](https://tianchi.aliyun.com/forum/postDetail?spm=5176.12282027.0.0.53981580RCjNVb&postId=46819).
  This repo is a simplified seresnet-50 single model,
  which trained in 128 image size,achieved 0.87+ in four
  test set in average.
  
  To achieve our final stage2-a 0.887, stage2-b 0.884 score,
  you may need   add the following works:
  1. Extarct seresnet-50-128 model's fc feature of trainval data
     then use kmeans to get 5 folds, which can relieve heavy 
     repeatness.(It seams 4 thousands up in single model)
  2. Train simplified res50 seres50 xcep incepres in scale [112,128,144,160]
      and ensemble.
  3. Predict softmax label of all testset, use threshold >0.85 to keep
     some semi-data, add to trainset and finetune or retrain the models.
     
# run
  mkdir ./data<br>
  mkdir ./preprocess_dir<br>
  (put the h5 raw file into data)<br>
  python preprocess.py<br>
  python train.py<br>
