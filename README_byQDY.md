### 3. 打造自己的图像识别模型

#### 运行方法

一、**3.2 数据准备**
	一是把数据集切分为训练集和验证集，二是转换为tfrecord格式。

在data_prepare/pic/中，分为train和validation两个目录来保存原始的图像（.jpg）
在data_prepare/中。使用data_convert.py将图片转换为tfrecord格式：
	
首先需要将数据转换成tfrecord的形式。在data_prepare文件夹下，运行：
```
python data_convert.py -t pic/ \
  --train-shards 2 \
  --validation-shards 2 \
  --num-threads 2 \
  --dataset-name satellite
```

这样在pic文件夹下就会生成4个tfrecord文件和1个label.txt文件。

-t pic/ 表示转换pic文件夹中的数据，pic文件夹中必须有一个train目录和validation目录，每个目录下按类别存放图像数据
--train-shards 2 表示将数据集分为两份，即最后的训练数据就是两个tfrecord格式的文件。（可分为更多）
--validation-shards 2 将验证集分为两份
--num-threads 2 表示采用两个线程产生的数据，线程数必须要能整除train-shards 和--validation-shards 2
--dataset-name satellite 将生成的数据集起名为satellite

二、**3.3.2 定义新的datasets 文件**  slim/datasets/satellite.py
		在同目录的dataset_factory.py中注册satellite数据库：from datasets import satellite
					datasets_map ={'satellite':satellite}
					
	
参考3.3.2小节对Slim源码做修改。

三、**3.3.3 准备训练文件夹**

在slim文件夹下新建一个satellite目录。在这个目录下做下面几件事情：
1、 新建一个data 目录，并将第3.2中准备好的5个转换好格式的训练数据复制进去。
2、 新建一个空的train_dir目录，用来保存训练过程中的日志和模型。
3、 新建一个pretrained目录，在slim的GitHub页面找到Inception V3 模型的下载地址http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz 下载并解压后，会得到一个inception_v3.ckpt 文件，将该文件复制到pretrained 目录下（这个文件在chapter_3_data/文件中也提供了）

四、**3.3.4 开始训练**

（在slim文件夹下运行）训练Logits层：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=2 \
  --log_every_n_steps=10 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
```

训练所有层：
```
python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v3 \
  --checkpoint_path=satellite/pretrained/inception_v3.ckpt \
  --checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
  （去掉了--trainable_scopes）
```

--train_dir=satellite/train_dir \ 在satellite/train_dir目录下保存日志和checkpoint
--dataset_name=satellite \ 指定训练的数据集
--dataset_split_name=train \ 同上
--dataset_dir=satellite/data \ 指定训练数据集保存的位置
--model_name=inception_v3 \	使用的模型名称
--checkpoint_path=satellite/pretrained/inception_v3.ckpt \ 预训练模型的保存位置
--checkpoint_exclude_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \  在恢复预训练模型时，不恢复这两层
																这两层对应InceptionV3模型的末端层，与当前模型不符
--trainable_scopes=InceptionV3/Logits,InceptionV3/AuxLogits \ 规定了在模型中微调变量的范围，这里表示
					只对InceptionV3/Logits,InceptionV3/AuxLogits两个变量进行微调，其他变量保持不变。
					InceptionV3/Logits,InceptionV3/AuxLogits相当于末端层，如果不设定就会对所有参数进行训练。
--max_number_of_steps=100000 \ 最大训练步骤100000
--batch_size=32 \ 每步使用的batch数
--learning_rate=0.001 \ 学习速率
--learning_rate_decay_type=fixed \ 学习率是否自动下降，此时固定
--save_interval_secs=300 \ 每隔300s，把当前模型保存到train_dir中
--save_summaries_secs=2 \ 每2秒将日志写入到train_dir中（考虑性能可设置较长时间）
--log_every_n_steps=10 \ 每10步打印训练信息
--optimizer=rmsprop \ 选定的优化器
--weight_decay=0.00004 模型中所有参数的二次正则化超参数
  
  
五、**3.3.6 验证模型准确率**

在slim文件夹下运行：
```
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v3
```
--checkpoint_path=satellite/train_dir \ 既可以接受一个目录的路径，也可以接受一个文件的路径。
							若是目录satellite/train_dir，则会在目录中寻找最新保存的模型文件进行验证
							也可以指定一个模型进行验证，如satellite/train_dir/modrl.ckpt-10000
--eval_dir=satellite/eval_dir \ 执行结果的日志就保存在satellite/eval_dir中
--dataset_name=satellite \ 指定需要执行的训练集
--dataset_split_name=validation \ 此处使用验证集
--dataset_dir=satellite/data \ 数据集保存的位置
--model_name=inception_v3 使用的模型名称

六、**3.3.7 TensorBoard 可视化与超参数选择**

打开TensorBoard：
```
tensorboard --logdir satellite/train_dir
```

七、**3.3.8 导出模型并对单张图片进行识别**

在slim文件夹下运行：
```
python export_inference_graph.py \
  --alsologtostderr \
  --model_name=inception_v3 \
  --output_file=satellite/inception_v3_inf_graph.pb \ 
  --dataset_name satellite
```

在chapter_3文件夹下运行（需将5271改成train_dir中保存的实际的模型训练步数）：
```
python freeze_graph.py \
  --input_graph slim/satellite/inception_v3_inf_graph.pb \
  --input_checkpoint slim/satellite/train_dir/model.ckpt-5271 \
  --input_binary true \
  --output_node_names InceptionV3/Predictions/Reshape_1 \
  --output_graph slim/satellite/frozen_graph.pb
```

--input_graph slim/satellite/inception_v3_inf_graph.pb \ 表示使用的网结构文件
--input_checkpoint slim/satellite/train_dir/model.ckpt-5271 \ 具体将哪一个checkpoint的参数载入网络结构中
--input_binary true \	导入的inception_v3_inf_graph.pb实际是一个protobuf文件，有文本和二进制两种保存格式。
						这里是二进制形式，所以是true
--output_node_names InceptionV3/Predictions/Reshape_1 \	在导出的模型中指定一个输出节点，这里的是最后的输出层
--output_graph slim/satellite/frozen_graph.pb	将最后导出的模型保存为frozen_graph.pb

运行导出模型分类单张图片：
```
python classify_image_inception_v3.py \
  --model_path slim/satellite/frozen_graph.pb \
  --label_path data_prepare/pic/label.txt \
  --image_file test_image.jpg
```

--model_path slim/satellite/frozen_graph.pb \  就是之前导出的模型
--label_path data_prepare/pic/label.txt \	指定了一个label文件，label文件中按顺序存储了各个类别的名称
											这样脚本就可以吧类别的ID号转换为实际的类别名
--image_file test_image.jpg	是需要测试的单张图片


#### 拓展阅读

- TensorFlow Slim 是TensorFlow 中用于定义、训练和验证复杂网络的 高层API。官方已经使用TF-Slim 定义了一些常用的图像识别模型， 如AlexNet、VGGNet、Inception模型、ResNet等。本章介绍的Inception V3 模型也是其中之一， 详细文档请参考： https://github.com/tensorflow/models/tree/master/research/slim。
- 在第3.2节中，将图片数据转换成了TFRecord文件。TFRecord 是 TensorFlow 提供的用于高速读取数据的文件格式。读者可以参考博文（ http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/ ）详细了解如何将数据转换为TFRecord 文件，以及 如何从TFRecord 文件中读取数据。
- Inception V3 是Inception 模型（即GoogLeNet）的改进版，可以参考论文Rethinking the Inception Architecture for Computer Vision 了解 其结构细节。


InceptionV4:

python train_image_classifier.py \
  --train_dir=satellite/train_dir \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=inception_v4 \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004


python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_v4
 
 
python export_inference_graph.py  --alsologtostderr --model_name=inception_v4 --output_file=satellite/inception_v4_inf_graph.pb  --dataset_name satellite
  
python freeze_graph.py --input_graph slim/satellite/inception_v4_inf_graph.pb --input_checkpoint slim/satellite/train_dir/model.ckpt-100000  --input_binary true  --output_node_names InceptionV4/Logits/Predictions --output_graph slim/satellite/frozen_IV4_graph.pb  

python classify_image_inception_v4.py --model_path slim/satellite/frozen_IV4_graph.pb --label_path data_prepare/pic/label.txt --image_file test_image.jpg
###

Inception_resnet_v2:

python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=inception_resnet_v2

python export_inference_graph.py  --alsologtostderr --model_name=inception_resnet_v2 --output_file=satellite/inception_resnet_v2_graph.pb  --dataset_name satellite  

python freeze_graph.py --input_graph slim/satellite/inception_resnet_v2_graph.pb --input_checkpoint slim/satellite/train_dir/model.ckpt-100000  --input_binary true  --output_node_names InceptionResnetV2/Logits/Predictions --output_graph slim/satellite/frozen_inception_resnet_v2_graph.pb

python classify_image_inception_resnet_v2.py --model_path slim/satellite/frozen_inception_resnet_v2_graph.pb --label_path data_prepare/pic/label.txt --image_file test_image.jpg

###

Resnet_v2_152

python train_image_classifier.py \
  --train_dir=satellite/train_dir_resv2 \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=resnet_v2_152 \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004

python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir_resv2 \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=resnet_v2_152
 
python export_inference_graph.py  --alsologtostderr --model_name=resnet_v2_152 --output_file=satellite/resnet_v2_152_graph.pb  --dataset_name satellite --image_size=299

python freeze_graph.py --input_graph slim/satellite/resnet_v2_152_graph.pb --input_checkpoint slim/satellite/train_dir_resv2/model.ckpt-100000  --input_binary true  --output_node_names resnet_v2_152/Predictions/Softmax --output_graph slim/satellite/frozen_RV2_152_graph.pb

python classify_image_resnet_v2_152.py --model_path slim/satellite/frozen_RV2_152_graph.pb --label_path data_prepare/pic/label.txt --image_file test_image.jpg
#
PNAS

python train_image_classifier.py \
  --train_dir=satellite/train_dir_nas \
  --dataset_name=satellite \
  --dataset_split_name=train \
  --dataset_dir=satellite/data \
  --model_name=pnasnet_large \
  --max_number_of_steps=100000 \
  --batch_size=32 \
  --learning_rate=0.001 \
  --learning_rate_decay_type=fixed \
  --save_interval_secs=300 \
  --save_summaries_secs=10 \
  --log_every_n_steps=1 \
  --optimizer=rmsprop \
  --weight_decay=0.00004
  
python eval_image_classifier.py \
  --checkpoint_path=satellite/train_dir_nas \
  --eval_dir=satellite/eval_dir \
  --dataset_name=satellite \
  --dataset_split_name=validation \
  --dataset_dir=satellite/data \
  --model_name=pnasnet_large
  
python export_inference_graph.py  --alsologtostderr --model_name=pnasnet_large --output_file=satellite/pnasnet_large_graph.pb  --dataset_name satellite --image_size=299

python freeze_graph.py --input_graph slim/satellite/pnasnet_large_graph.pb --input_checkpoint slim/satellite/train_dir_nas/model.ckpt-100000  --input_binary true  --output_node_names final_layer/predictions --output_graph slim/satellite/frozen_pnasnet_large_graph.pb

python classify_image_pnasnet_large.py --model_path slim/satellite/frozen_pnasnet_large_graph.pb --label_path data_prepare/pic/label.txt --image_file test_image.jpg