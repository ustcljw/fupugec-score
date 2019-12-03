This is the project for automated scoring articles.

##打分思路
    1、两阶段学习策略，第一阶段中利用lstm得到semantic score (basic score), coherence score 和 prompt relevant score,
    2、第二阶段利用第一阶段中学到的几个score 和 文章的人工特征进行拼接，基于xgboost进行进一步学习
    详细打分参考论文地址: https://arxiv.org/abs/1901.07744

##文件介绍
    1、score.py: semantic score, coherence score, prompt relevant score 和 overall score的训练和测试相关内容 
    2、util.py: 项目中需要的一些工具类函数，包括读取训练数据到标准格式（tfrecord, xgboost的训练numpy,）， Document类，封装gec的结果。
    3、bert: bert-as-service比较老的版本的。。用来read tfrecord，
    4、config: 模型训练，推理，存储路径所用的参数配置文件
    5、server: service模块。暴露对外的接口

##训练
    1、python3.5 score.py -model bsp #训练basic score模型
    2、python3.5 score.py -model csp #训练coherence score模型
    3、python3.5 score.py -model psp #训练prompt relevant score模型
    4、python3.5 score.py -model osp #训练overall score模型
    (训练或者测试的时候，需要提前在config/* 下面的相关配置文件中进行配置)
    
##准备工作
    本项目使用tensorflow serving进行服务，tensorflow serving使用如下所示（以basic_score模型为例）：    
    1、docker pull tensorflow/serving
    2、sudo docker run -p 8500:8500 -p 8501:8501 --name bsp_container -v /data/liujiawei/eilts_score/basic_score/SavedModel:/models/bsp -e MODEL_NAME=bsp -t tensorflow/serving &
    实际运行需要load三个tensorflow serving 服务，分别启用basic score, coherence score 和 prompt-relevant score的服务
    3、查看启动的tensorflow serving服务有没有成功: curl http://127.0.0.1:8501/v1/models/bsp (以bsp模型为例)
    4、启动bert-as-service第三方项目，做encoding.

##外部调用：
    外部项目通过grpc对本项目请求服务，当外部发送请求，本项目将请求中的文章分别发送到上述三个tensorflow serving服务中，将返回的结果与手工
    特征进行拼接，利用预先训练的xgboost模型求解最终的