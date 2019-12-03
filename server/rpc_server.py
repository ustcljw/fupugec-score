#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jiawei Liu on 2019/03/07
import os
import time
from concurrent import futures
import logging
import json

import grpc
import numpy as np
from grpc.beta import implementations
from bert_serving.client import BertClient
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2
import xgboost as xgb

from server import rpc_server_pb2, rpc_server_pb2_grpc
from util import sys_conf, Document, sentence_tokenize

LOGGER = logging.getLogger(__name__)

ONE_DAY_SECONDS = 60 * 60 * 24


class ScoreServer(rpc_server_pb2_grpc.ScoreServerServicer):
    """

    Attributes:
        __bsp_stub: basic score的stub，用于跟basic_score模型的serving服务进行交互
        __csp_stub: coherence score的stub，用于跟coherence score模型的serving服务进行交互
        __psp_stub: prompt-relevant score的stub，用于跟prompt-relevant score模型的serving服务进行交互
        __bert_predictor: bert-as-service项目里面的BertClient对象，用于跟bert-server进行通信
    """

    def __init__(self):
        bsp_channel = implementations.insecure_channel("127.0.0.1", int(sys_conf["bsp_docker_port"]))
        self.__bsp_stub = prediction_service_pb2.beta_create_PredictionService_stub(bsp_channel)

        csp_channel = implementations.insecure_channel("127.0.0.1", int(sys_conf["csp_docker_port"]))
        self.__csp_stub = prediction_service_pb2.beta_create_PredictionService_stub(csp_channel)

        psp_channel = implementations.insecure_channel("127.0.0.1", int(sys_conf["psp_docker_port"]))
        self.__psp_stub = prediction_service_pb2.beta_create_PredictionService_stub(psp_channel)

        self.__bert_predictor = BertClient()
        self.__bst = xgb.Booster(model_file=os.path.join(sys_conf["osp_output_dir"], "osp.xgboost"))

    def __format_request(self, model_name, features):
        """ 创建basic_score, coherence_score 和 prompt-relevant score

        Args:
            model_name: 模型的名称，用于注册request
            features: doc通过bert encoding的系列特征，包括encoding，shape等等

        Returns:

        """
        if not model_name in ["bsp", "csp", "psp"]:
            raise ValueError("model_name need to be chosen from bsp, csp and psp!")

        request = predict_pb2.PredictRequest()
        request.model_spec.name = model_name
        request.inputs["doc_encodes"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["doc_encodes"], shape=features["doc_encodes"].shape))
        request.inputs["article_set"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["article_set"], shape=features["article_set"].shape))
        request.inputs["domain1_score"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["domain1_score"], shape=features["domain1_score"].shape))
        request.inputs["article_id"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["article_id"], shape=features["article_id"].shape))
        request.inputs["doc_sent_num"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["doc_sent_num"], shape=features["doc_sent_num"].shape))
        request.inputs["prompt_encodes"].CopyFrom(
            tf.contrib.util.make_tensor_proto(features["prompt_encodes"], shape=features["prompt_encodes"].shape))
        return request

    def DoMark(self, request, context):
        """ gec服务的处理函数

        Args:
            request: rpc client发送过来的请求
            context: 没啥卵用

        Returns: gec处理后的文本和相应的错误信息，json字符串格式

        """
        gec_output = json.loads(request.marker_input.strip())
        doc_sents = [gec_output["sentence_" + str(i)]["corr_sent"] for i in range(gec_output["sent_nums"])]
        doc_encodes = self.__bert_predictor.encode(doc_sents)
        prompt_sents = sentence_tokenize(gec_output["title"])
        prompt_encodes = self.__bert_predictor.encode(prompt_sents)

        # article_id, article_set在推理的时候并没有什么作用，只是为了数据类型和train保持一致
        features = dict()
        features["doc_encodes"] = np.expand_dims(doc_encodes, 0)
        features["article_set"] = np.array([9], np.int64)
        features["domain1_score"] = np.array([0], np.float32)
        features["article_id"] = np.array([0], np.int64)
        features["doc_sent_num"] = np.array([gec_output["sent_nums"]], np.int64)
        features["prompt_encodes"] = np.expand_dims(prompt_encodes, 0)
        try:
            bsp_score = self.__bsp_stub.Predict(self.__format_request("bsp", features), 10)
            bsp_score = bsp_score.outputs["batch_scores"].float_val[0]
            csp_score = self.__csp_stub.Predict(self.__format_request("csp", features), 10)
            csp_score = csp_score.outputs["batch_scores"].float_val[0]
            psp_score = self.__psp_stub.Predict(self.__format_request("psp", features), 10)
            psp_score = psp_score.outputs["batch_scores"].float_val[0]

            temp_doc = Document(gec_output)
            handcrafted_features = temp_doc.features
            doc_result = temp_doc.doc_result
            handcrafted_features.extend([bsp_score, csp_score, psp_score])
            dtest = xgb.DMatrix(handcrafted_features)
            overall_score = self.__bst.predict(dtest)[0]
            doc_result["score_lexical"] = bsp_score
            doc_result["score_coherence"] = csp_score
            doc_result["score_gramatical"] = bsp_score
            doc_result["score_task"] = psp_score
            doc_result["score_summary"] = float(overall_score)
            gec_output["score_result"] = doc_result
            return rpc_server_pb2.MarkerData(marker_output=json.dumps(gec_output))
        except:
            raise ValueError("")


def serve():
    grpc_server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    rpc_server_pb2_grpc.add_ScoreServerServicer_to_server(ScoreServer(), grpc_server)
    grpc_server.add_insecure_port("[::]:" + str(sys_conf["rpc_config"]["port"]))
    grpc_server.start()
    logging.info("Gec grpc server started!")
    try:
        while True:
            time.sleep(ONE_DAY_SECONDS)
    except KeyboardInterrupt:
        grpc_server.stop()


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s",
                        level=logging.INFO)
    serve()
