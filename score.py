#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by Jiawei Liu on 2018/06/18

import os
import logging

import random
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb

from bert import optimization
from qwk import *
from util import *

Logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s %(name)s %(levelname)s %(message)s", level=logging.INFO)


class BasicScorePredictor:
    """ essay的semantic方面的评分，也称基础得分，后续的overall score是在其基础上进一步得到的

    Attributes:
        bert_emb_dim: 因为输入的句子是用bert进行encoded， 该属性记录了所用bert模型的输出维度
        dropout_prob: dropout的概率
        lstm_layers_num: lstm的层数
        lstm_hidden_size: lstm的隐层单元的宽度
        fnn_hidden_size: 输出接的fnn的每个隐层的宽度，
        bidirectional: 标注是否使用双向模型
        tdnn_step: 如果使用tdnn的话，tdnn跳跃的步长

    """

    def __init__(self):
        self.bert_emb_dim = 768
        self.dropout_prob = 0.3
        self.lstm_layers_num = 1
        self.lstm_hidden_size = 1024
        self.fnn_hidden_size = []
        self.bidirectional = False
        self.tdnn_step = 4

    @staticmethod
    def generate_asap_train_and_test_set(generate_type="shuffle_all"):
        """ 生成asap的训练数据集合

        Args:
            generate_type: 以何种方式产生训练集和测试集，
                           shuffle_prompt: 表示从某个prompt下，选取一定的数据训练，
                           other_prompt: 表示利用所有别的prompt下的样本对该样本进行训练，一般不推荐此种方式，效果最差
                           shuffle_all: 表示shuffle所有的样本，然后抽样，为缺省方式

        Returns:
            articles_id: 文章的id
            articles_set: 文章的所属的prompt的id集合
            handmark_scores: 手工标注的分数
            correspond_train_id_set: 分割的训练集id
            correspond_test_id_set: 分割的测试集id
        """
        articles_id, articles_set, set_ids, handmark_scores = read_asap_dataset()
        np.random.seed(train_conf["random_seed"])
        if generate_type == "shuffle_prompt":
            # 在每个set内80%用来训练，20%用来测试
            permutation_ids = np.random.permutation(set_ids[train_conf["prompt_id"]])
            correspond_train_id_set = permutation_ids[
                                      0:int(len(permutation_ids) * train_conf["train_set_prob"])]
            correspond_test_id_set = permutation_ids[
                                     int(len(permutation_ids) * train_conf["train_set_prob"]):]
        elif generate_type == "other_prompt":
            # 对每个set，用其他set的数据进行训练
            correspond_test_id_set = set_ids[train_conf["prompt_id"]]
            correspond_train_id_set = []
            for i in range(1, 9):
                if i == train_conf["prompt_id"]:
                    continue
                else:
                    correspond_train_id_set.extend(set_ids[i])
        elif generate_type == "shuffle_all":
            # 将所有set的数据混合打散，取80%进行训练，剩下测试
            permutation_ids = np.random.permutation(articles_id)
            correspond_test_id_set = permutation_ids[int(len(articles_id) * train_conf["train_set_prob"]):]
            correspond_train_id_set = permutation_ids[:int(len(articles_id) * train_conf["train_set_prob"])]
        else:
            raise ValueError("generate_type must be choose in ('shuffle_prompt','other_prompt','shuffle_all')")
        return articles_id, articles_set, handmark_scores, correspond_train_id_set, correspond_test_id_set

    def build_graph(self, batch_doc_encodes,
                    batch_doc_sent_nums,
                    batch_article_set,
                    batch_domain1_score,
                    batch_size,
                    is_training):
        """ 建立模型的图

        Args:
            batch_doc_encodes: 一个batch的文章的bert encoding的结果
            batch_doc_sent_nums: 记录一个batch内每个doc的句数
            batch_article_set: 记录一个batch内每个doc所属的类别，按prompt分
            batch_domain1_score: 记录一个batch内每个doc的人工评分
            batch_size: 将batch的大小放在图中，
            is_training: 标致是否为train的状态

        Returns: loss 和 logits

        """

        def normalize_value(score, min_value, max_value):
            result = tf.div(tf.subtract(score, min_value), tf.to_float(tf.subtract(max_value, min_value)))
            return result

        batch_index_o = tf.constant(0)
        standard_batch_domain_score_o = tf.convert_to_tensor([])

        def cond(batch_index, normalized_score):
            return tf.less(batch_index, batch_size)

        def body(batch_index, normalized_score):
            min_value = tf.convert_to_tensor([-1, 2, 1, 0, 0, 0, 0, 0, 0, 0], tf.float32)[
                batch_article_set[batch_index]]
            max_value = tf.convert_to_tensor([-1, 12, 6, 3, 3, 4, 4, 30, 60, 9], tf.float32)[
                batch_article_set[batch_index]]
            temp_score = batch_domain1_score[batch_index]
            temp_score = normalize_value(tf.to_float(temp_score), min_value, max_value)
            normalized_score = tf.concat([normalized_score, [temp_score]], axis=0)
            return tf.add(batch_index, 1), normalized_score

        _, standard_batch_domain_score = tf.while_loop(cond,
                                                       body,
                                                       [batch_index_o, standard_batch_domain_score_o],
                                                       shape_invariants=[batch_index_o.get_shape(),
                                                                         tf.TensorShape([None])])

        if self.bidirectional:
            fw_cell, bw_cell = create_rnn_cell(self.lstm_hidden_size,
                                               self.dropout_prob,
                                               self.lstm_layers_num,
                                               self.bidirectional,
                                               is_training)
            (output_fw, output_bw), states = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_cell,
                                                                             cell_bw=bw_cell,
                                                                             inputs=batch_doc_encodes,
                                                                             sequence_length=batch_doc_sent_nums,
                                                                             dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=2)

            # padding for output is 0, hence can mean directly
            mean_time_output = tf.reduce_mean(output, axis=1)
            w = tf.get_variable(shape=[self.lstm_hidden_size * 2, 1],
                                initializer=create_initializer(),
                                name="weight",
                                dtype=tf.float32)
            b = tf.get_variable(initializer=tf.zeros_initializer(),
                                shape=[1],
                                name="bias",
                                dtype=tf.float32)
            logit = tf.squeeze(tf.sigmoid(tf.matmul(mean_time_output, w) + b))
            loss = tf.losses.mean_squared_error(labels=standard_batch_domain_score, predictions=logit)
        else:
            fw_cell = create_rnn_cell(self.lstm_hidden_size,
                                      self.dropout_prob,
                                      self.lstm_layers_num,
                                      self.bidirectional,
                                      is_training)
            output, states = tf.nn.dynamic_rnn(cell=fw_cell,
                                               inputs=batch_doc_encodes,
                                               sequence_length=batch_doc_sent_nums,
                                               dtype=tf.float32)
            # 增加TDNN
            if 0:
                with tf.variable_scope("tdnn"):
                    tdnn_fw_cell = create_rnn_cell(self.lstm_hidden_size,
                                                   self.dropout_prob,
                                                   self.lstm_layers_num,
                                                   self.bidirectional,
                                                   is_training)
                    tdnn_input_slice = tf.range(0, tf.reduce_max(batch_doc_sent_nums), self.tdnn_step)
                    tdnn_input = tf.gather(output, tdnn_input_slice, axis=1)

                    batch_index_o = tf.constant(0)
                    actual_length_o = tf.convert_to_tensor([])

                    def cond(batch_index, actual_length):
                        return tf.less(batch_index, batch_size)

                    def body(batch_index, actual_length):
                        temp_slice = tf.range(0, batch_doc_sent_nums[batch_index], self.tdnn_step)
                        actual_length = tf.concat([actual_length, [tf.shape(temp_slice)[0]]], axis=0)
                        return tf.add(batch_index, 1), actual_length

                    _, actual_length = tf.while_loop(cond,
                                                     body,
                                                     [batch_index_o, actual_length_o],
                                                     shape_invariants=[batch_index_o.get_shape(),
                                                                       tf.TensorShape([None])])

                    output, states = tf.nn.dynamic_rnn(cell=tdnn_fw_cell,
                                                       inputs=tdnn_input,
                                                       sequence_length=actual_length,
                                                       dtype=tf.float32)

            # 因为cell没有设置num_proj，所以hidden不会被投影， output等于states， output用0padding, states用最后一个state来填充，
            # 本处使用hiddenstate的mean，和最后一个hidden分别试验。
            # mean_time_output = tf.reduce_mean(output, axis=1)
            # last_time_hidden = states
            # mean_time_output = tf.reduce_mean(batch_doc_encodes, axis=1)
            mean_time_output = states[0].h

            for hs in self.fnn_hidden_size:
                if is_training:
                    mean_time_output = tf.nn.dropout(mean_time_output, keep_prob=1 - self.dropout_prob)
                mean_time_output = tf.layers.dense(mean_time_output,
                                                   hs,
                                                   activation=tf.nn.relu,
                                                   kernel_initializer=create_initializer())

            if self.fnn_hidden_size:
                x_dim = self.fnn_hidden_size[-1]
            else:
                x_dim = self.lstm_hidden_size

            w = tf.get_variable(shape=[x_dim, 1],
                                initializer=create_initializer(),
                                name="weight",
                                dtype=tf.float32)
            b = tf.get_variable(initializer=tf.zeros_initializer(),
                                shape=[1],
                                name="bias",
                                dtype=tf.float32)
            if is_training:
                mean_time_output = tf.nn.dropout(mean_time_output, keep_prob=1 - self.dropout_prob)
            logit = tf.squeeze(tf.sigmoid(tf.matmul(mean_time_output, w) + b))
            loss = tf.losses.mean_squared_error(labels=standard_batch_domain_score, predictions=logit)

        return loss, logit

    def model_fn_builder(self, learning_rate, num_train_step, num_warmup_steps):
        """

        Args:
            learning_rate: 学习速率
            num_train_step: 学习步数
            num_warmup_steps: 预热的步数

        Returns: 函数model_fn的函数句柄

        """

        def model_fn(features, labels, mode, params):
            batch_doc_encodes = tf.identity(features["doc_encodes"])  # shape = [batch_size, None, bert_dim]
            batch_article_set = tf.identity(features["article_set"])  # shape = [batch_size]
            batch_doc_sent_nums = tf.identity(features["doc_sent_num"])  # shape = [batch_size]
            batch_domain1_score = tf.identity(features["domain1_score"])  # shape = [batch_size]
            batch_doc_id = tf.identity(features["article_id"])  # shape = [batch_size]
            batch_size = tf.shape(batch_doc_sent_nums)[0]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            loss, logit = self.build_graph(batch_doc_encodes,
                                           batch_doc_sent_nums,
                                           batch_article_set,
                                           batch_domain1_score,
                                           batch_size,
                                           is_training)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(loss=loss,
                                                         init_lr=learning_rate,
                                                         num_train_steps=num_train_step,
                                                         num_warmup_steps=num_warmup_steps,
                                                         use_tpu=False)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op
                )
            elif mode == tf.estimator.ModeKeys.EVAL:
                # predictions = logit
                # accuracy = tf.metrics.accuracy(fnn_labels, predictions)
                # eval_metrics = {
                #    "eval_accuracy": accuracy
                # }
                # tf.summary.scalar("eval_accuracy", accuracy)
                # output_spec = tf.estimator.EstimatorSpec(
                #    mode=mode,
                #    loss=loss,
                #    eval_metric_ops=eval_metrics
                # )
                pass
            else:
                predictions = {
                    "batch_scores": logit,
                    "batch_doc_id": batch_doc_id,
                    "batch_article_set": batch_article_set
                }
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={
                        "predictions": tf.estimator.export.PredictOutput(predictions)
                    }
                )
            return output_spec

        return model_fn

    @staticmethod
    def eval_metric(result, handmark_scores, articles_set, articles_id):
        """ 评价函数

        Args:
            result: predictor 推理出的结果
            handmark_scores: 手工标注的结果，会被约束到0-10的范围
            articles_set: 文章所属的prompt_id的集合，list类似
            articles_id: 文章自己的id.

        Returns: qwk的指标

        """
        predict_scores = {}
        for item in result:
            min_value = 0
            max_value = 10
            normalize_score = item["batch_scores"]
            overall_score = round(normalize_score * (max_value - min_value) + min_value)
            predict_scores[item["batch_doc_id"]] = overall_score

        test_handmark_scores = []
        test_predict_scores = []

        for key, value in predict_scores.items():
            article_set_id = articles_set[articles_id.index(key)]
            min_value = dataset_score_range[article_set_id][0]
            max_value = dataset_score_range[article_set_id][1]
            hs = handmark_scores[key]
            temp_hs = round(((hs - min_value) / (max_value - min_value)) * 10)
            test_predict_scores.append(value)
            test_handmark_scores.append(temp_hs)
            # print("id: {}, predict: {}, handmark: {}".format(key, value, handmark_scores[key]))

        test_handmark_scores = np.asarray(test_handmark_scores, dtype=np.int32)
        test_predict_scores = np.asarray(test_predict_scores, dtype=np.int32)

        qwk = quadratic_weighted_kappa(test_predict_scores, test_handmark_scores)
        print("##############qwk value is {}".format(qwk))


class CoherenceScore:
    """ essay的coherence方面的评分，也称连贯性得分，后续的overall score是在其基础上进一步得到的

    Attributes:
        bert_emb_dim: 因为输入的句子是用bert进行encoded， 该属性记录了所用bert模型的输出维度
        dropout_prob: dropout的概率
        lstm_layers_num: lstm的层数
        lstm_hidden_size: lstm的隐层单元的宽度
        fnn_hidden_size: 输出接的fnn的每个隐层的宽度，
        bidirectional: 标注是否使用双向模型

    """

    def __init__(self):
        self.bert_emb_dim = 768
        self.dropout_prob = 0.5
        self.lstm_hidden_size = 1024
        self.lstm_layers_num = 1
        self.fnn_hidden_size = []
        self.bidirectional = False

    @staticmethod
    def generate_asap_train_and_test_set(generate_type="shuffle_all"):
        """ 生成asap的训练数据集合

        Args:
            generate_type: 以何种方式产生训练集和测试集，
                           shuffle_prompt: 表示从某个prompt下，选取一定的数据训练，
                           other_prompt: 表示利用所有别的prompt下的样本对该样本进行训练，一般不推荐此种方式，效果最差
                           shuffle_all: 表示shuffle所有的样本，然后抽样，为缺省方式

        Returns:
            articles_id: 文章的id
            articles_set: 文章的所属的prompt的id集合
            handmark_scores: 手工标注的分数
            correspond_train_id_set: 分割的训练集id
            correspond_test_id_set: 分割的测试集id
        """
        articles_id, articles_set, set_ids, handmark_scores = read_asap_dataset()
        np.random.seed(train_conf["random_seed"])

        if generate_type == "shuffle_prompt":
            # 在每个set内80%用来训练，20%用来测试
            permutation_ids = np.random.permutation(set_ids[train_conf["prompt_id"]])
            correspond_train_id_set = permutation_ids[
                                      0:int(len(permutation_ids) * train_conf["train_set_prob"])]
            correspond_test_id_set = permutation_ids[
                                     int(len(permutation_ids) * train_conf["train_set_prob"]):]
        elif generate_type == "other_prompt":
            # 对每个set，用其他set的数据进行训练
            correspond_test_id_set = set_ids[train_conf["prompt_id"]]
            correspond_train_id_set = []
            for i in range(1, 9):
                if i == train_conf["prompt_id"]:
                    continue
                else:
                    correspond_train_id_set.extend(set_ids[i])

        elif generate_type == "shuffle_all":
            # 将所有set的数据混合打散，取80%进行训练，剩下测试
            permutation_ids = np.random.permutation(articles_id)
            correspond_test_id_set = permutation_ids[int(len(articles_id) * train_conf["train_set_prob"]):]
            correspond_train_id_set = permutation_ids[:int(len(articles_id) * train_conf["train_set_prob"])]
        else:
            raise ValueError("generate_type must be choose in ('shuffle_prompt','other_prompt','shuffle_all')")

        correspond_train_id_set = list(correspond_train_id_set)
        correspond_test_id_set = list(correspond_test_id_set)
        negative_train_id_permuted = [item + 100000 for item in correspond_train_id_set]
        negative_test_id_permuted = [item + 100000 for item in correspond_test_id_set]
        correspond_train_id_set.extend(negative_train_id_permuted)
        correspond_test_id_set.extend(negative_test_id_permuted)

        return articles_id, articles_set, handmark_scores, correspond_train_id_set, correspond_test_id_set

    def build_graph(self,
                    batch_doc_encodes,
                    batch_doc_sent_nums,
                    batch_article_set,
                    batch_domain1_score,
                    batch_doc_id,
                    batch_size,
                    is_training):
        """ 建立模型的图

        Args:
            batch_doc_encodes: 一个batch的文章的bert encoding的结果
            batch_doc_sent_nums: 记录一个batch内每个doc的句数
            batch_article_set: 记录一个batch内每个doc所属的类别，按prompt分
            batch_domain1_score: 记录一个batch内每个doc的人工评分
            batch_doc_id: 记录一个batch内每个doc的id
            batch_size: 将batch的大小放在图中，
            is_training: 标致是否为train的状态

        Returns: loss 和 logits

        """

        def normalize_value(score, min_value, max_value):
            result = tf.div(tf.subtract(score, min_value), tf.to_float(tf.subtract(max_value, min_value)))
            return result

        batch_index_o = tf.constant(0)
        standard_batch_domain_score_o = tf.convert_to_tensor([])

        def cond(batch_index, normalized_score):
            return tf.less(batch_index, batch_size)

        def body(batch_index, normalized_score):
            min_value = tf.convert_to_tensor([-1, 2, 1, 0, 0, 0, 0, 0, 0, 0], tf.float32)[batch_article_set[batch_index]]
            max_value = tf.convert_to_tensor([-1, 12, 6, 3, 3, 4, 4, 30, 60, 9], tf.float32)[
                batch_article_set[batch_index]]
            temp_score = tf.cond(tf.greater(100000, batch_doc_id[batch_index]),
                                 lambda: batch_domain1_score[batch_index],
                                 lambda: min_value)
            temp_score = normalize_value(tf.to_float(temp_score), min_value, max_value)
            normalized_score = tf.concat([normalized_score, [temp_score]], axis=0)
            return tf.add(batch_index, 1), normalized_score

        _, standard_batch_domain_score = tf.while_loop(cond,
                                                       body,
                                                       [batch_index_o,
                                                        standard_batch_domain_score_o],
                                                       shape_invariants=[batch_index_o.get_shape(),
                                                                         tf.TensorShape([None])])
        fw_cell = create_rnn_cell(self.lstm_hidden_size,
                                  self.dropout_prob,
                                  self.lstm_layers_num,
                                  self.bidirectional,
                                  is_training)
        output, states = tf.nn.dynamic_rnn(cell=fw_cell,
                                           inputs=batch_doc_encodes,
                                           sequence_length=batch_doc_sent_nums,
                                           dtype=tf.float32)
        last_state = states[0].h

        for hs in self.fnn_hidden_size:
            if is_training:
                last_state = tf.nn.dropout(last_state, keep_prob=1 - self.dropout_prob)
                last_state = tf.layers.dense(last_state,
                                             hs,
                                             activation=tf.nn.relu,
                                             kernel_initializer=create_initializer())
        if self.fnn_hidden_size:
            x_dim = self.fnn_hidden_size[-1]
        else:
            x_dim = self.lstm_hidden_size

        w = tf.get_variable(shape=[x_dim, 1],
                            initializer=create_initializer(),
                            name="weight",
                            dtype=tf.float32)
        b = tf.get_variable(initializer=tf.zeros_initializer(),
                            shape=[1],
                            name="bias",
                            dtype=tf.float32)
        if is_training:
            last_state = tf.nn.dropout(last_state, keep_prob=1 - self.dropout_prob)
        logit = tf.squeeze(tf.sigmoid(tf.matmul(last_state, w) + b))
        loss = tf.losses.mean_squared_error(labels=standard_batch_domain_score, predictions=logit)

        return loss, logit

    def model_fn_builder(self, learning_rate, num_train_step, num_warmup_steps):
        """

        Args:
            learning_rate: 学习速率
            num_train_step: 学习步数
            num_warmup_steps: 预热的步数

        Returns: 函数model_fn的函数句柄

        """

        def model_fn(features, labels, mode, params):
            batch_doc_encodes = tf.identity(features["doc_encodes"]) # shape = [batch_size, None, bert_dim]
            batch_article_set = tf.cast(tf.identity(features["article_set"]), tf.int32)  # shape = [batch_size]
            batch_doc_sent_nums = tf.cast(tf.identity(features["doc_sent_num"]), tf.int32)  # shape = [batch_size]
            batch_domain1_score = tf.identity(features["domain1_score"])  # shape = [batch_size]
            batch_doc_id = tf.cast(tf.identity(features["article_id"]), tf.int32)  # shape = [batch_size]
            batch_size = tf.shape(batch_doc_sent_nums)[0]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            loss, logit = self.build_graph(batch_doc_encodes,
                                           batch_doc_sent_nums,
                                           batch_article_set,
                                           batch_domain1_score,
                                           batch_doc_id,
                                           batch_size,
                                           is_training)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(loss=loss,
                                                         init_lr=learning_rate,
                                                         num_train_steps=num_train_step,
                                                         num_warmup_steps=num_warmup_steps,
                                                         use_tpu=False)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op
                )
            elif mode == tf.estimator.ModeKeys.EVAL:
                # predictions = logit
                # accuracy = tf.metrics.accuracy(fnn_labels, predictions)
                # eval_metrics = {
                #    "eval_accuracy": accuracy
                # }
                # tf.summary.scalar("eval_accuracy", accuracy)
                # output_spec = tf.estimator.EstimatorSpec(
                #    mode=mode,
                #    loss=loss,
                #    eval_metric_ops=eval_metrics
                # )
                pass
            else:
                predictions = {
                    "batch_scores": logit,
                    "batch_doc_id": batch_doc_id,
                    "batch_article_set": batch_article_set
                }
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={
                        "predictions": tf.estimator.export.PredictOutput(predictions)
                    }
                )

            return output_spec

        return model_fn

    @staticmethod
    def eval_metric(result, handmark_scores, articles_set, articles_id):
        """

        Args:
            result:
            handmark_scores:
            articles_set:
            articles_id:

        Returns:

        """
        predict_scores = {}
        for item in result:
            min_value = 0
            max_value = 10
            normalize_score = item["batch_scores"]
            overall_score = round(normalize_score * (max_value - min_value) + min_value)
            predict_scores[item["batch_doc_id"]] = overall_score

        for key, value in predict_scores.items():
            temp_key = key
            if key > 100000:
                temp_key -= 100000
            article_set_id = articles_set[articles_id.index(temp_key)]
            min_value = dataset_score_range[article_set_id][0]
            max_value = dataset_score_range[article_set_id][1]
            if key > 100000:
                hs = min_value
            else:
                hs = handmark_scores[key]
            temp_hs = round(((hs - min_value) / (max_value - min_value)) * 10)
            print("id:{}, predict:{}, handmark:{}".format(key, value, temp_hs))


class PromptRelevantScore:
    """ essay的coherence方面的评分，也称连贯性得分，后续的overall score是在其基础上进一步得到的

    Attributes:
        bert_emb_dim: 因为输入的句子是用bert进行encoded， 该属性记录了所用bert模型的输出维度
        dropout_prob: dropout的概率
        prompts_embedding: 用于训练的所有的题目的embedding, 也是使用bert进行encode
        lstm_layers_num: lstm的层数
        lstm_hidden_size: lstm的隐层单元的宽度
        fnn_hidden_size: 输出接的fnn的每个隐层的宽度，
        bidirectional: 标注是否使用双向模型

    """

    def __init__(self):
        self.bert_emb_dim = 768
        self.dropout_prob = 0.5

        self.lstm_hidden_size = 1024
        self.lstm_layers_num = 1
        self.fnn_hidden_size = []
        self.bidirectional = False

    def generate_asap_train_and_test_set(self, generate_type="shuffle_all"):
        """ 生成asap的训练数据集合

        Args:
            generate_type: 以何种方式产生训练集和测试集，
                           shuffle_prompt: 表示从某个prompt下，选取一定的数据训练，
                           other_prompt: 表示利用所有别的prompt下的样本对该样本进行训练，一般不推荐此种方式，效果最差
                           shuffle_all: 表示shuffle所有的样本，然后抽样，为缺省方式

        Returns:
            articles_id: 文章的id
            articles_set: 文章的所属的prompt的id集合
            handmark_scores: 手工标注的分数
            correspond_train_id_set: 分割的训练集id
            correspond_test_id_set: 分割的测试集id
        """
        articles_id, articles_set, set_ids, handmark_scores = read_asap_dataset()
        np.random.seed(train_conf["random_seed"])

        permutation_ids = np.random.permutation(set_ids[train_conf["prompt_id"]])
        correspond_train_id_set = permutation_ids[
                                  0:int(len(permutation_ids) * train_conf["train_set_prob"])]
        correspond_test_id_set = permutation_ids[
                                 int(len(permutation_ids) * train_conf["train_set_prob"]):]

        correspond_train_id_set = list(correspond_train_id_set)
        correspond_test_id_set = list(correspond_test_id_set)

        other_ids = []
        for i in range(1, 9):
            if i == train_conf["prompt_id"]:
                continue
            else:
                other_ids.extend(set_ids[i])

        self.negative_samples = random.sample(other_ids, len(set_ids[train_conf["prompt_id"]]))
        negative_samples_train = self.negative_samples[
                                 0:int(len(permutation_ids) * train_conf["train_set_prob"])]
        negative_samples_test = self.negative_samples[
                                int(len(permutation_ids) * train_conf["train_set_prob"]):]

        correspond_train_id_set.extend(negative_samples_train)
        correspond_test_id_set.extend(negative_samples_test)

        return articles_id, articles_set, handmark_scores, correspond_train_id_set, correspond_test_id_set

    def build_graph(self,
                    batch_doc_encodes,
                    batch_doc_sent_nums,
                    batch_article_set,
                    batch_domain1_score,
                    batch_size,
                    prompt_encodes,
                    is_training):
        """ 建立模型的图

        Args:
            batch_doc_encodes: 一个batch的文章的bert encoding的结果
            batch_doc_sent_nums: 记录一个batch内每个doc的句数
            batch_article_set: 记录一个batch内每个doc所属的类别，按prompt分
            batch_domain1_score: 记录一个batch内每个doc的人工评分
            batch_size: 将batch的大小放在图中，
            prompt_encodes: 题目的标签
            is_training: 标致是否为train的状态

        Returns: loss 和 logits

        """

        def normalize_value(score, min_value, max_value):
            result = tf.div(tf.subtract(score, min_value), tf.to_float(tf.subtract(max_value, min_value)))
            return result

        prompt_encodes = prompt_encodes[0]  # 因为一个batch内的sample的prompt_encodes都是一样的，我们取地0个就好了
        p_shape = tf.shape(prompt_encodes)
        prompt_encodes = tf.convert_to_tensor(prompt_encodes)

        batch_index_o = tf.constant(0)
        standard_batch_domain_score_o = tf.convert_to_tensor([])
        batch_prompt_doc_encodes_o = tf.zeros([1, self.bert_emb_dim], dtype=tf.float32)

        def cond(batch_index, normalized_score, batch_prompt_doc_encodes):
            return tf.less(batch_index, batch_size)

        def body(batch_index, normalized_score, batch_prompt_doc_encodes):
            min_value = tf.convert_to_tensor([-1, 2, 1, 0, 0, 0, 0, 0, 0, 0], tf.float32)[batch_article_set[batch_index]]
            max_value = tf.convert_to_tensor([-1, 12, 6, 3, 3, 4, 4, 30, 60, 9], tf.float32)[
                batch_article_set[batch_index]]

            temp_score = tf.cond(tf.equal(train_conf["prompt_id"], batch_article_set[batch_index]),
                                 lambda: batch_domain1_score[batch_index],
                                 lambda: min_value)
            temp_score = normalize_value(tf.to_float(temp_score), min_value, max_value)
            normalized_score = tf.concat([normalized_score, [temp_score]], axis=0)

            temp_encodes = tf.concat([prompt_encodes, batch_doc_encodes[batch_index]], 0)
            batch_prompt_doc_encodes = tf.concat([batch_prompt_doc_encodes, temp_encodes], 0)

            return tf.add(batch_index, 1), normalized_score, batch_prompt_doc_encodes

        _, standard_batch_domain_score, batch_prompt_doc_encodes = tf.while_loop(cond,
                                                                                 body,
                                                                                 [batch_index_o,
                                                                                  standard_batch_domain_score_o,
                                                                                  batch_prompt_doc_encodes_o],
                                                                                 shape_invariants=[
                                                                                     batch_index_o.get_shape(),
                                                                                     tf.TensorShape([None]),
                                                                                     tf.TensorShape(
                                                                                         [None, self.bert_emb_dim])])

        batch_doc_sent_nums = tf.add(batch_doc_sent_nums, p_shape[0])
        batch_prompt_doc_encodes = tf.reshape(batch_prompt_doc_encodes[1:], [batch_size, -1, self.bert_emb_dim])

        fw_cell = create_rnn_cell(self.lstm_hidden_size,
                                  self.dropout_prob,
                                  self.lstm_layers_num,
                                  self.bidirectional,
                                  is_training)
        output, states = tf.nn.dynamic_rnn(cell=fw_cell,
                                           inputs=batch_prompt_doc_encodes,
                                           sequence_length=batch_doc_sent_nums,
                                           dtype=tf.float32)
        last_state = states[0].h

        for hs in self.fnn_hidden_size:
            if is_training:
                last_state = tf.nn.dropout(last_state, keep_prob=1 - self.dropout_prob)
                last_state = tf.layers.dense(last_state,
                                             hs,
                                             activation=tf.nn.relu,
                                             kernel_initializer=create_initializer()
                                             )
        if self.fnn_hidden_size:
            x_dim = self.fnn_hidden_size[-1]
        else:
            x_dim = self.lstm_hidden_size

        w = tf.get_variable(shape=[x_dim, 1],
                            initializer=create_initializer(),
                            name="weight",
                            dtype=tf.float32)
        b = tf.get_variable(initializer=tf.zeros_initializer(),
                            shape=[1],
                            name="bias",
                            dtype=tf.float32)
        if is_training:
            last_state = tf.nn.dropout(last_state, keep_prob=1 - self.dropout_prob)
        logit = tf.squeeze(tf.sigmoid(tf.matmul(last_state, w) + b))
        loss = tf.losses.mean_squared_error(labels=standard_batch_domain_score, predictions=logit)

        return loss, logit

    def model_fn_builder(self, learning_rate, num_train_step, num_warmup_steps):
        """

        Args:
           learning_rate: 学习速率
           num_train_step: 学习步数
           num_warmup_steps: 预热的步数

        Returns: 函数model_fn的函数句柄

        """

        def model_fn(features, labels, mode, params):
            batch_doc_encodes = tf.identity(features["doc_encodes"])  # shape = [batch_size, None, bert_dim]
            batch_article_set = tf.cast(tf.identity(features["article_set"]), tf.int32)  # shape = [batch_size]
            batch_doc_sent_nums = tf.cast(tf.identity(features["doc_sent_num"]), tf.int32)  # shape = [batch_size]
            batch_domain1_score = tf.identity(features["domain1_score"])  # shape = [batch_size]
            batch_doc_id = tf.cast(tf.identity(features["article_id"]), tf.int32) # shape = [batch_size]
            prompt_encodes = tf.identity(features["prompt_encodes"])  # shape = [batch_size, None, bert_dim]

            batch_size = tf.shape(batch_doc_sent_nums)[0]

            is_training = (mode == tf.estimator.ModeKeys.TRAIN)

            loss, logit = self.build_graph(batch_doc_encodes,
                                           batch_doc_sent_nums,
                                           batch_article_set,
                                           batch_domain1_score,
                                           batch_size,
                                           prompt_encodes,
                                           is_training)

            output_spec = None
            if mode == tf.estimator.ModeKeys.TRAIN:
                train_op = optimization.create_optimizer(loss=loss,
                                                         init_lr=learning_rate,
                                                         num_train_steps=num_train_step,
                                                         num_warmup_steps=num_warmup_steps,
                                                         use_tpu=False)
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    loss=loss,
                    train_op=train_op
                )
            elif mode == tf.estimator.ModeKeys.EVAL:
                # predictions = logit
                # accuracy = tf.metrics.accuracy(fnn_labels, predictions)
                # eval_metrics = {
                #    "eval_accuracy": accuracy
                # }
                # tf.summary.scalar("eval_accuracy", accuracy)
                # output_spec = tf.estimator.EstimatorSpec(
                #    mode=mode,
                #    loss=loss,
                #    eval_metric_ops=eval_metrics
                # )
                pass
            else:
                predictions = {
                    "batch_scores": logit,
                    "batch_doc_id": batch_doc_id,
                    "batch_article_set": batch_article_set
                }
                output_spec = tf.estimator.EstimatorSpec(
                    mode=mode,
                    predictions=predictions,
                    export_outputs={
                        "predictions": tf.estimator.export.PredictOutput(predictions)
                    }
                )

            return output_spec

        return model_fn

    def eval_metric(self, result, handmark_scores, articles_set, articles_id):
        """

        Args:
            result:
            handmark_scores:
            articles_set:
            articles_id:

        Returns:

        """
        predict_scores = {}
        for item in result:
            min_value = 0
            max_value = 10
            normalize_score = item["batch_scores"]
            overall_score = round(normalize_score * (max_value - min_value) + min_value)
            predict_scores[item["batch_doc_id"]] = overall_score

        for key, value in predict_scores.items():
            article_set_id = articles_set[articles_id.index(key)]
            min_value = dataset_score_range[article_set_id][0]
            max_value = dataset_score_range[article_set_id][1]
            if key in self.negative_samples:
                hs = min_value
            else:
                hs = handmark_scores[key]
            temp_hs = round(((hs - min_value) / (max_value - min_value)) * 10)
            print("id:{}, predict:{}, handmark:{}".format(key, value, temp_hs))


class OverallScorePredictor:
    """ 融合deep semantic的特征和handcrafted的特征，所得到的最终的分数

    Attributes:
        __bsp_estimator: basic score 模型的estimator对象
        __csp_estimator: coherence score 模型的estimator对象
        __psp_estimator: prompt relevant score 模型的estimator对象

    """

    def __init__(self,
                 bsp_estimator: tf.estimator.Estimator,
                 csp_estimator: tf.estimator.Estimator,
                 psp_estimator: tf.estimator.Estimator):
        self.__bsp_estimator = bsp_estimator
        self.__csp_estimator = csp_estimator
        self.__psp_estimator = psp_estimator

    def generate_asap_train_and_test_set(self,
                                         ps_generate_type="shuffle_all",
                                         ns_generate_type="no_negative"):
        """ 生成asap的训练数据集合

        Args:
            ns_generate_type: 以何种方式产生训练集和测试集的负样本，
                           no_negative: 表示没有负样本，此情况只适用于只考虑basic score 不考虑负样本的情况
                           prompt_irrelevant: 表示只产生与话题无关的负样本的id,
                           permuted: 表示只生产permuted essay 作为负样本，
                           both: 表示生产数量相等的prompt_irrelevant和permuted essay 作为负样本训练
            ps_generate_type: 以何种方式产生训练集和测试集中的正样本，
                           shuffle_prompt: 表示从某个prompt下，选取一定的数据训练，
                           shuffle_all: 表示shuffle所有的样本，然后抽样，为缺省方式

        Returns:
            articles_id: 文章的id
            articles_set: 文章的所属的prompt的id集合
            handmark_scores: 手工标注的分数
            correspond_train_id_set: 分割的训练集id
            correspond_test_id_set: 分割的测试集id
        """
        articles_id, articles_set, set_ids, handmark_scores = read_asap_dataset()
        np.random.seed(train_conf["random_seed"])
        self.__ns_from_other_prompt_train = []
        self.__ns_from_other_prompt_test = []
        self.__ns_from_permuted_train = []
        self.__ns_from_permuted_test = []

        if ps_generate_type == "shuffle_all":
            # 将所有set的数据混合打散，取80%进行训练，剩下测试
            permutation_ids = np.random.permutation(articles_id)
            correspond_test_id_set = permutation_ids[int(len(articles_id) * train_conf["train_set_prob"]):]
            correspond_train_id_set = permutation_ids[:int(len(articles_id) * train_conf["train_set_prob"])]
        elif ps_generate_type == "shuffle_prompt":
            permutation_ids = np.random.permutation(set_ids[train_conf["prompt_id"]])
            correspond_train_id_set = permutation_ids[
                                      0:int(len(permutation_ids) * train_conf["train_set_prob"])]
            correspond_test_id_set = permutation_ids[
                                     int(len(permutation_ids) * train_conf["train_set_prob"]):]
        else:
            raise ValueError("generate_type must be choose in ('shuffle_prompt', 'shuffle_all')")

        correspond_train_id_set = list(correspond_train_id_set)
        correspond_test_id_set = list(correspond_test_id_set)

        other_ids = []
        for i in range(1, 9):
            if i == train_conf["prompt_id"]:
                continue
            else:
                other_ids.extend(set_ids[i])
        self.__ns_from_permuted_train = [item + 100000 for item in correspond_train_id_set]
        self.__ns_from_permuted_test = [item + 100000 for item in correspond_test_id_set]

        negative_samples_from_other_prompt = random.sample(other_ids, len(set_ids[train_conf["prompt_id"]]))
        self.__ns_from_other_prompt_train = negative_samples_from_other_prompt[
                                            0:int(len(negative_samples_from_other_prompt) * train_conf["train_set_prob"])]
        self.__ns_from_other_prompt_test = negative_samples_from_other_prompt[
                                           int(len(negative_samples_from_other_prompt) * train_conf["train_set_prob"]):]
        if ns_generate_type == "permuted":
            # 在每个set内80%用来训练，20%用来测试
            correspond_train_id_set.extend(self.__ns_from_permuted_train)
            correspond_test_id_set.extend(self.__ns_from_permuted_test)
            articles_id.extend(self.__ns_from_permuted_train)
            articles_id.extend(self.__ns_from_permuted_test)
        elif ns_generate_type == "prompt_irrelevant":
            correspond_train_id_set.extend(self.__ns_from_other_prompt_train)
            correspond_test_id_set.extend(self.__ns_from_other_prompt_test)
        elif ns_generate_type == "both":
            correspond_train_id_set.extend(self.__ns_from_permuted_train)
            correspond_train_id_set.extend(self.__ns_from_other_prompt_train)
            correspond_test_id_set.extend(self.__ns_from_permuted_test[0:int(len(self.__ns_from_permuted_test) / 2)])
            correspond_test_id_set.extend(
                self.__ns_from_other_prompt_test[0:int(len(self.__ns_from_other_prompt_test) / 2)])
            articles_id.extend(self.__ns_from_permuted_train)
            articles_id.extend(self.__ns_from_permuted_test)
        elif ns_generate_type == "no_negative":
            pass
        else:
            raise ValueError("generate_type must be choose in ('shuffle_prompt','other_prompt','shuffle_all')")

        return articles_id, articles_set, handmark_scores, correspond_train_id_set, correspond_test_id_set

    def train(self,
              articles_id,
              correspond_train_id_set,
              correspond_test_id_set,
              tfrecord_file_path,
              xgboost_train_file_path,
              saved_model_dir):
        input_fn = input_fn_from_tfrecord(tfrecord_path=tfrecord_file_path,
                                          batch_size=train_conf["predict_batch_size"],
                                          is_training=False,
                                          element_ids=articles_id)
        bsp_result = self.__bsp_estimator.predict(input_fn)
        psp_result = self.__psp_estimator.predict(input_fn)
        csp_result = self.__csp_estimator.predict(input_fn)

        # normalized_scores
        basic_scores = {}
        for item in bsp_result:
            basic_scores[item["batch_doc_id"]] = item["batch_scores"]

        promp_scores = {}
        for item in psp_result:
            promp_scores[item["batch_doc_id"]] = item["batch_scores"]

        coher_scores = {}
        for item in csp_result:
            coher_scores[item["batch_doc_id"]] = item["batch_scores"]

        features = np.load(xgboost_train_file_path)["features"][()]

        train_features = []
        train_handmark_normalized_scores = []
        for i in correspond_train_id_set:
            if (i in features or (
                    i - 100000) in features) and i in basic_scores and i in promp_scores and i in coher_scores:
                temp_i = i
                if temp_i > 100000:
                    temp_i -= 100000
                temp_features = features[temp_i][:-2]
                temp_features.append(basic_scores[i])
                #temp_features.append(coher_scores[i])
                #temp_features.append(promp_scores[i])
                temp_features.append(1.0)
                temp_features.append(1.0)
                train_features.append(temp_features)
                if i in self.__ns_from_other_prompt_train or i in self.__ns_from_permuted_train:
                    train_handmark_normalized_scores.append(0)
                else:
                    train_handmark_normalized_scores.append(features[i][-1])

        test_features = []
        test_handmark_normalized_scores = []
        for i in correspond_test_id_set:
            if (i in features or (
                    i - 100000) in features) and i in basic_scores and i in promp_scores and i in coher_scores:
                temp_i = i
                if temp_i > 100000:
                    temp_i -= 100000
                temp_features = features[temp_i][:-2]
                temp_features.append(basic_scores[i])
                #temp_features.append(coher_scores[i])
                #temp_features.append(promp_scores[i])
                temp_features.append(1.0)
                temp_features.append(1.0)
                test_features.append(temp_features)
                if i in self.__ns_from_other_prompt_train or i in self.__ns_from_permuted_train:
                    test_handmark_normalized_scores.append(0)
                else:
                    test_handmark_normalized_scores.append(features[i][-1])

        xgb_rg = xgb.XGBRegressor(n_estimators=5000, learning_rate=0.001, max_depth=6, gamma=0.05,
                                  objective="reg:logistic")
        xgb_rg.fit(train_features,
                   train_handmark_normalized_scores,
                   eval_set=[(test_features, test_handmark_normalized_scores)],
                   early_stopping_rounds=100,
                   verbose=True)
        xgb_rg.save_model(os.path.join(saved_model_dir, "osp.xgboost"))

        pred_scores = xgb_rg.predict(test_features)
        test_predict_scores = []
        test_handmark_scores = [round(item * 10) for item in test_handmark_normalized_scores]
        for i in range(len(correspond_test_id_set)):
            min_value = 0
            max_value = 10
            overall_score = round(pred_scores[i] * (max_value - min_value) + min_value)
            test_predict_scores.append(overall_score)
            print("id:{}, basic:{}, coher:{}, prompt:{}, predict:{}, handmark:{}".format(correspond_test_id_set[i],
                                                                                         basic_scores[
                                                                                             correspond_test_id_set[
                                                                                                 i]] * 10,
                                                                                         coher_scores[
                                                                                             correspond_test_id_set[
                                                                                                 i]] * 10,
                                                                                         promp_scores[
                                                                                             correspond_test_id_set[
                                                                                                 i]] * 10,
                                                                                         overall_score,
                                                                                         test_handmark_scores[i]))

        test_handmark_scores = np.asarray(test_handmark_scores, dtype=np.int32)
        test_predict_scores = np.asarray(test_predict_scores, dtype=np.int32)

        qwk = quadratic_weighted_kappa(test_predict_scores, test_handmark_scores)
        print("##############qwk value is {}".format(qwk))

    def eval_metric(self, result, handmark_scores, articles_set, articles_id):
        pass


def train(estm, train_file_path, correspond_train_id_set, saved_model_dir):
    train_set_length = len(correspond_train_id_set)
    num_train_steps = int((train_set_length * train_conf["num_train_epochs"]) / train_conf["train_batch_size"])
    input_fn = input_fn_from_tfrecord(tfrecord_path=train_file_path,
                                      batch_size=train_conf["train_batch_size"],
                                      is_training=True,
                                      element_ids=correspond_train_id_set)
    estm.train(input_fn=input_fn, steps=num_train_steps)
    estm.export_saved_model(saved_model_dir, serving_input_receiver_fn())


def test(estm, test_file_path, correspond_test_id_set, handmark_scores, articles_set, articles_id, sp):
    input_fn = input_fn_from_tfrecord(tfrecord_path=test_file_path,
                                      batch_size=train_conf["predict_batch_size"],
                                      is_training=False,
                                      element_ids=correspond_test_id_set)
    predict_result = estm.predict(input_fn)
    sp.eval_metric(result=predict_result,
                   handmark_scores=handmark_scores,
                   articles_set=articles_set,
                   articles_id=articles_id)


def generate_tf_estimator(model_dir, sp, num_train_steps=-1, num_warmup_steps=-1):
    run_config = tf.estimator.RunConfig(
        model_dir=model_dir,
        save_checkpoints_steps=train_conf["save_checkpoints_step"],
        save_summary_steps=20
    )
    model_fn = sp.model_fn_builder(
        learning_rate=train_conf["learning_rate"],
        num_train_step=num_train_steps,
        num_warmup_steps=num_warmup_steps
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config
    )
    return estimator


def main():
    parser = argparse.ArgumentParser(description="四种score的训练以及公共数据集测试")
    parser.add_argument("-model", help="训练模型的类别")
    args = parser.parse_args()
    if args.model == "bsp":
        model_dir = sys_conf["bsp_output_dir"]
        sp = BasicScorePredictor()
    elif args.model == "csp":
        model_dir = sys_conf["csp_output_dir"]
        sp = CoherenceScore()
    elif args.model == "psp":
        model_dir = sys_conf["psp_output_dir"]
        sp = PromptRelevantScore()
    elif args.model == "osp":
        bsp = BasicScorePredictor()
        bsp_model_dir = sys_conf["bsp_output_dir"]
        bsp_estimator = generate_tf_estimator(bsp_model_dir, bsp)
        csp = CoherenceScore()
        csp_model_dir = sys_conf["csp_output_dir"]
        csp_estimator = generate_tf_estimator(csp_model_dir, csp)
        psp = PromptRelevantScore()
        psp_model_dir = sys_conf["psp_output_dir"]
        psp_estimator = generate_tf_estimator(psp_model_dir, psp)
        osp = OverallScorePredictor(bsp_estimator, csp_estimator, psp_estimator)
    else:
        raise ValueError("model need to be chosen from bsp, csp, psp and osp")

    if args.model == "osp":
        model_dir = sys_conf["osp_output_dir"]
        tfrecord_file_path = os.path.join(sys_conf["data_dir"], "asap_dataset.tfrecord")
        xgboost_train_file_path = os.path.join(sys_conf["data_dir"], "asap_xgboost.npz")
        if not (os.path.exists(tfrecord_file_path) or os.path.exists(xgboost_train_file_path)):
            raise ValueError("tfrecord file path or xgboost train file path is invalid.")
        articles_id, articles_set, handmark_scores, correspond_train_id_set, correspond_test_id_set = \
            osp.generate_asap_train_and_test_set()
        osp.train(articles_id, correspond_train_id_set, correspond_test_id_set, tfrecord_file_path, xgboost_train_file_path, model_dir)
    else:
        saved_model_dir = os.path.join(model_dir, "SavedModel")
        articles_id, articles_set, handmark_scores, correspond_train_id_set, correspond_test_id_set = sp.generate_asap_train_and_test_set()
        train_set_length = len(correspond_train_id_set)
        num_train_steps = int((train_set_length * train_conf["num_train_epochs"]) / train_conf["train_batch_size"])
        num_warmup_steps = int(num_train_steps * train_conf["warmup_proportion"])
        estimator = generate_tf_estimator(model_dir, sp, num_train_steps=num_train_steps, num_warmup_steps=num_warmup_steps)

        if train_conf["do_train"]:
            train_file_path = os.path.join(sys_conf["data_dir"], "asap_dataset.tfrecord")
            if not os.path.exists(train_file_path):
                raise ValueError("train_file_path is invalid.")
            train(estimator, train_file_path, correspond_train_id_set, saved_model_dir)

        if train_conf["do_predict"]:
            test_file_path = os.path.join(sys_conf["data_dir"], "asap_dataset.tfrecord")
            if not os.path.exists(train_file_path):
                raise ValueError("train_file_path is invalid.")
            test(estimator, test_file_path, correspond_test_id_set, handmark_scores, articles_set, articles_id, sp)


if __name__ == "__main__":
    main()
