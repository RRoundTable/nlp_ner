
# -*- coding: utf-8 -*-

# tf.FLAGS 적용하기


import tensorflow as tf
import numpy as np

flags=tf.app.flags

FLAGS=flags.FLAGS



flags.DEFINE_float('score',0.0,'score of model')

FLAGS=flags.FLAGS

class Model:
    def __init__(self, parameter):
        self.parameter = parameter

    def build_model(self):
        self._build_placeholder()

        # { "morph": 0, "morph_tag": 1, "tag" : 2, "character": 3, .. }
        self._embedding_matrix = []
        for item in self.parameter["embedding"]:
            # item print 결과
            # item : ['word', 331273, 40] # unique한 갯수. dimension
            # item : ['character', 2176, 20] # unique한 갯수, dimension
            print("##############################")
            print(item)
            self._embedding_matrix.append(self._build_embedding(item[1], item[2], name="embedding_" + item[0]))
            # embedding_matrix = [[word embedding weight],[character embedding weight]] 이런 식으로 구성

        # 각각의 임베딩 값을 가져온다
        # tf.nn.embedding_lookup(
        #     params, # embedding tensor
        #     ids,    #  index number
        #     partition_strategy='mod',
        #     name=None,
        #     validate_indices=True,
        #     max_norm=None
        # )
        self._embeddings = []
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[0], self.morph))  # self.morph : word index
        self._embeddings.append(tf.nn.embedding_lookup(self._embedding_matrix[1], self.character)) # self.character: character

        # 음절을 이용한 임베딩 값을 구한다.
        character_embedding = tf.reshape(self._embeddings[1], [-1, self.parameter["word_length"], self.parameter["embedding"][1][2]])
        # character_embedding shape = (?,8,20)
        char_len = tf.reshape(self.character_len, [-1]) # shape (?,8,20) word_length(8) char_embedding_size(20)
        print("-------------character_embedding-------------")
        self.print_tensor(character_embedding)
        print("-------------char len-------------")
        self.print_tensor(char_len)

        character_emb_rnn, _, _ = self._build_birnn_model(character_embedding, char_len, self.parameter["char_lstm_units"], self.dropout_rate, last=True, scope="char_layer")
        # self.print_tensor(character_emb_rnn) # shape (?, 180, 80)
        # 위에서 구한 모든 임베딩 값을 concat 한다.
        all_data_emb = self.ne_dict  # ne_dict : NER 품사사전 size 15 [row,col,depth]
        # self.print_tensor(self.ne_dict) # shape (?,?,15)
        for i in range(0, len(self._embeddings) - 1):  # self._embeddings[0] ,self._embeddings[1]
            all_data_emb = tf.concat([all_data_emb, self._embeddings[i]], axis=2)  # 15+40 =55
        all_data_emb = tf.concat([all_data_emb, character_emb_rnn], axis=2)  # shape (30,180,135) sentence_length(180) 55+80=135

        # 모든 데이터를 가져와서 Bi-RNN 실시
        sentence_output, W, B = self._build_birnn_model(all_data_emb, self.sequence, self.parameter["lstm_units"], self.dropout_rate, scope="all_data_layer")
        # sentence_output shape =(?,48) // 48=2 * lstm_units
        # W shape =(48,30)
        # B shape =(30,)
        self._build_attention_layer(sentence_output)
        #weights = tf.get_variable(name='weights', shape=(sentence_output.get_shape()[-1], 15),initializer=tf.contrib.layers.xavier_initializer())
        #self._score = tf.map_fn(lambda elm: tf.matmul(elm, weights), sentence_output)
          # print("------------sentence output--------------")
        # self.print_tensor(sentence_output)
        # print("------------W--------------")
        # self.print_tensor(W)
        # print("------------B--------------")
        # self.print_tensor(B)

        sentence_output = tf.matmul(sentence_output, W) + B # crf_cost
        # self._output = sentence_output  # logit
        # sentence output shape=(?,30) batch size=30
        print("------------sentence output--------------")
        self.print_tensor(sentence_output)

        # 마지막으로 CRF 를 실시 한다
        crf_cost, crf_weight, crf_bias = self._build_crf_layer(sentence_output)
        # crf_cost : ()
        # crf_weigh : (30,30)
        # crf_bias : (30,)

        # seq2seq loss

        # seq_cost=self._build_seq2seq_layer(self.label) # target? self.Y 라벨링된 데이터

        # print("-------------crf_cost-------------")
        # self.print_tensor(crf_cost)
        # print("-------------crf_weight-------------")
        # self.print_tensor(crf_weight)
        # print("-------------crf_bias-------------")
        # self.print_tensor(crf_bias)
        self.train_op = self._build_output_layer(crf_cost)  # crf_cost
        # self.train_op = self._build_output_layer(seq_cost)
        self.cost = crf_cost # crf cost
        # self.cost=seq_cost


    # print 시도해보기
    def _build_placeholder(self):
        # morph, ne_dict, character, seq_len, char_len, label, step in dataset.get_data_batch_size(parameter["batch_size"], train)
        self.morph = tf.placeholder(tf.int32, [None, None])
        self.ne_dict = tf.placeholder(tf.float32, [None, None, int(self.parameter["n_class"] / 2)])
        self.character = tf.placeholder(tf.int32, [None, None, None])
        self.dropout_rate = tf.placeholder(tf.float32)
        self.sequence = tf.placeholder(tf.int32, [None])
        self.character_len = tf.placeholder(tf.int32, [None, None])
        self.label = tf.placeholder(tf.int32, [None, None])
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

    def _build_embedding(self, n_tokens, dimention, name="embedding"):
        embedding_weights = tf.get_variable(
            name, [n_tokens, dimention],
            dtype=tf.float32,
        )
        return embedding_weights

    def _build_single_cell(self, lstm_units, keep_prob):
        cell = tf.contrib.rnn.LayerNormBasicLSTMCell(lstm_units, activation=tf.nn.tanh)
        cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=keep_prob, output_keep_prob=keep_prob)

        return cell


    def _build_multi_cell(self, lstm_units, keep_prob, depth):
        self.depth=depth
        return tf.nn.rnn_cell.MultiRNNCell([self._build_single_cell(lstm_units, keep_prob) for i in range(depth)])

    def _build_stacked_cell(self,lstm_units,keep_prob,depth):
        rnn_cells=[]
        rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=lstm_units, activation=tf.nn.tanh)
        rnn_cell = tf.contrib.rnn.DropoutWrapper(cell=rnn_cell, output_keep_prob=keep_prob)
        rnn_cells.append(rnn_cell)
        return rnn_cells


    def _build_weight(self, shape, scope="weight"):
        with tf.variable_scope(scope):
            W = tf.get_variable(name="W", shape=[shape[0], shape[1]], dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name="b", shape=[shape[1]], dtype=tf.float32, initializer = tf.zeros_initializer())
        return W, b

    # rnn 에 대해서 자세히 살펴보기
    # 모델을 조금 더 복잡하게 만들고 싶다면,... multyRNNCell
    def _build_birnn_model(self, target, seq_len, lstm_units, keep_prob, last=False, scope="layer"):


        if last:
            with tf.variable_scope("forward_" + scope):
                lstm_fw_cell = self._build_single_cell(lstm_units, keep_prob) # multi cell로 바꾸기
                # lstm_fw_cell = self._build_multi_cell(lstm_units=lstm_units,keep_prob=keep_prob,depth=5)  # multi cell로 바꾸기
                # lstm_fw_cell = tf.contrib.rnn.OutputProjectionWrapper(lstm_fw_cell, output_size=lstm_units) # 24 -> 40
                print("--------------------lstm_fw_cell--------------------------------------")
                #self.print_tensor(lstm_fw_cell)
            with tf.variable_scope("backward_" + scope):
                lstm_bw_cell = self._build_single_cell(lstm_units, keep_prob) # multi cell로 바꾸기 -> rank가 5로 바뀐다(ouput state)
                # lstm_bw_cell = self._build_multi_cell(lstm_units=lstm_units, keep_prob=keep_prob, depth=5)
                # lstm_bw_cell = tf.contrib.rnn.OutputProjectionWrapper(lstm_bw_cell, output_size=lstm_units)
            with tf.variable_scope("birnn-lstm_" + scope):
                # return (outputs, output_states)
                _output = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, dtype=tf.float32,
                                                            inputs = target, sequence_length = seq_len, scope="rnn_" + scope)

                _, ((_, output_fw), (_, output_bw)) = _output
                outputs = tf.concat([output_fw, output_bw], axis=1)
                outputs = tf.reshape(outputs, shape=[-1, self.parameter["sentence_length"], 2 * lstm_units])


        else :
            with tf.variable_scope("forward_" + scope):
                # lstm_fw_cell = self._build_single_cell(lstm_units, keep_prob)  # multi cell로 바꾸기
                # lstm_fw_cell = self._build_multi_cell(lstm_units=lstm_units,keep_prob=keep_prob,depth=2)  # multi cell로 바꾸기
                # lstm_fw_cell = tf.contrib.rnn.OutputProjectionWrapper(lstm_fw_cell, output_size=lstm_units) # 24 -> 40
                lstm_fw_cell=self._build_stacked_cell(lstm_units,keep_prob,depth=2)

                print("--------------------lstm_fw_cell--------------------------------------")
                # self.print_tensor(lstm_fw_cell)
            with tf.variable_scope("backward_" + scope):
                # lstm_bw_cell = self._build_single_cell(lstm_units,
                #                                        keep_prob)  # multi cell로 바꾸기 -> rank가 5로 바뀐다(ouput state)
                # lstm_bw_cell = self._build_multi_cell(lstm_units=lstm_units, keep_prob=keep_prob, depth=2)
                # lstm_bw_cell = tf.contrib.rnn.OutputProjectionWrapper(lstm_bw_cell, output_size=lstm_units)
                lstm_bw_cell = self._build_stacked_cell(lstm_units, keep_prob, depth=2)
            with tf.variable_scope("birnn-lstm_" + scope):
                # return (outputs, output_states)
                # _output = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell, dtype=tf.float32,
                #                                           inputs=target, sequence_length=seq_len, scope="rnn_" + scope)

                (_output,_,_)=tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw = lstm_fw_cell,
                                                                           cells_bw = lstm_bw_cell,
                                                                           inputs = target,
                                                                           sequence_length = seq_len,
                                                                           dtype = tf.float32,scope="rnn_"+scope)

                # (output_fw, output_bw) = _output
                # outputs = tf.concat([output_fw, output_bw], axis=2)
                outputs = tf.reshape(_output, shape=[-1, 2 * lstm_units]) # crf 할 때 필요 multi cell
                # outputs=_output
        #     print("----------------------------------------------------------------------------")
        #     self.print_tensor(_output[0]) # single cell : [shape=(2, ?, 8, 40) // shape=(2, ?, 180, 24)]
        #
        #     self.print_tensor(_output[1]) # single cell : [shape=(2, 2, ?, 40) //  shape=(2, 2, ?, 24)]
        #     # self.print_tensor(_output)
        # if last: #  character_emb_rnn, _, _ = self._build_birnn_model(character_embedding, char_len, self.parameter["char_lstm_units"], self.dropout_rate, last=True, scope="char_layer")
        #     _, ((_, output_fw), (_, output_bw)) = _output #  output state를 보는 부분
        #     # _, outputs_state = _output
        #     # output_fw,output_bw=outputs_state
        #     # _,output_fw=output_fw
        #     # _,output_bw=output_bw
        #     self.print_tensor(output_fw) # shape=(5, 2, ?, 40) # (2,?,40)
        #     self.print_tensor(output_bw)  # shape=(5, 2, ?, 40)
        #     print("#############################")
        #     outputs = tf.concat([output_fw, output_bw], axis=1)  # column는 유지 row은 증가
        #     outputs = tf.reshape(outputs, shape=[-1, self.parameter["sentence_length"], 2 * lstm_units])
        #     print("last 통과")
        #
        # else: #  sentence_output, W, B = self._build_birnn_model(all_data_emb, self.sequence, self.parameter["lstm_units"], self.dropout_rate, scope="all_data_layer")
        #     (output_fw, output_bw), _ = _output # outputs를 보는 부분
        #     outputs = tf.concat([output_fw, output_bw], axis=2) # depth 기준으로
        #     outputs = tf.reshape(outputs, shape=[-1, 2 * lstm_units])

        W, b = self._build_weight([2 * self.parameter["lstm_units"], self.parameter["n_class"]], scope="output" + scope)
        return outputs, W, b

    def _build_crf_layer(self, target):
        with tf.variable_scope("crf_layer"):
            W, B = self._build_weight([self.parameter["n_class"], self.parameter["n_class"]], scope="weight_bias")
            matricized_unary_scores = tf.matmul(target, W) + B
            matricized_unary_scores = tf.reshape(matricized_unary_scores, [-1, self.parameter["sentence_length"], self.parameter["n_class"]])

            self.matricized_unary_scores = matricized_unary_scores
            self.log_likelihood, self.transition_params = tf.contrib.crf.crf_log_likelihood(self.matricized_unary_scores, self.label, self.sequence)
            cost = tf.reduce_mean(-self.log_likelihood)

            # tf.contrib.crf.crf_decode(
            #     potentials, // A [batch_size, max_seq_len, num_tags] tensor of unary potentials.
            #     transition_params, // A [num_tags, num_tags] matrix of binary potentials.
            #     sequence_length // A [batch_size] vector of true sequence lengths.
            # )
            self.viterbi_sequence, viterbi_score = tf.contrib.crf.crf_decode(matricized_unary_scores, self.transition_params, self.sequence) # decode_tag, best_score
            print("****************************************")

        return cost, W, B

    def _build_attention_layer(self,target): # target.shape=(batch_size, lstm_units*2)
        with tf.variable_scope("attention_layer"):
            W, B = self._build_weight([target.shape[1], target.shape[1]], scope="weight_bias")
            attention=tf.matmul(target,W)
            print("##########attention1################")
            print(attention)
            attention=tf.matmul(attention,tf.transpose(target))
            print("##########attention2################")
            print(attention) # shape=(lstm_units*2,lstm_units*2)
            # attention * H

            # softmax

            attention=tf.nn.softmax(attention)
            print("##########attention3################")
            print(attention) # shape=(lstm_units*2,lstm_units*2)

            # M= A*H
            result=tf.matmul(attention,target) # (Matrix shape = batch_size ,lstm_unix*2)
            print("##########result################")
            print(result)











    # def _build_seq2seq_layer(self, target):
    #     with tf.variable_scope('seq2seq_loss'):
    #         masks = tf.sequence_mask(lengths=self.sequence, maxlen=self.parameter["sentence_length"], dtype=tf.float32) # self._X_length : , max_len : self.parameter["sentence_length"]
    #         self.seq2seq_loss = tf.contrib.seq2seq.sequence_loss(logits=self._output, targets=target,weights=masks)
    #
    #         #self.viterbi_sequence, viterbi_score = tf.contrib.seq2seq.dynamic_decode(matricized_unary_scores,self.transition_params, self.sequence)
    #
    #
    #
    #         return self.seq2seq_loss

    def _build_output_layer(self, cost):
        with tf.variable_scope("output_layer"):
            train_op = tf.train.AdamOptimizer(self.parameter["learning_rate"]).minimize(cost, global_step=self.global_step)
        return train_op
 
    def sparse_cross_entropy_loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets) # self.logits 어디서 온 것인지
        cost = tf.reduce_mean(loss)
        self.masked_losses = self.mask * loss
        self.masked_losses = tf.reshape(self.masked_losses, tf.shape(self.Y))
        cost = tf.reduce_mean(tf.reduce_sum(self.masked_losses, 1) / tf.to_float(self.S))
        return loss, cost

    def print_tensor(self,tensor):
        tensor=tf.Print(tensor,[tensor])
        print("-----------------tensor print -----------------")
        print(tensor)

if __name__ == "__main__":
    parameter = {"embedding" : {
                    "morph" : [ 10, 10 ],
                    "morph_tag" : [ 10, 10 ],
                    "tag" : [ 10, 10 ],
                    "ne_dict" : [ 10, 10 ],
                    "character" : [ 10, 10 ],
                    }, "lstm_units" : 32, "keep_prob" : 0.65,
                    "sequence_length": 300, "n_class" : 100, "batch_size": 128,
                    "learning_rate" : 0.002
                }
    model = Model(parameter)
    model.build_model()
