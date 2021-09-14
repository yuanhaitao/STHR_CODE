# MOSTO OF IT TAKEN FROM https://github.com/kolloldas/torchnlp
# MINOR CHANGES
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn.init as I
import numpy as np
import math


class EncoderLayer(nn.Module):
    """
    Represents one Encoder layer of the Transformer Encoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(EncoderLayer, self).__init__()

        self.multi_head_attention = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                       hidden_size, num_heads, None, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding='both',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs):
        x = inputs

        # if torch.isnan(x).any():
        #     print(x)
        #     assert torch.isnan(x).any() == False
        # Layer Normalization

        x_norm = self.layer_norm_mha(x)
        # if torch.isnan(x_norm).any():
        #     print(x, x_norm)
        #     assert torch.isnan(x_norm).any() == False

        # Multi-head attention
        # print("x_norm.shape", x_norm.shape)
        y = self.multi_head_attention(x_norm, x_norm, x_norm)
        # assert torch.isnan(y).any() == False

        # Dropout and residual
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)
        # assert torch.isnan(x_norm).any() == False

        # batch_size = x_norm.shape[0]
        batch_size, lens,  dim = x_norm.shape
        # x_norm = x_norm.permute(
        #     0, 2, 3, 1, 4).contiguous().reshape(-1, lens, dim)
        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)
        # assert torch.isnan(y).any() == False
        # y = y.reshape(batch_size, w, h, lens, -1).permute(0, 3, 1, 2, -1)

        # Dropout and residual
        y = self.dropout(x + y)

        return y


class DecoderLayer(nn.Module):
    """
    Represents one Decoder layer of the Transformer Decoder
    Refer Fig. 1 in https://arxiv.org/pdf/1706.03762.pdf
    NOTE: The layer normalization step has been moved to the input as per latest version of T2T
    """

    def __init__(self, hidden_size, total_key_depth, total_value_depth, filter_size, num_heads,
                 bias_mask, layer_dropout=0.0, attention_dropout=0.0, relu_dropout=0.0,  kernel_size=3, wide=None, height=None):
        """
        Parameters:
            hidden_size: Hidden size
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            layer_dropout: Dropout for this layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(DecoderLayer, self).__init__()

        self.multi_head_attention_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                           hidden_size, num_heads, bias_mask, attention_dropout)

        self.multi_head_attention_enc_dec = MultiHeadAttention(hidden_size, total_key_depth, total_value_depth,
                                                               hidden_size, num_heads, None, attention_dropout)

        self.positionwise_feed_forward = PositionwiseFeedForward(hidden_size, filter_size, hidden_size,
                                                                 layer_config='ll', padding='left',
                                                                 dropout=relu_dropout)
        self.dropout = nn.Dropout(layer_dropout)
        self.layer_norm_mha_dec = LayerNorm(hidden_size)
        self.layer_norm_mha_enc = LayerNorm(hidden_size)
        self.layer_norm_ffn = LayerNorm(hidden_size)

    def forward(self, inputs, src_mask):
        """
        NOTE: Inputs is a tuple consisting of decoder inputs and encoder output
        """
        x, encoder_outputs = inputs

        # Layer Normalization before decoder self attention
        x_norm = self.layer_norm_mha_dec(x)

        # Masked Multi-head attention
        y = self.multi_head_attention_dec(x_norm, x_norm, x_norm, src_mask)

        # Dropout and residual after self-attention
        x = self.dropout(x + y)

        # Layer Normalization before encoder-decoder attention
        x_norm = self.layer_norm_mha_enc(x)

        # Multi-head encoder-decoder attention
        # print(x_norm.shape, encoder_outputs.shape)
        y = self.multi_head_attention_enc_dec(
            x_norm, encoder_outputs, encoder_outputs)

        # Dropout and residual after encoder-decoder attention
        x = self.dropout(x + y)

        # Layer Normalization
        x_norm = self.layer_norm_ffn(x)

        # Positionwise Feedforward
        y = self.positionwise_feed_forward(x_norm)

        # Dropout and residual after positionwise feed forward layer
        y = self.dropout(x + y)

        # Return encoder outputs as well to work with nn.Sequential
        return y, encoder_outputs


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention as per https://arxiv.org/pdf/1706.03762.pdf
    Refer Figure 2
    """

    def __init__(self, input_depth, total_key_depth, total_value_depth, output_depth,
                 num_heads, bias_mask=None, dropout=0.0, query_is_picture=False, value_is_picture=False, use_picture_4_key=False, kernel_size=3, wide=None, height=None):
        """
        Parameters:
            input_depth: Size of last dimension of input
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            num_heads: Number of attention heads
            bias_mask: Masking tensor to prevent connections to future elements
            dropout: Dropout probability (Should be non-zero only during training)
            query_is_picture: True means the query has the affiliated shape (wide & length), like a picture, the input depth corresponds to the channel of the picture, so we have to use CNN to encoder the picture into hidden representation
            value_is_picture: True means the key & value have the affiliated shape (wide & length), like a picture, the output depth corresponds to the channel of outputs, so we have to use CNN
        """
        super(MultiHeadAttention, self).__init__()
        # Checks borrowed from
        # https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
        if total_key_depth % num_heads != 0:
            raise ValueError("Key depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_key_depth, num_heads))
        if total_value_depth % num_heads != 0:
            raise ValueError("Value depth (%d) must be divisible by the number of "
                             "attention heads (%d)." % (total_value_depth, num_heads))

        self.num_heads = num_heads
        self.query_scale = (total_key_depth//num_heads)**-0.5
        self.bias_mask = bias_mask
        self.query_is_picture = query_is_picture
        self.value_is_picture = value_is_picture
        self.use_picture_4_key = use_picture_4_key

        # Key and query depth will be same
        if self.query_is_picture is False:
            self.query_linear = nn.Linear(
                input_depth, total_key_depth, bias=False)
        else:
            self.query_cnn = nn.Conv2d(
                input_depth, total_key_depth, kernel_size, padding=kernel_size//2)
            if self.use_picture_4_key is False:
                self.query_linear = nn.Linear(
                    wide*height*total_key_depth, total_key_depth, bias=False)  # 得到每个时间步的权重

        if self.value_is_picture is False:
            self.key_linear = nn.Linear(
                input_depth, total_key_depth, bias=False)
            self.value_linear = nn.Linear(
                input_depth, total_value_depth, bias=False)

        else:
            self.key_cnn = nn.Conv2d(
                input_depth, total_key_depth, kernel_size, padding=kernel_size//2)
            if self.use_picture_4_key is False:
                self.key_linear = nn.Linear(
                    wide*height*total_key_depth, total_key_depth, bias=False)

            self.value_cnn = nn.Conv2d(
                input_depth, total_value_depth, kernel_size, padding=kernel_size//2)

        self.output_linear = nn.Linear(
            total_value_depth, output_depth, bias=False)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x, is_picture):
        """
        Split x such to add an extra num_heads dimension
        Input:
            x: a Tensor with shape [batch_size, seq_length, depth]  if is_picture is False, else [batch_size, seq_length, wide, height, depth]
        Returns:
            A Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads] if is_picure is False, else [batch_size, num_heads, seq_length, wide, height, depth]
        """
        if is_picture is False:
            if len(x.shape) != 3:
                raise ValueError("x must have rank 3")
            shape = x.shape
            return x.reshape(shape[0], shape[1], self.num_heads, shape[2]//self.num_heads).permute(0, 2, 1, 3).contiguous()
        else:
            if len(x.shape) != 5:
                raise ValueError("x must have rank 5")
            shape = x.shape
            return x.reshape(shape[0], shape[1], shape[2], shape[3], self.num_heads, shape[4]//self.num_heads).permute(0, 4, 1, 2, 3, 5).contiguous()

    def _merge_heads(self, x, is_picture):
        """
        Merge the extra num_heads into the last dimension
        Input:
            x: a Tensor with shape [batch_size, num_heads, seq_length, depth/num_heads] if is_picture is False, else [batch_size, num_heads, seq_length, wide, height, depth]
        Returns:
            A Tensor with shape [batch_size, seq_length, depth] if is_picture is False, else [batch_size, seq_length, wide, height, depth]
        """
        if is_picture is False:
            if len(x.shape) != 4:
                raise ValueError("x must have rank 4")
            shape = x.shape
            return x.permute(0, 2, 1, 3).contiguous().reshape(shape[0], shape[2], shape[3]*self.num_heads)
        else:
            if len(x.shape) != 6:
                raise ValueError("x must have rank 6")
            shape = x.shape
            return x.permute(0, 2, 3, 4, 1, 5).contiguous().reshape(shape[0], shape[2], shape[3], shape[4], shape[5]*self.num_heads)

    def forward(self, queries, keys, values, src_mask=None):
        # src_mask: [batch_size, k_seq_length]
        # Do a linear for each component
        # - query_is_picture:
        #       - use_picture_4_key: return [batch_size, q_seq_length, q_wide, q_height, output_depth]
        #       - other:
        #           - value_is_picture: return [batch_size, q_seq_length, k_wide, k_height, output_depth]
        #           - other: return [batch_size, q_seq_length, output_depth]
        # - other:
        #       - use_picture_4_key: return [batch_size, q_seq_length, output_depth]
        #       - other:
        #           - value_is_picture: return [batch_size, q_seq_length, k_wide, k_height, output_depth]
        #           - other: return [batch_size, q_seq_length, output_depth]

        # if bias_mask is not None:
        #     print(bias_mask.shape, bias_mask[:, :10, :10])
        # if src_mask is not None:
        #     print(src_mask.shape)

        if self.query_is_picture:
            batch_size, q_seq_length, q_wide, q_height, q_input_depth = queries.shape
            queries = self.query_cnn(
                queries.reshape(-1, q_wide, q_height, q_input_depth).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(batch_size, q_seq_length, q_wide, q_height, -1)

            if self.use_picture_4_key is False:
                queries = self.query_linear(
                    queries.reshape(batch_size, q_seq_length, -1))

        else:
            batch_size, q_seq_length, q_input_depth = queries.shape
            queries = self.query_linear(queries)

        if self.value_is_picture:
            batch_size, k_seq_length, k_wide, k_height, k_input_depth = keys.shape
            keys = self.key_cnn(
                keys.reshape(-1, k_wide, k_height, k_input_depth).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous().reshape(batch_size, k_seq_length, k_wide, k_height, -1)
            if self.use_picture_4_key is False:
                keys = self.key_linear(keys.reshape(
                    batch_size, k_seq_length, -1))

            values = self.value_cnn(values.reshape(-1, k_wide, k_height, k_input_depth).permute(0, 3, 1, 2).contiguous(
            )).permute(0, 2, 3, 1).contiguous().reshape(batch_size, k_seq_length, k_wide, k_height, -1)

        else:
            batch_size, k_seq_length, k_input_depth = keys.shape
            keys = self.key_linear(keys)
            values = self.value_linear(values)

        # Split into multiple heads
        queries = self._split_heads(
            queries, self.query_is_picture and self.use_picture_4_key)
        keys = self._split_heads(
            keys, self.value_is_picture and self.use_picture_4_key)
        values = self._split_heads(values, self.value_is_picture)

        # Scale queries
        queries = queries * self.query_scale

        # Combine queries and keys
        # merge query
        if self.use_picture_4_key:
            if self.query_is_picture:
                mean_queries = torch.mean(queries.reshape(
                    batch_size, self.num_heads, -1, q_wide*q_height, queries.shape[-1]), dim=-2)
                second_query_weights = nn.functional.softmax(torch.matmul(queries.reshape(
                    batch_size, self.num_heads, -1, q_wide*q_height, queries.shape[-1]), mean_queries.reshape(batch_size, self.num_heads, -1, queries.shape[-1], 1)).reshpae(batch_size, self.num_heads, -1, q_wide*q_height)/self.query_scale, dim=-1)
            if self.value_is_picture:
                mean_keys = torch.mean(keys.reshape(
                    batch_size, self.num_heads, -1, k_wide*k_height, keys.shape[-1]), dim=-2)

                second_key_weights = nn.functional.softmax(torch.matmul(keys.reshape(
                    batch_size, self.num_heads, -1, k_wide*k_height, keys.shape[-1]), mean_keys.reshape(batch_size, self.num_heads, -1, keys.shape[-1], 1)).reshape(batch_size, self.num_heads, -1, k_wide*k_height)*self.query_scale, dim=-1)

            if self.query_is_picture:
                if self.value_is_picture:
                    logits = torch.matmul(
                        mean_queries, mean_keys.permute(0, 1, 3, 2).contiguous())
                else:
                    logits = torch.matmul(
                        mean_queries, keys.permute(0, 1, 3, 2).contiguous())
            else:
                if self.value_is_picture:
                    logits = torch.matmul(
                        queries, mean_keys.permute(0, 1, 3, 2).contiguous())
                else:
                    logits = torch.matmul(
                        queries, keys.permute(0, 1, 3, 2).contiguous())
        else:
            logits = torch.matmul(
                queries, keys.permute(0, 1, 3, 2).contiguous())

        if src_mask is not None:
            logits = logits.masked_fill(src_mask.unsqueeze(
                1).unsqueeze(2) == 0, -np.inf)

        # Add bias to mask future values
        if self.bias_mask is not None:
            logits = logits + self.bias_mask[:, :, :logits.shape[-2],
                                             :logits.shape[-1]].type_as(logits.data)

        # Convert to probabilites
        weights = nn.functional.softmax(logits, dim=-1)

        # Dropout
        weights = self.dropout(weights)

        # 如果是需要local求权重
        if self.value_is_picture:
            if self.use_picture_4_key:
                context = torch.matmul(weights, torch.matmul(values.reshape(batch_size, self.num_heads, k_seq_length, k_wide*k_height, -1).permute(0, 1, 2, 4, 3),
                                                             second_key_weights.unsqueeze(4)).reshape(batch_size, self.num_heads, k_seq_length, -1))
                if self.query_is_picture:
                    context = context.unsqueeze(
                        3) * second_query_weights.unsqueeze(4).reshape(batch_size, self.num_heads, q_seq_length, q_wide, q_height, -1)
            else:
                context = torch.matmul(weights, values.reshape(
                    batch_size, self.num_heads, k_seq_length, -1)).reshape(
                        batch_size, self.num_heads, q_seq_length, k_wide, k_height, -1)

                if self.query_is_picture is False:
                    context = torch.mean(context.reshape(
                        batch_size, self.num_heads, q_seq_length, k_wide*k_height, -1), dim=-2)
        else:
            context = torch.matmul(weights, values.reshape(
                batch_size, self.num_heads, k_seq_length, -1))
            if self.use_picture_4_key and self.query_is_picture:
                context = context.unsqueeze(
                    3) * second_query_weights.unsqueeze(4).reshape(
                        batch_size, self.num_heads, q_seq_length, q_wide, q_height, -1)
            elif self.query_is_picture:
                context = context.unsqueeze(3).repeat(1, 1, 1, q_wide*q_height, 1).reshape(
                    batch_size, self.num_heads, q_seq_length, q_wide, q_height, -1)

        # Merge heads
        context = self._merge_heads(
            context, self.query_is_picture)
        #contexts = torch.tanh(contexts)

        # Linear to get output
        outputs = self.output_linear(context)

        # print("attention outputs shape is:{}".format(outputs.shape))
        return outputs


class Conv(nn.Module):
    """
    Convenience class that does padding and convolution for inputs in the format
    [batch_size, sequence length, hidden size]
    """

    def __init__(self, input_size, output_size, kernel_size, pad_type):
        """
        Parameters:
            input_size: Input feature size
            output_size: Output feature size
            kernel_size: Kernel width
            pad_type: left -> pad on the left side (to mask future data), 
                      both -> pad on both sides
        """
        super(Conv, self).__init__()
        padding = (
            kernel_size - 1, 0) if pad_type == 'left' else (kernel_size//2, (kernel_size - 1)//2)
        self.pad = nn.ConstantPad1d(padding, 0)
        self.conv = nn.Conv1d(input_size, output_size,
                              kernel_size=kernel_size, padding=0)

    def forward(self, inputs):
        inputs = self.pad(inputs.permute(0, 2, 1))
        outputs = self.conv(inputs).permute(0, 2, 1)

        return outputs


class PositionwiseFeedForward(nn.Module):
    """
    Does a Linear + RELU + Linear on each of the timesteps
    """

    def __init__(self, input_depth, filter_size, output_depth, layer_config='ll', padding='left', dropout=0.0):
        """
        Parameters:
            input_depth: Size of last dimension of input
            filter_size: Hidden size of the middle layer
            output_depth: Size last dimension of the final output
            layer_config: ll -> linear + ReLU + linear
                          cc -> conv + ReLU + conv etc.
            padding: left -> pad on the left side (to mask future data), 
                     both -> pad on both sides
            dropout: Dropout probability (Should be non-zero only during training)
        """
        super(PositionwiseFeedForward, self).__init__()

        layers = []
        sizes = ([(input_depth, filter_size)] +
                 [(filter_size, filter_size)]*(len(layer_config)-2) +
                 [(filter_size, output_depth)])

        for lc, s in zip(list(layer_config), sizes):
            if lc == 'l':
                layers.append(nn.Linear(*s))
            elif lc == 'c':
                layers.append(Conv(*s, kernel_size=3, pad_type=padding))
            else:
                raise ValueError("Unknown layer type {}".format(lc))

        self.layers = nn.ModuleList(layers)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs):
        x = inputs
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers):
                x = self.relu(x)
                x = self.dropout(x)
        return x


class LayerNorm(nn.Module):
    # Borrowed from jekbradbury
    # https://github.com/pytorch/pytorch/issues/1959
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


def _gen_bias_mask(max_length):
    """
    Generates bias values (-Inf) to mask future timesteps during attention
    """
    np_mask = np.triu(np.full([max_length, max_length], -np.inf), 1)
    torch_mask = torch.from_numpy(np_mask).type(torch.FloatTensor)

    return torch_mask.unsqueeze(0).unsqueeze(1)


def _gen_timing_signal(length, channels, min_timescale=1.0, max_timescale=1.0e4):
    """
    Generates a [1, length, channels] timing signal consisting of sinusoids
    Adapted from:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/layers/common_attention.py
    """
    position = np.arange(length)
    num_timescales = channels // 2
    log_timescale_increment = (math.log(
        float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * \
        np.exp(np.arange(num_timescales).astype(
            np.float) * -log_timescale_increment)
    scaled_time = np.expand_dims(position, 1) * \
        np.expand_dims(inv_timescales, 0)

    signal = np.concatenate([np.sin(scaled_time), np.cos(scaled_time)], axis=1)
    signal = np.pad(signal, [[0, 0], [0, channels % 2]],
                    'constant', constant_values=[0.0, 0.0])
    signal = signal.reshape([1, length, channels])

    return torch.from_numpy(signal).type(torch.FloatTensor)


def position_encoding(sentence_size, embedding_dim):
    encoding = np.ones((embedding_dim, sentence_size), dtype=np.float32)
    ls = sentence_size + 1
    le = embedding_dim + 1
    for i in range(1, le):
        for j in range(1, ls):
            encoding[i-1, j-1] = (i - (embedding_dim+1)/2) * \
                (j - (sentence_size+1)/2)
    encoding = 1 + 4 * encoding / embedding_dim / sentence_size
    # Make position encoding of time words identity to avoid modifying them
    # encoding[:, -1] = 1.0
    return np.transpose(encoding)


class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist)


class NoamOpt:
    "Optim wrapper that implements rate."

    def __init__(self, model_size, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
             min(step ** (-0.5), step * self.warmup ** (-1.5)))


# CONVERTED FROM https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/research/universal_transformer_util.py#L1062
class ACT_basic(nn.Module):
    def __init__(self, hidden_size):
        super(ACT_basic, self).__init__()
        self.sigma = nn.Sigmoid()
        self.p = nn.Linear(hidden_size, 1)
        self.p.bias.data.fill_(1)
        self.threshold = 1 - 0.1

    def forward(self, state, inputs, fn, time_enc, pos_enc, max_hop, encoder_output=None, src_mask=None):
        # init_hdd
        ## [B, S]
        # print(state.shape, inputs.shape, time_enc.shape,
        #       pos_enc.shape, max_hop)
        device = inputs.device
        halting_probability = torch.zeros(
            inputs.shape[0], inputs.shape[1]).to(device)
        # [B, S
        remainders = torch.zeros(inputs.shape[0], inputs.shape[1]).to(device)
        ## [B, S]
        n_updates = torch.zeros(inputs.shape[0], inputs.shape[1]).to(device)
        ## [B, S, HDD]
        previous_state = torch.zeros_like(state).to(device)
        step = 0
        # for l in range(self.num_layers):
        while(((halting_probability < self.threshold) & (n_updates < max_hop)).byte().any()):
            # Add timing signal
            if len(state.shape) == 3:
                state = state + time_enc[:,
                                         :inputs.shape[1], :].type_as(inputs.data)
                # print(pos_enc.shape)
                # if torch.isnan(state).any():
                #     print(state)
                #     assert torch.isnan(state).any() == False

                state = state + \
                    pos_enc[:, step, :].unsqueeze(1).repeat(
                        1, inputs.shape[1], 1).type_as(inputs.data)
                # if torch.isnan(state).any():
                #     print(state)
                #     assert torch.isnan(state).any() == False

            else:
                state = state + time_enc[:,
                                         :inputs.shape[1], :, :, :].type_as(inputs.data)
                state = state + \
                    pos_enc[:, step, :, :, :].unsqueeze(1).repeat(
                        1, inputs.shape[1], 1, 1, 1).type_as(inputs.data)

            p = self.sigma(
                self.p(state.reshape(inputs.shape[0], inputs.shape[1], -1))).squeeze(-1)

            # Mask for inputs which have not halted yet
            still_running = (halting_probability < 1.0).float()

            # Mask of inputs which halted at this step
            new_halted = (halting_probability + p * still_running >
                          self.threshold).float() * still_running

            # Mask of inputs which haven't halted, and didn't halt this step
            still_running = (halting_probability + p * still_running <=
                             self.threshold).float() * still_running

            # Add the halting probability for this step to the halting
            # probabilities for those input which haven't halted yet
            halting_probability = halting_probability + p * still_running

            # Compute remainders for the inputs which halted at this step
            remainders = remainders + new_halted * (1 - halting_probability)

            # Add the remainders to those inputs which halted at this step
            halting_probability = halting_probability + new_halted * remainders

            # Increment n_updates for all inputs which are still running
            n_updates = n_updates + still_running + new_halted

            # Compute the weight to be applied to the new state and output
            # 0 when the input has already halted
            # p when the input hasn't halted yet
            # the remainders when it halted this step
            update_weights = p * still_running + new_halted * remainders

            if encoder_output is not None:
                state, _ = fn((state, encoder_output), src_mask)
            else:
                # apply transformation on the state
                # print(state.shape)
                state = fn(state)

            # update running part in the weighted state and keep the rest
            if len(state.shape) == 3:
                previous_state = ((state * update_weights.unsqueeze(-1)) +
                                  (previous_state * (1 - update_weights.unsqueeze(-1))))

                # if torch.isnan(previous_state).any():
                #     print(previous_state)
                #     assert torch.isnan(previous_state).any() == False

            else:
                # print(previous_state.shape, state.shape)
                previous_state = ((state * update_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)) +
                                  (previous_state * (1 - update_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))))
            # previous_state is actually the new_state at end of hte loop
            # to save a line I assigned to previous_state so in the next
            # iteration is correct. Notice that indeed we return previous_state
            step += 1
        return previous_state, (remainders, n_updates)


class Encoder(nn.Module):
    """
    A Transformer Encoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, act=False, use_att=True):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
            use_mask: Set to True to turn on future value masking
        """

        super(Encoder, self).__init__()

        self.timing_signal = _gen_timing_signal(
            max_length, hidden_size)
        # for t
        self.position_signal = _gen_timing_signal(
            num_layers, hidden_size)

        self.num_layers = num_layers
        self.act = act
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  layer_dropout,
                  attention_dropout,
                  relu_dropout)

        self.proj_flag = False
        if(embedding_size != hidden_size):
            self.embedding_proj = nn.Linear(
                embedding_size, hidden_size, bias=False)
            self.proj_flag = True

        self.use_att = use_att
        if self.use_att:
            self.enc = EncoderLayer(*params)
        else:
            self.enc = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(
            input_dropout) if input_dropout > 0 else None
        if(self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs):
        # print("self.proj_flag", self.proj_flag)
        # Add input dropout
        x = self.input_dropout(
            inputs) if self.input_dropout != None else inputs

        if torch.isnan(x).any():
            print(x)
            assert torch.isnan(x).any() == False

        # print(x.shape, self.embedding_proj.weight.shape)
        if(self.proj_flag):
            # Project to hidden size
            x = self.embedding_proj(x)

        if torch.isnan(x).any():
            print(x)
            assert torch.isnan(x).any() == False

        if self.use_att:
            if(self.act):
                # print("x shape is ", x.shape, inputs.shape)
                x, (remainders, n_updates) = self.act_fn(x, inputs, self.enc,
                                                         self.timing_signal, self.position_signal, self.num_layers)
                return x, (remainders, n_updates)
            else:
                for l in range(self.num_layers):
                    x = x + self.timing_signal[:,
                                               :inputs.shape[1], :].type_as(inputs.data)
                    x = x + self.position_signal[:, l, :].unsqueeze(1).repeat(
                        1, inputs.shape[1], 1).type_as(inputs.data)
                    x = self.enc(x)
        else:
            x = x + self.timing_signal[:,
                                       :inputs.shape[1], :].type_as(inputs.data)
            x = x + self.position_signal[:, 0, :].unsqueeze(1).repeat(
                1, inputs.shape[1], 1).type_as(inputs.data)
            x = self.enc(x)
        return x, None


class Decoder(nn.Module):
    """
    A Transformer Decoder module. 
    Inputs should be in the shape [batch_size, length, hidden_size]
    Outputs will have the shape [batch_size, length, hidden_size]
    Refer Fig.1 in https://arxiv.org/pdf/1706.03762.pdf
    """

    def __init__(self, embedding_size, hidden_size, num_layers, num_heads, total_key_depth, total_value_depth,
                 filter_size, max_length=100, input_dropout=0.0, layer_dropout=0.0,
                 attention_dropout=0.0, relu_dropout=0.0, act=False, kernel_size=3, wide=None, height=None, use_mask=True, use_att=True):
        """
        Parameters:
            embedding_size: Size of embeddings
            hidden_size: Hidden size
            num_layers: Total layers in the Encoder
            num_heads: Number of attention heads
            total_key_depth: Size of last dimension of keys. Must be divisible by num_head
            total_value_depth: Size of last dimension of values. Must be divisible by num_head
            output_depth: Size last dimension of the final output
            filter_size: Hidden size of the middle layer in FFN
            max_length: Max sequence length (required for timing signal)
            input_dropout: Dropout just after embedding
            layer_dropout: Dropout for each layer
            attention_dropout: Dropout probability after attention (Should be non-zero only during training)
            relu_dropout: Dropout probability after relu in FFN (Should be non-zero only during training)
        """

        super(Decoder, self).__init__()

        self.timing_signal = _gen_timing_signal(max_length, hidden_size)
        self.position_signal = _gen_timing_signal(num_layers, hidden_size)
        self.num_layers = num_layers
        self.act = act
        params = (hidden_size,
                  total_key_depth or hidden_size,
                  total_value_depth or hidden_size,
                  filter_size,
                  num_heads,
                  _gen_bias_mask(
                      max_length) if use_mask else None,  # mandatory
                  layer_dropout,
                  attention_dropout,
                  relu_dropout,
                  kernel_size,
                  wide,
                  height)

        self.proj_flag = False
        if(embedding_size != hidden_size):  # 如果输出的embedding size大小不等于hidden size，就要做一个proj
            self.embedding_proj = nn.Linear(
                embedding_size, hidden_size, bias=False)
            self.proj_flag = True

        self.use_att = use_att
        if self.use_att:
            self.dec = DecoderLayer(*params)
        else:
            self.dec = nn.Sequential(
                nn.Linear(2*hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size)
            )

        self.layer_norm = LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(input_dropout)
        if(self.act):
            self.act_fn = ACT_basic(hidden_size)

    def forward(self, inputs, encoder_output, src_mask):
        # Add input dropout
        x = self.input_dropout(inputs)

        if(self.proj_flag):
            # Project to hidden size
            x = self.embedding_proj(x)

        if self.use_att:
            if(self.act):
                x, (remainders, n_updates) = self.act_fn(x, inputs, self.dec,
                                                         self.timing_signal, self.position_signal, self.num_layers, encoder_output, src_mask)
                return x, (remainders, n_updates)
            else:
                for l in range(self.num_layers):
                    x = x + self.timing_signal[:,
                                               :inputs.shape[1], :].type_as(inputs.data)
                    x = x + self.position_signal[:, l, :].unsqueeze(1).repeat(
                        1, inputs.shape[1], 1).type_as(inputs.data)
                    x, _ = self.dec((x, encoder_output), src_mask)
        else:
            encoder_output = torch.mean(encoder_output, dim=1).unsqueeze(
                1).repeat(1, inputs.shape[1], 1)
            x = x + self.timing_signal[:,
                                       :inputs.shape[1], :].type_as(inputs.data)
            x = x + self.position_signal[:, 0, :].unsqueeze(1).repeat(
                1, inputs.shape[1], 1).type_as(inputs.data)
            x = torch.cat([x, encoder_output], dim=-1)
        return x, None
