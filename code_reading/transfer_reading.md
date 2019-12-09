# Transformer Code Reading 

### Paper: [Attention is all you need](https://arxiv.org/pdf/1706.03762.pdf)
### Main code files:
```
# generator.py
# fairseq/sequence_generator.py
# fairseq/models/transformer.py
# fairseq/modules/multihead_attention.py
```
## Code Reading 

### Encoder ###

- input: src_tockens [128, 10]
  batch_size = 128, 每个句子包括10个单词

- embedding part
  - data embedding
    [128, 10]  -> [128, 10, 1024] 字典长度为1024
  - position embedding
    [128, 10, 1024], 每个单词对应的1024维的position是一样的
  - embedding 后的输入
    x = data_embedding + position_embedding
    x. transpose = [10, 128, 1024]， 将128个句子中的第一个单词作为一个batch输入
  - 生成encoder_padding_mask
    [128, 10] 表示10个单词中有几个是有效单词

- multi-head attention part
  - linear操作，生成query, key, value
    q, k, v = linear( x.transpose )
    q, k, v: [10, 128, 1024]
  - 生成 multi-head
    将q, k, v split  [10, 128, 1024] -> [2048, 10, 64]
    最后一维(1024) 分成16组，每组64维，将16组提到第一维作为batch (2048=128x16)
  - 计算attn_weight
    attn_weight = bmm(q, k.transpose(1,2))
    attn_weight:  [2048,10,10,] -> [128, 16, 10, 10] -> attn_mask_fill 
    (embedding part生成的mask,将padding的单词部分置为-inf，后续生成softmax值为0)
    对attn_weight做softmax + dropout (归一化权重 attn_weight)
  - 计算attention
    attention = bmm(attn_weight, v)
    attention [2048, 10, 64] -> [10, 128, 1024] 相当于将multi-head cat在一起
  - 计算multi-head attn的output linear
    multi-head attn output = linear(attention)
  - multi_head attn 的残差add + layer_norm
    
- feed forward part
  - feed forward两个linear操作
  - feed forward 的残差add + layer_norm

- output: Encoder outs = [10, 640, 1024], 640=128 x beam_size


### Decoder ###

- input: src_tockens, 按time_step依次输入一个/两个/三个etc个单词
  每个单词对应扩充为beam_size维 [batch_size*beam_size]

- embedding part
  - data_embedding + position_embedding
    将prev_output_tokens [640, 1] embedding成 [640, 1, 1024]

- decoder part
  - Masked Multi-Head Attention， 当前decoder的单词与后续(之后time step)的单词无关

- sequence_generator part
  - 一直重复embedding + decoder直到遇到结束符或者句子长度超过最大长度为止
