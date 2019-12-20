# OpenTransformer

This is a speech transformer model for end-to-end speech recognition.

# Requirements
Pytorch: 1.2.0

Torchaudio: 0.3.0

## Function

- Speech Transformer

- Label Smoothing

- Share weights of Embedding with output softmax layer

- Data Augmentation([SpecAugument](https://arxiv.org/abs/1904.08779))

- Extract Fbank features in a online funshion

- Visualization based Tensorboard

- Batch Beam Search with Length Penalty

- Multiple Optimizers and Schedulers

- Multiple Activation Functions in FFN

- Multi GPU (Three Mode: [dp](https://pytorch.org/docs/stable/nn.html#dataparallel), [ddp](https://pytorch.org/docs/stable/nn.html#distributeddataparallel)

- Mixed Precision Training based [apex](https://github.com/NVIDIA/apex)

## Train
- Single GPU
```python
python rnn.py --config egs/aishell/conf/transformer.yaml
```
- Multi GPU Training based DataParallel
```python
python run.py --config egs/aishell/transformer.yaml --parallel_mode dp --ngpu 2
```
- Multi GPU Training based distributeddataparallel
```python
OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=2 run.py --config egs/aishell/transformer.yaml --parallel_mode ddp --ngpu 2
```
- Mutil GPU Training based Hovorod
```python
horovodrun -np 4 -H localhost:4 python run.py --config egs/aishell/transformer.yaml --parallel_mode hvd --ngpu 2
```

## Eval
```python
python eval.py --load_model model.pt
```

## Experiments
Our Model can achieve a CER of 7.1% without CMVN, any external LM and joint-CTC training on [AISHELL-1](http://www.openslr.org/33/), which is better than 7.5% of Chain Model in Kaldi.

## Acknowledge
OpenTransformer refer to [ESPNET](https://github.com/espnet/espnet).
