# OpenTransformer

This is a speech transformer model for end-to-end speech recognition.

## Function

-[x] Speech Transformer

-[x] Label Smoothing

-[x] Share weights of Embedding with output softmax layer

-[x] Data Augmentation([SpecAugument](https://arxiv.org/abs/1904.08779))

-[x] Extract Fbank features in a online funshion

-[x] Visualization based Tensorboard

-[x] Batch Beam Search with Length Penalty

-[x] Multiple Optimizers and Schedulers

-[x] Multiple Activation Functions in FFN

-[x] Multi GPU (Three Mode: [dp](https://pytorch.org/docs/stable/nn.html#dataparallel), [ddp](https://pytorch.org/docs/stable/nn.html#distributeddataparallel), [hvd](https://github.com/horovod/horovod))

[x] Mixed Precision Training based [apex](https://github.com/NVIDIA/apex)

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
Our Model can achieve a CER of 7.6% on [AISHELL-1](http://www.openslr.org/33/), which is close to 7.5% of Chain Model in Kaldi.

## Acknowledge
OpenTransformer refer to [ESPNET](https://github.com/espnet/espnet).
