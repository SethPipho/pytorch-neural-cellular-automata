# pytorch-neural-cellular-automata

An pytorch implementation of [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca/). 

![demo](https://i.imgur.com/QHXVKgW.gif)

## Quickstart

### Installation

```
git clone https://github.com/SethPipho/pytorch-neural-cellular-automata.git 
cd pytorch-neural-cellular-automata
pip install -e .
```

### Demo
```
pytorch-neural-ca demo --model resources/demo-model.pk
```

### Training 
```
pytorch-neural-ca train --image resources/emoji.png --output ../training-output
```

### CLI Reference

```
pytorch-neural-ca train --help
Device: cpu
Usage: pytorch-neural-ca train [OPTIONS]

Options:
  --image TEXT             Path to target image
  --output TEXT            Directory to output model and training data
  --width INTEGER          Width to resize image (preserves aspect ratio)
                           (default: 64)

  --padding INTEGER        Amount to pad image edges (default: 12)
  --batch-size INTEGER     Batch size (default: 8)
  --epochs INTEGER         Number of training iterations (default: 8000)
  --lr FLOAT               learning rate (default: 1e-3)
  --step-range INTEGER...  min and max steps to grow ca (default: 64 96)
  --grad-clip-val FLOAT    max norm value for gradient clipping (default: .1)
  --sample-every INTEGER   Output pool samples and demo videos every n epochs
                           (default: 1000)

  --channels INTEGER       Number of hidden state channels in model (default:
                           12)

  --help                   Show this message and exit.
```

```
pytorch-neural-ca demo --help
Device: cpu
Usage: pytorch-neural-ca demo [OPTIONS]

Options:
  --model TEXT           path to model
  --size INTEGER         size of grid (default: 64)
  --window-size INTEGER  size of window (default: 256)
  --help                 Show this message and exit.

```

## References

> Mordvintsev, et al., "Growing Neural Cellular Automata", Distill, 2020.