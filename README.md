# Speaker-Invariant Domain-Adversarial Neural Networks

Source codes for our ICMI'20 paper: Speaker-Invariant Adversarial Domain Adaptation for Emotion Recognition

## Source Listing
### experiment_a and experiment_v
Source codes of detecting arousal and valence values.
Implemented models include baseline, domain-adversarial neural networks (DANN), and speaker-invariant DANN (SIDANN)

### preprocess_iemocap and preprocess_msp_improv
Feature extraction codes for the two datasets.
You need to download [ResNet-152 model](https://download.pytorch.org/models/resnet152-b121ed2d.pth) and [VGGish model](https://storage.googleapis.com/audioset/vggish_model.ckpt) to `/models` and `/AUDIOSET` for feature extraction.
