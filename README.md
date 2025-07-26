# Sangīn

The files in this repository provide for the training and evaluation of a
machine learning model that is meant to detect the meter of a hemistich of
classical Persian poetry. A recent evaluation run yielded the following results:

```
Accuracy: 0.9801
F1: 0.9792
Precision: 0.9789
Recall: 0.9801
Loss: 0.0996
```

## Base model

I chose [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base) from
Facebook AI. (Is there a better, more recently developed option? If so, I would
be happy to switch to it.)

## Training data

The data used here comes from [Ganjoor](https://ganjoor.net/). So far, I have
added to the dataset the following works, representing a total of 277,248 unique
hemistichs:

- The complete *ghazal*s of Ṣāʾib Tabrīzī
- The complete *ghazal*s of Ḥāfiẓ
- The complete *ghazal*s of Saʿdī
- All the *ghazal*s in the _Dīvān-i Shams_ of Rūmī (excepting some with obscure
  meters)
- The first _daftar_ of the _Maṡnavī_ of Rūmī
- A few thousand lines of the _Shāhnāma_ of Firdawsī
- Four of the poems in the _Khamsa_ of Niẓāmī: _Laylī u Majnūn_, _Khusraw va
  Shīrīn_, the _Haft paykar_, and the _Makhzan al-asrār_

More should still be added, but this is a start. The model is already quite good
at detecting any of the common meters. Some meters are rare enough that they
almost never appear in classical Persian poetry, let alone in this training set.

## Plans

- Get the model properly versioned and published
- Add to the training data and adjust training parameters to improve performance
- Present this work at a conference or workshop or similar (if anyone has
  suggestions...)
- Deploy an inference server and a web front end, i.e., a web app where a user
  could paste one or more hemistichs from a given poem and have the meter
  detected (see `consensus.py` for an idea of how this would work, looking for a
  consensus of high-confidence meter predictions)
