# Gender-Classifier-Audio

- Pure python based implementation of [primaryobjects/voice-gender](https://github.com/primaryobjects/voice-gender)
audio classifier
- Classifiers are trained on a subset of the [Mozilla Common Voice](https://commonvoice.mozilla.org/en/datasets) corpus in a 5 fold cross-validation setup 
- Selected 5352 audio files each for male and female (with most positive feedback and almost no negative feedback) 
- Max accuracy achieved : 92% (MLP) 