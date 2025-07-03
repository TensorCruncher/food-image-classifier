# Introduction
We build an Image classifier for the Food101 dataset. We use the EffNetB2 architecture and finetune it on all 101 classes but only 20% of the data. This still takes about 30 minutes (!) using a T4 GPU on the colab free tier.

First, though, we explore AlexNet, EfficientNetB2 and ViT-B/16 architectures on 3 classes (Pizza, Samosa, Tacos) and 20% data of the Food101 dataset.

We choose to proceed with EffNetB2 for the 101 class dataset due to its small model size for comparable (slightyly better) performance to AlexNet, and 1/4 th the inference time to the ViT model (despite slightly less test accuracy).

Finally, we deploy a gradio demo with the main EfficientNetB2 model on [Hugging Face Spaces](https://huggingface.co/spaces/TensorCruncher/foodImageClassifier).

# Scope for improvement
We could do an analysis of the performance of the final model using a confusion matrix. This is important since the EffNetB2 test accuracy goes from high 80s to about 60% as we move from the 3 class dataset to the 101 class dataset.

This could imply that the EffNetB2 architecture cannot capture the complexities of the 101 classes well enough.

This can be seen in the demo, where if you choose the samosa sample image, it gets predicted as a spring roll.

Possible improvements can be:

* Training model on more images of classes it gets wrong
* Better data augmentation
* Possibly try different versions of EffNet, such as B3 - B7 (although that would take longer to train)
* Training for further epochs (>10) is not recommended for B2 since the loss appears to plateau by then.

# Project extensions
We can train another neural net to predict if an image is of food or not. If the image is of food, we can then pass it to our model to classify it as one of the Food101 categories.

This assumes that images of food supplied to the model at test time are from the Food101 list of foods.
