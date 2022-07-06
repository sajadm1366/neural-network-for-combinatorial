# neural-network-for-combinatorial

## Project description
In this project, we implement a deep neural network for combinatorial optimization problems.
The input is a set of items  and output consists of a sequence of items. Mathematically, the objective 
is to learn the joint distribution of output given input. An important property of  this problem is that the model 
is invariant to the inputs permutation which makes it suitable for set data. The output is assumed to be ordered important

The model consists of an encoder and a decoder. The encoder is implemented using [DeepSet]  model and 
the decoder is LSTM.

As an example, we trained the model to sort a set of integer numbers.

## Guide
To run the model simply use
```sh
python main.py 
```

The trained models parameters will be saved at a folder named "saved_model"
.




[DeepSet]: <https://proceedings.neurips.cc/paper/2017/hash/f22e4747da1aa27e363d86d40ff442fe-Abstract.html>
