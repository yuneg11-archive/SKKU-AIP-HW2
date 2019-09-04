# MNIST2Sequence with Tensorflow

## 0. MNIST Dataset Layout

  ### Train data: 60000 samples
  * Input shape: (28, 28, 1) -> 28x28 Image
  * Label shape: (5) -> Text label

  ### Test data: 10000 samples
  * Input shape: (28, 28, 1) -> 28x28 Image
  * Label shape: (5) -> Text label

## 1. Model Structure

* ### Encoder

  #### Input
  | Layer        | Output Shape    | Connected to |
  |--------------|-----------------|--------------|
  | input_1      | (?, 28, 28, 1)  |              |

  #### CNN Block 1
  | Layer        | Output Shape    | Connected to |
  |--------------|-----------------|--------------|
  | conv2d_1     | (?, 28, 28, 32) | input_1      |
  | activation_1 | (?, 28, 28, 32) | conv2d_1     |
  | max_pool_1   | (?, 14, 14, 32) | activation_1 |

  #### CNN Block 2
  | Layer        | Output Shape    | Connected to |
  |--------------|-----------------|--------------|
  | conv2d_2     | (?, 14, 14, 64) | max_pool_1   |
  | activation_2 | (?, 14, 14, 64) | conv2d_2     |
  | max_pool_2   | (?, 7, 7, 64)   | activation_2 |
  
  #### Encoder State Output
  | Layer        | Output Shape    | Connected to |
  |--------------|-----------------|--------------|
  | flatten_1    | (?, 128)        | max_pool_2   |
  | dense_1      | (?, 128)        | flatten_1    |
  | activation_3 | (?, 128)        | dense_1      |

* ### Decoder

  #### RNN Block
  | Layer       | Output Shape    | Connected to  |
  |-------------|-----------------|---------------|
  | basic_rnn_1 | (?, 5)          |               |
  
  #### Output
  | Layer       | Output Shape    | Connected to  |
  |-------------|-----------------|---------------|
  | dense_2     | (?, 5)          | basic_rnn_1   |

## 2. Model Train Result
  ![Train Plot](https://user-images.githubusercontent.com/2123763/64228564-2fd77d80-cf22-11e9-9076-72118eec94f7.png)
  * Average train cost: **0.00002,** (at 10 epoch)
  
  #### Sample Prediction with Test Data
  | Index | Label  | Output Sequence | | Index | Label  | Output Sequence |
  |-------|--------|-----------------|-|-------|--------|-----------------|
  | 1     | 2      | twoPP           | | 26    | 7      | seven           |
  | 2     | 1      | onePP           | | 27    | 4      | fourP           |
  | 3     | 0      | zeroP           | | 28    | 0      | zeroP           |
  | 4     | 4      | fourP           | | 29    | 1      | onePP           |
  | 5     | 1      | onePP           | | 30    | 3      | three           |
  | 6     | 4      | fourP           | | 31    | 1      | onePP           |
  | 7     | 9      | nineP           | | 32    | 3      | three           |
  | 8     | 5      | fiveP           | | 33    | 4      | fourP           |
  | 9     | 9      | nineP           | | 34    | 7      | seven           |
  | 10    | 0      | zeroP           | | 35    | 2      | twoPP           |
  | 11    | 6      | sixPP           | | 36    | 7      | seven           |
  | 12    | 9      | nineP           | | 37    | 1      | onePP           |
  | 13    | 0      | zeroP           | | 38    | 2      | twoPP           |
  | 14    | 1      | onePP           | | 39    | 1      | onePP           |
  | 15    | 5      | fiveP           | | 40    | 1      | onePP           |
  | 16    | 9      | nineP           | | 41    | 7      | seven           |
  | 17    | 7      | seven           | | 42    | 4      | fourP           |
  | 18    | 3      | three           | | 43    | 2      | twoPP           |
  | 19    | 4      | fourP           | | 44    | 3      | three           |
  | 20    | 9      | nineP           | | 45    | 5      | fiveP           |
  | 21    | 6      | sixPP           | | 46    | 1      | onePP           |
  | 22    | 6      | sixPP           | | 47    | 2      | twoPP           |
  | 23    | 5      | fiveP           | | 48    | 4      | fourP           |
  | 24    | 4      | fourP           | | 49    | 4      | fourP           |
  | 25    | 0      | zeroP           | | 50    | 6      | seven           |
