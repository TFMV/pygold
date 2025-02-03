# pygold

Advanced Time Series Prediction with Transformers & CNN-LSTM

## Overview

Pygold is a deep learning-based framework for time series forecasting, integrating state-of-the-art machine learning techniques. It leverages an ensemble approach combining Transformer models and CNN-LSTM architectures for highly accurate financial time series predictions.

![pygold](assets/pygold.png)

## Performance

## Features

- Advanced Feature Engineering: Incorporates log transformations, EMA, volatility, and wavelet transforms.
- Sliding Window Dataset: Efficient data preparation for sequential modeling.
- Custom Asymmetric Loss Function: Penalizes underestimation more than overestimation.
- Hybrid Model Architecture: Combines Transformer encoders with CNN-LSTM networks.
- Hyperparameter Optimization: Uses Optuna for tuning key parameters.
- Performance Metrics: Achieved a final test MAE of 0.1012.

### Best Hyperparameters

| Parameter                   | Value  |
|-----------------------------|--------|
| time_step                   | 23     |
| epochs                      | 74     |
| batch_size                  | 32     |
| head_size                   | 55     |
| num_heads                   | 4      |
| ff_dim                      | 87     |
| num_transformer_blocks      | 1      |
| mlp_units                   | 105    |
| transformer_dropout         | 0.1679 |
| transformer_mlp_dropout     | 0.2120 |
| filters                     | 52     |
| kernel_size                 | 2      |
| lstm_units                  | 25     |
| cnn_lstm_dropout            | 0.1074 |

### Model Architecture

```
Model: "functional_41"
Total params: 19,015 (74.28 KB)
Trainable params: 19,015 (74.28 KB)
Non-trainable params: 0 (0.00 B)
```

### Training Performance

```
Final Test MAE: 0.1012

Sample Predictions vs. Actuals:
Predicted: 0.66 | Actual: 0.62
Predicted: 0.65 | Actual: 0.65
Predicted: 0.66 | Actual: 0.76
```

## Author

[Thomas F McGeehan V](https://github.com/TFMV)

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
