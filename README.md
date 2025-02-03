# pygold

Advanced Time Series Prediction with Transformers & CNN-LSTM

## Overview

Pygold is a deep learning-based framework for time series forecasting, integrating state-of-the-art machine learning techniques. It leverages an ensemble approach combining Transformer models and CNN-LSTM architectures for highly accurate financial time series predictions.

![pygold](assets/pygold.webp)

## Features

- Advanced Feature Engineering: Incorporates log transformations, EMA, volatility, and wavelet transforms.
- Sliding Window Dataset: Efficient data preparation for sequential modeling.
- Custom Asymmetric Loss Function: Penalizes underestimation more than overestimation.
- Hybrid Model Architecture: Combines Transformer encoders with CNN-LSTM networks.
- Hyperparameter Optimization: Uses Optuna for tuning key parameters.
- Performance Metrics: Achieved a final test MAE of 0.1012.

## Author

[Thomas F McGeehan V](https://github.com/TFMV)

## License

This project is licensed under the MIT License. See the [LICENSE file](LICENSE) for details.
