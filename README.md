# DTOLR: Decision Tree One Label for outlier explanations

![DTOLR](https://github.com/rcrupiISP/DTOLR/assets/92302358/d4e0cb89-3efd-46d1-bb8b-e865a4395b03)


DTOLR (Decision Tree One Label Regressor) is a Python library for explaining outlier detection results using decision trees. It provides a method to interpret the decisions made by an outlier detection model, specifically tailored for anomaly detection tasks.

## Overview

DTOLR aims to improve the interpretability of outlier detection models, particularly in scenarios where understanding the reasons behind a data point being flagged as an outlier is crucial. It offers methods to extract decision rules from an ensemble of decision trees trained to distinguish outliers from normal data points.

## Key Features

- Explainability for outlier detection models
- Decision tree-based interpretation
- Support for both local and global explanations
- Customizable parameters for decision tree construction

## Usage
Here's a basic example demonstrating how to use DTOLR to explain outlier predictions:

```python 
from dtolr import DTOLR

# Initialize DTOLR with your training data, outlier predictions, and the outlier detection model
dtolr = DTOLR(X_train, y_pred_train, clf_ad)

# Explain instances
explanations = dtolr.explain_instances(X_expl, y_pred_expl)

# Print the explanations
print(explanations)
```

## Contributing
Contributions to DTOLR are welcome! If you have any suggestions, bug reports, or feature requests, please open an issue or submit a pull request on GitHub.
