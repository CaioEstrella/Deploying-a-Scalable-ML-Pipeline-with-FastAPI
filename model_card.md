# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

This model is a `RandomForestClassifier` from scikit-learn, trained with `random_state=42` and default hyperparameters. It was developed as part of a Udacity ML DevOps Engineer course project. The model performs binary classification to predict whether an individual's annual income exceeds $50,000.

## Intended Use

The model is intended for educational purposes to demonstrate how to build, train, and deploy a scalable ML pipeline using FastAPI. It should not be used for actual financial or hiring decisions. The primary users are students and developers learning MLOps practices.

## Training Data

The model was trained on the [Census Income dataset](https://archive.ics.uci.edu/ml/datasets/census+income) (also known as the "Adult" dataset) from the UCI Machine Learning Repository. The dataset contains demographic information extracted from the 1994 U.S. Census database.

80% of the data (~26,000 samples) was used for training. Categorical features were encoded using `OneHotEncoder` and the target label was binarized using `LabelBinarizer`. No scaling was applied to continuous features.

The categorical features used are: `workclass`, `education`, `marital-status`, `occupation`, `relationship`, `race`, `sex`, and `native-country`.

## Evaluation Data

The remaining 20% of the data (~6,500 samples) was held out for evaluation, using the same preprocessing pipeline with encoders fitted on the training data only (to prevent data leakage).

## Metrics

The model was evaluated using precision, recall, and F1 score (fbeta with beta=1).

| Metric    | Value  |
|-----------|--------|
| Precision | 0.7419 |
| Recall    | 0.6384 |
| F1 Score  | 0.6863 |

Performance on individual data slices (per categorical feature value) is available in `slice_output.txt`.

## Ethical Considerations

The dataset includes sensitive demographic attributes such as `race`, `sex`, and `native-country`. Models trained on this data may reflect historical biases present in 1994 U.S. Census data. Performance varies across demographic slices, which means the model may be less accurate for certain groups. This model should not be used in any context where biased predictions could cause harm to individuals.

## Caveats and Recommendations

- The dataset is from 1994 and may not reflect current income distributions or labor market conditions.
- Continuous features are not scaled, which may affect model performance.
- The model uses default hyperparameters; hyperparameter tuning could improve performance.
- Slice performance should be reviewed before any deployment to identify disparate impact across demographic groups.
