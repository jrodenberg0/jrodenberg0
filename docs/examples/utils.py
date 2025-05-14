from lightgbmlss.model import *

class MyLGBMLSS(LightGBMLSS):
    def __init__(self,dist,params = None):
        super().__init__(dist)
        self.params = params if params is not None else {}

    def fit(self, X, y, **kwargs):
        """Fits the model using the provided data.

        Args:
            X: The training data.
            y: The target variable.
            **kwargs: Additional arguments to pass to the 'train' method.
        """
        train_set = lgb.Dataset(data=X, label=y)  # Create lgb.Dataset
        self.train(params = self.params,train_set=train_set, **kwargs)  # Call the original 'train' method

    def predict(self, X, **kwargs):
        """Predicts using the fitted model.

        Args:
            X: The input data for prediction.
            **kwargs: Additional arguments to pass to the 'predict' method.

        Returns:
            The predictions.
        """
        # Access the original 'predict' method using super() # take first column (rate or loc)
        predictions = super().predict(X, **kwargs).iloc[:,0].values.ravel()

        # Add custom logic here to modify predictions
        # Example: Convert raw predictions to probabilities for classification
        # predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid for probabilities

        return predictions  # Return the modified predictions


    def get_params(self, deep=True):
      """Gets parameters for this estimator.

      Args:
        deep (bool, optional): If True, will return the parameters for this estimator and
            contained subobjects that are estimators. Defaults to True.

      Returns:
        dict: Parameter names mapped to their values.
      """
      params = {'dist': self.dist, 'params': self.params}
      return params
