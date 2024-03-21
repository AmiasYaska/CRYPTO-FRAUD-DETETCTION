from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

# Load the trained logistic regression model
lr_model = joblib.load("logistic_regression_model.pkl")

class DataPrep(BaseEstimator, TransformerMixin):
    """Implementation preprocess dataset in several strategies"""

    def __init__(self, num_feature_list: list, cat_feature_list: list, drop_feature_list=None,
                 cat_encoder_type='label', cat_min_count=10,
                 fillna=0, q_up_clip_outliers=None,
                 q_down_clip_outliers=None, build_feature=False):
        """
            `num_feature_list` - list with num features name
            `cat_feature_list` - list with cat features name
            `cat_encoder_type` - use `dummy` or `label` or both methods to encode features
            `drop_feature_list` - features to drop
            `cat_min_count` - min count to separate category from `other` category
            `fillna` - fill nans with 0, `mean` or `median` feature value
            `q_up_clip_outliers` - up quantile to clip outliers
            `q_down_clip_outliers` - down quantile to clip outliers
            `build_feature` - build new feature flag
        """
        self.cat_feature_list = cat_feature_list
        self.num_feature_list = num_feature_list
        self.cat_encoder_type = cat_encoder_type
        self.drop_feature_list = drop_feature_list
        self.cat_min_count = 50
        self.fillna = fillna
        self.q_up_clip_outliers = q_up_clip_outliers
        self.q_down_clip_outliers = q_down_clip_outliers
        self.build_feature = build_feature

    def fit(self, df):
        # Your fitting logic here
        return self

    def transform(self, df):
        # Your transformation logic here
        return df

# Initialize DataPrep object with sample data
num_feature_list = ['feature1', 'feature2', 'feature3']  # Sample numerical features
cat_feature_list = ['feature4', 'feature5']  # Sample categorical features
dp = DataPrep(
    num_feature_list=num_feature_list,
    cat_feature_list=cat_feature_list,
    cat_encoder_type=None,
    cat_min_count=10,
    fillna='median',
)

# Sample StandardScaler object initialization
scaler = StandardScaler()

# Sample LogisticRegression model initialization
lr_model = LogisticRegression(max_iter=1000, random_state=0)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the JSON data from the request
        data = request.get_json()

        # Convert the JSON data to a DataFrame
        df = pd.DataFrame(data)

        # Preprocess the data (assuming you have a DataPrep class defined)
        df_prep = dp.transform(df)

        # Scale the numerical features
        df_prep_scaled = scaler.transform(df_prep[num_feature_list])

        # Make predictions using the trained model
        predictions = lr_model.predict_proba(df_prep_scaled)[:, 1]

        # Return the predictions as JSON response
        return jsonify({'predictions': predictions.tolist()}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Use Gunicorn as the WSGI server in production
    app.run()
