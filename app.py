from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained logistic regression model
lr_model = joblib.load("logistic_regression_model.pkl")

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

class DataPrep(BaseEstimator, TransformerMixin):
    """Implementation preprocess dataset in several strategies"""

    def __init__(self, num_feature_list: list, cat_feature_list: list, drop_feature_list: Optional[list] = None,
                 cat_encoder_type: Union[str, list] = 'label', cat_min_count: int = 10,
                 fillna: Union[int, str] = 0, q_up_clip_outliers: Optional[float] = None,
                 q_down_clip_outliers: Optional[float] = None, build_feature=False):
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
        self.num_fillna_dict = {}
        self.num_q_up_dict = {}
        self.num_q_down_dict = {}
        self.cat_emb_dict = {}

        # numerical fillna fit
        if self.fillna == 'median':
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = df[feature].median()
        elif self.fillna == 'mean':
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = df[feature].mean()
        elif self.fillna == 0:
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = 0
        else:
            for feature in self.num_feature_list:
                self.num_fillna_dict[feature] = None

        # numerical outliers fit
        if self.q_up_clip_outliers:
            for feature in self.num_feature_list:
                self.num_q_up_dict[feature] = df[feature].quantile(self.q_up_clip_outliers)

        if self.q_down_clip_outliers:
            for feature in self.num_feature_list:
                self.num_q_down_dict[feature] = df[feature].quantile(self.q_down_clip_outliers)

        # cat fit
        for feature in self.cat_feature_list:
            cat_series = df[feature].value_counts()
            cat_series[cat_series.lt(self.cat_min_count)] = 1
            self.cat_emb_dict[feature] = cat_series.to_dict()

        if self.drop_feature_list:
            self.num_feature_list = list(set(self.num_feature_list) - set(self.drop_feature_list))
            self.cat_feature_list = list(set(self.cat_feature_list) - set(self.drop_feature_list))

        return self

    def transform(self, df):
        check_is_fitted(self, attributes=['num_fillna_dict', 'cat_emb_dict'])

        # drop features
        if self.drop_feature_list:
            df = df.drop(columns=self.drop_feature_list)

        # numerical fillna
        for feature in self.num_feature_list:
            df.loc[df[feature].isna(), feature] = self.num_fillna_dict[feature]

        # numerical outliers
        if self.q_up_clip_outliers:
            for feature in self.num_feature_list:
                df.loc[df[feature] > self.num_q_up_dict[feature], feature] = self.num_q_up_dict[feature]

        if self.q_down_clip_outliers:
            for feature in self.num_feature_list:
                df.loc[df[feature] < self.num_q_down_dict[feature], feature] = self.num_q_down_dict[feature]

        # categorical embed
        df[self.cat_feature_list] = df[self.cat_feature_list].fillna('None')
        for feature in self.cat_feature_list:
            df[feature] = df[feature].map(self.cat_emb_dict[feature]).fillna(1)

        cat_encoder_type_list = self.cat_encoder_type if isinstance(self.cat_encoder_type, list) else [
            self.cat_encoder_type]

        if 'dummy' in cat_encoder_type_list:
            for feature in self.cat_feature_list:
                df_dummy = pd.get_dummies(df[feature], prefix=feature)
                df = df.merge(df_dummy, left_index=True, right_index=True)

        if 'label' not in cat_encoder_type_list:
            df = df.drop(columns=self.cat_feature_list)

        # feature engineering example
        if self.build_feature:
            df['total_Ether_ratio'] = df['total Ether sent'] / (df['total ether received'] + 1)
            df['total_Ether_ratio_v2'] = (df['total Ether sent'] - df['total ether received']) / (
                        df['total Ether sent'] + df['total ether received'] + 1)

            df['ERC20_uniq_addr_ratio'] = df[' ERC20 uniq sent addr'] / (df[' ERC20 uniq rec addr'] + 1)
            df['ERC20_uniq_addr_ratio_v2'] = (df[' ERC20 uniq sent addr'] - df[' ERC20 uniq rec addr']) / (
                        df[' ERC20 uniq sent addr'] + df[' ERC20 uniq rec addr'] + 1)

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
    app.run(debug=True)
