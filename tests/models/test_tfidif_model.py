import unittest
from sklearn.linear_model import LogisticRegression
from scripts.models.tfidf_model import TfidfModel
from tests.models.basic_model_test import BasicModelTest


class TestTfidfModel(BasicModelTest):

    def test_one_model_on_train(self):
        model = TfidfModel(clf=LogisticRegression(max_iter=10),
                           config=self.config,
                           text_column_name='text',
                           max_df=1)
        model.fit(self.full_data)
        model.predict_proba(self.full_data['train'])


if __name__ == '__main__':
    unittest.main()
