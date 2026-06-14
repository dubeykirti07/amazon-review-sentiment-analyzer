import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

data = {
    'review': [
        'This product is amazing and works perfectly',
        'Very good quality and fast delivery',
        'Excellent item, I am very happy',
        'Totally satisfied with the product',
        'Bad quality and waste of money',
        'Very poor experience, not recommended',
        'The product is broken and useless',
        'Worst purchase ever',
        'Good value for money',
        'Really nice and useful product'
    ],
    'sentiment': [
        'positive', 'positive', 'positive', 'positive',
        'negative', 'negative', 'negative', 'negative',
        'positive', 'positive'
    ]
}

df = pd.DataFrame(data)

X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.3, random_state=42
)

model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression())
])

model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print(classification_report(y_test, pred))

def predict_review(text):
    result = model.predict([text])[0]
    return result

sample_review = "This product is very nice and useful"
print("Sample Review Sentiment:", predict_review(sample_review))