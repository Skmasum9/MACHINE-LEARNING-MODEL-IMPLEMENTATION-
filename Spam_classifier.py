# spam_classifier.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Dataset
url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_table(url, header=None, names=['label', 'message'])

# 2. Encode labels
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# 3. Visualize data
sns.countplot(data=df, x='label')
plt.title("Distribution of Spam vs Ham")
plt.show()

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label_num'], test_size=0.2, random_state=42
)

# 5. Vectorize Text
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 6. Train Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# 7. Predict
y_pred = model.predict(X_test_vec)

# 8. Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))