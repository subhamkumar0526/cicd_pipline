# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, precision_score, f1_score, recall_score

sns.set(style='white')


# Load Data
dataset = pd.read_csv('iris.csv')


# Clean column names
dataset.columns = [
    colname.strip(' (cm)').replace(" ", "_")
    for colname in dataset.columns.tolist()
]

features_names = dataset.columns.tolist()[:4]


# Feature Engineering
dataset['sepal_length_width_ratio'] = (
    dataset['sepal_length'] / dataset['sepal_width']
)

dataset['petal_length_width_ratio'] = (
    dataset['petal_length'] / dataset['petal_width']
)


# Select Features
dataset = dataset[
    [
        'sepal_length',
        'sepal_width',
        'petal_length',
        'petal_width',
        'sepal_length_width_ratio',
        'petal_length_width_ratio',
        'target'
    ]
]


# Split Data
train_data, test_data = train_test_split(
    dataset,
    test_size=0.2,
    random_state=44,
    stratify=dataset['target']
)


# X_train, y_train, X_test, y_test
X_train = train_data.drop('target', axis=1).values.astype('float32')
y_train = train_data['target'].values.astype('int32')

X_test = test_data.drop('target', axis=1).values.astype('float32')
y_test = test_data['target'].values.astype('int32')


# =========================================================
# Logistic Regression
# =========================================================

# multi_class removed because newer sklearn versions
# may not accept this parameter
logreg = LogisticRegression(
    C=0.0001,
    solver='lbfgs',
    max_iter=100
)

logreg.fit(X_train, y_train)

predictions_lr = logreg.predict(X_test)


# Logistic Regression Metrics
cm_lr = confusion_matrix(y_test, predictions_lr)

f1_lr = f1_score(
    y_test,
    predictions_lr,
    average='micro'
)

prec_lr = precision_score(
    y_test,
    predictions_lr,
    average='micro'
)

recall_lr = recall_score(
    y_test,
    predictions_lr,
    average='micro'
)


# Logistic Regression Accuracy
train_acc_lr = logreg.score(X_train, y_train) * 100
test_acc_lr = logreg.score(X_test, y_test) * 100


# =========================================================
# Random Forest Classifier
# =========================================================

rf_clf = RandomForestClassifier(
    n_estimators=100,
    random_state=44
)

rf_clf.fit(X_train, y_train)

predictions_rf = rf_clf.predict(X_test)


# Random Forest Metrics
f1_rf = f1_score(
    y_test,
    predictions_rf,
    average='micro'
)

prec_rf = precision_score(
    y_test,
    predictions_rf,
    average='micro'
)

recall_rf = recall_score(
    y_test,
    predictions_rf,
    average='micro'
)


# Random Forest Accuracy
train_acc_rf = rf_clf.score(X_train, y_train) * 100
test_acc_rf = rf_clf.score(X_test, y_test) * 100


# =========================================================
# Confusion Matrix Plotting Function
# =========================================================

def plot_cm(
    cm,
    target_name,
    title="Confusion Matrix",
    cmap=None,
    normalize=True
):

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(12, 8))

    if normalize:
        cm_display = cm.astype('float') / cm.sum(
            axis=1
        )[:, np.newaxis]
    else:
        cm_display = cm

    plt.imshow(
        cm_display,
        interpolation='nearest',
        cmap=cmap
    )

    plt.title(title)
    plt.colorbar()

    if target_name is not None:

        tick_marks = np.arange(len(target_name))

        plt.xticks(
            tick_marks,
            target_name,
            rotation=45
        )

        plt.yticks(
            tick_marks,
            target_name
        )

    thresh = (
        cm_display.max() / 1.5
        if normalize
        else cm_display.max() / 2
    )

    for i, j in itertools.product(
        range(cm_display.shape[0]),
        range(cm_display.shape[1])
    ):

        if normalize:

            plt.text(
                j,
                i,
                "{:0.4f}".format(cm_display[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if cm_display[i, j] > thresh
                    else "black"
                )
            )

        else:

            plt.text(
                j,
                i,
                "{:,}".format(cm_display[i, j]),
                horizontalalignment="center",
                color=(
                    "white"
                    if cm_display[i, j] > thresh
                    else "black"
                )
            )

    plt.tight_layout()

    plt.ylabel('True Label')

    plt.xlabel(
        'Predicted Label\n'
        'accuracy={:0.4f}; misclass={:0.4f}'.format(
            accuracy,
            misclass
        )
    )

    plt.savefig(
        'ConfusionMatrix.png',
        dpi=120,
        bbox_inches='tight'
    )

    plt.show()

    plt.close()


# Target Names
target_name = np.array([
    'setosa',
    'versicolor',
    'virginica'
])


# Plot Confusion Matrix
plot_cm(
    cm_lr,
    target_name,
    title="Confusion Matrix (Logistic Regression)",
    cmap=None,
    normalize=True
)


# =========================================================
# Random Forest Feature Importance
# =========================================================

importances = rf_clf.feature_importances_

labels = dataset.columns[:-1]

feature_df = pd.DataFrame(
    list(zip(labels, importances)),
    columns=['feature', 'importance']
)

features = feature_df.sort_values(
    by='importance',
    ascending=False
)


# Feature Importance Plot
plt.figure(figsize=(12, 6))

ax = sns.barplot(
    x='importance',
    y='feature',
    data=features
)

ax.set_xlabel(
    'Importance',
    fontsize=14
)

ax.set_ylabel(
    'Feature',
    fontsize=14
)

ax.set_title(
    'Random Forest Feature Importances',
    fontsize=12
)

plt.tight_layout()

plt.savefig(
    'FeatureImportance.png',
    dpi=120,
    bbox_inches='tight'
)

plt.show()

plt.close()


# =========================================================
# Saving Scores to File
# =========================================================

with open('scores.txt', 'w') as score:

    score.write(
        "Random Forest Train Accuracy: %2.1f%%\n"
        % train_acc_rf
    )

    score.write(
        "Random Forest Test Accuracy: %2.1f%%\n"
        % test_acc_rf
    )

    score.write(
        "F1 Score: %2.1f%%\n"
        % (f1_rf * 100)
    )

    score.write(
        "Recall Score: %2.1f%%\n"
        % (recall_rf * 100)
    )

    score.write(
        "Precision Score: %2.1f%%\n"
        % (prec_rf * 100)
    )

    score.write("\n\n")

    score.write(
        "Logistic Regression Train Accuracy: %2.1f%%\n"
        % train_acc_lr
    )

    score.write(
        "Logistic Regression Test Accuracy: %2.1f%%\n"
        % test_acc_lr
    )

    score.write(
        "F1 Score: %2.1f%%\n"
        % (f1_lr * 100)
    )

    score.write(
        "Recall Score: %2.1f%%\n"
        % (recall_lr * 100)
    )

    score.write(
        "Precision Score: %2.1f%%\n"
        % (prec_lr * 100)
    )

print("Training completed successfully!")
print("ConfusionMatrix.png created.")
print("FeatureImportance.png created.")
print("scores.txt created.")