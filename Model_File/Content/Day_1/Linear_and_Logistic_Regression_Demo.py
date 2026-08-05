
# ==========================================================
# Linear Regression & Logistic Regression - Complete Example
# ==========================================================
# This script demonstrates:
# 1. Creating a synthetic dataset
# 2. Viewing the input data
# 3. Splitting into training/testing sets
# 4. Training Linear Regression
# 5. Training Logistic Regression
# 6. Evaluating both models
# 7. Visualizing train/test data and model predictions
# 8. Predicting results for new students
# ==========================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
)

# -----------------------------
# Create Synthetic Dataset
# -----------------------------
np.random.seed(42)

study_hours = np.random.uniform(0, 10, 100).reshape(-1,1)
noise = np.random.normal(0,5,100)
true_slope = 8
true_intercept = 30
exam_scores = true_slope*study_hours.flatten()+true_intercept+noise
pass_fail = (exam_scores>=50).astype(int)

# -----------------------------
# Display Input Dataset
# -----------------------------
dataset = pd.DataFrame({
    "Study Hours":study_hours.flatten(),
    "Exam Score":exam_scores,
    "Pass/Fail":pass_fail
})

print("\nINPUT DATASET")
print("="*70)
print(dataset)

# -----------------------------
# Train/Test Split
# -----------------------------
X_train_lr,X_test_lr,y_train_lr,y_test_lr = train_test_split(
    study_hours,exam_scores,test_size=0.30,random_state=42
)

X_train_log,X_test_log,y_train_log,y_test_log = train_test_split(
    study_hours,pass_fail,test_size=0.30,
    random_state=42,stratify=pass_fail
)

print("\nLINEAR REGRESSION - TRAIN DATA")
print(pd.DataFrame({
    "Hours":X_train_lr.flatten(),
    "Score":y_train_lr
}))

print("\nLINEAR REGRESSION - TEST DATA")
print(pd.DataFrame({
    "Hours":X_test_lr.flatten(),
    "Score":y_test_lr
}))

print("\nLOGISTIC REGRESSION - TRAIN DATA")
print(pd.DataFrame({
    "Hours":X_train_log.flatten(),
    "Pass/Fail":y_train_log
}))

print("\nLOGISTIC REGRESSION - TEST DATA")
print(pd.DataFrame({
    "Hours":X_test_log.flatten(),
    "Pass/Fail":y_test_log
}))

# -----------------------------
# Linear Regression
# -----------------------------
linear_model = LinearRegression()
linear_model.fit(X_train_lr,y_train_lr)
y_pred_lr = linear_model.predict(X_test_lr)

print("\nLINEAR REGRESSION RESULTS")
print("="*70)
print(f"Equation : Score = {linear_model.coef_[0]:.2f} x Hours + {linear_model.intercept_:.2f}")
print(f"MSE      : {mean_squared_error(y_test_lr,y_pred_lr):.2f}")
print(f"R2 Score : {r2_score(y_test_lr,y_pred_lr):.4f}")

# -----------------------------
# Logistic Regression
# -----------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_log)
X_test_scaled = scaler.transform(X_test_log)

log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_scaled,y_train_log)

y_pred_log = log_model.predict(X_test_scaled)

print("\nLOGISTIC REGRESSION RESULTS")
print("="*70)
print(f"Accuracy : {accuracy_score(y_test_log,y_pred_log):.2%}")
print("\nConfusion Matrix")
print(confusion_matrix(y_test_log,y_pred_log))
print("\nClassification Report")
print(classification_report(y_test_log,y_pred_log))

# -----------------------------
# Visualization
# -----------------------------
fig,axs=plt.subplots(2,2,figsize=(14,10))

# Linear Train
axs[0,0].scatter(X_train_lr,y_train_lr,label="Training")
xs=np.sort(study_hours,axis=0)
axs[0,0].plot(xs,linear_model.predict(xs),linewidth=2,label="Regression")
axs[0,0].set_title("Linear Regression - Training")
axs[0,0].set_xlabel("Study Hours")
axs[0,0].set_ylabel("Exam Score")
axs[0,0].legend()
axs[0,0].grid(True)

# Linear Test
axs[0,1].scatter(X_test_lr,y_test_lr,label="Actual Test")
axs[0,1].scatter(X_test_lr,y_pred_lr,marker="x",s=80,label="Predicted")
axs[0,1].plot(xs,linear_model.predict(xs),linewidth=2)
axs[0,1].set_title("Linear Regression - Testing")
axs[0,1].set_xlabel("Study Hours")
axs[0,1].set_ylabel("Exam Score")
axs[0,1].legend()
axs[0,1].grid(True)

# Logistic Train
colors=["red" if i==0 else "green" for i in y_train_log]
axs[1,0].scatter(X_train_log,y_train_log,c=colors)
prob=log_model.predict_proba(scaler.transform(xs))[:,1]
axs[1,0].plot(xs,prob)
axs[1,0].axhline(.5,ls="--")
axs[1,0].set_title("Logistic Regression - Training")
axs[1,0].set_xlabel("Study Hours")
axs[1,0].set_ylabel("Probability")

# Logistic Test
colors=["red" if i==0 else "green" for i in y_test_log]
axs[1,1].scatter(X_test_log,y_test_log,c=colors,label="Actual")
prob_test=log_model.predict_proba(X_test_scaled)[:,1]
axs[1,1].scatter(X_test_log,prob_test,marker="x",s=80,label="Predicted Probability")
axs[1,1].plot(xs,prob)
axs[1,1].axhline(.5,ls="--")
axs[1,1].set_title("Logistic Regression - Testing")
axs[1,1].set_xlabel("Study Hours")
axs[1,1].set_ylabel("Probability")
axs[1,1].legend()

plt.tight_layout()
plt.show()

# -----------------------------
# Predict New Students
# -----------------------------
new=np.array([[2.5],[5],[7.5],[9]])
scores=linear_model.predict(new)
scaled=scaler.transform(new)
classes=log_model.predict(scaled)
probs=log_model.predict_proba(scaled)[:,1]

print("\nNEW STUDENT PREDICTIONS")
print("="*70)
for h,s,c,p in zip(new.flatten(),scores,classes,probs):
    print(f"Hours={h:4.1f}  Score={s:6.2f}  Pass={bool(c)}  Probability={p:.2%}")
