best_params = grid_search.best_params_
print("Best Parameters:", best_params)

# Train with the best hyperparameters
rf_best = RandomForestClassifier(
    n_estimators=best_params['n_estimators'],
    min_samples_split=best_params['min_samples_split'],
    max_features=best_params['max_features'],
    random_state=123,
    oob_score=True
)
rf_best.fit(train_data_wk.drop(columns='Y'), train_data_wk['Y'])

# Variable importance
importances = rf_best.feature_importances_
indices = np.argsort(importances)[::-1]
features = train_data_wk.drop(columns='Y').columns

# Print feature importances
print("Feature importances:")
for f in range(len(features)):
    print(f"{features[indices[f]]}: {importances[indices[f]]}")

# Plot the feature importances
plt.figure(figsize=(12, 6))
plt.title("Feature Importances")
plt.bar(range(len(features)), importances[indices], align='center')
plt.xticks(range(len(features)), [features[i] for i in indices], rotation=90)
plt.tight_layout()
plt.show()

# Test performance on test data
pred_test = rf_best.predict(test_data.drop(columns='Y'))
print("Test Accuracy:", accuracy_score(test_data['Y'], pred_test))
print("Confusion Matrix:\n", confusion_matrix(test_data['Y'], pred_test))
