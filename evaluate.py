from sklearn.metrics import classification_report

def evaluate_model(pipeline, X_train, y_train, X_val, y_val, X_test, y_test):
    print("\n[INFO] Model eğitiliyor...")
    pipeline.fit(X_train, y_train)

    val_score = pipeline.score(X_val, y_val)
    print(f"[INFO] Validation Accuracy: {val_score:.4f}")

    y_pred = pipeline.predict(X_test)
    print("\n[INFO] Test set classification report:")
    print(classification_report(y_test, y_pred))
