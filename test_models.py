"""
Test if the regenerated models load correctly
"""
import joblib
import sys

print("Testing model loading...\n")

models_to_test = [
    'Model/movies_data.joblib',
    'Model/similarity.joblib',
    'Model/sentiment_analysis_model.pkl',
    'Model/tfidf_vectorizer.pkl'
]

all_good = True
for model_path in models_to_test:
    try:
        print(f"Loading {model_path}...", end=" ")
        model = joblib.load(model_path)
        print("✓ SUCCESS")
    except EOFError as e:
        print(f"✗ FAILED - EOFError: {e}")
        all_good = False
    except Exception as e:
        print(f"✗ FAILED - {type(e).__name__}: {e}")
        all_good = False

print("\n" + "="*50)
if all_good:
    print("✓ All models loaded successfully!")
    print("The EOFError issue has been fixed.")
    sys.exit(0)
else:
    print("✗ Some models failed to load")
    sys.exit(1)
