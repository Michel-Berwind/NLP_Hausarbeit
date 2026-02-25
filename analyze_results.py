"""Analyze evaluation results."""
import json
from pathlib import Path

result_file = Path('results/test_evaluation_no_brands.json')
with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("="*60)
print("ALDI_PAGE01 ANALYSIS")
print("="*60)

page = [p for p in data['per_page'] if p['page_id'] == 'aldi_page01'][0]
print(f"TP={page['true_positives']}, FP={page['false_positives']}, FN={page['false_negatives']}")
print(f"Precision={page['precision']:.2f}, Recall={page['recall']:.2f}, F1={page['f1_score']:.2f}")

print("\nFALSE POSITIVES (predicted but not matched):")
for fp in page['false_positives_details']:
    pred = fp['prediction']
    product = pred.get('product') or '(no product)'
    price = pred.get('price', '?')
    print(f"  ❌ {product[:50]:50s} @ {price}")

print("\nFALSE NEGATIVES (ground truth not detected):")
for fn in page['false_negatives_details']:
    gt = fn['ground_truth']
    print(f"  ⚠️  {gt['product'][:50]:50s} @ {gt['price']}")

print("\n" + "="*60)



