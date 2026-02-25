"""Detailed analysis of all 5 completed Aldi pages."""
import json
from pathlib import Path

result_file = Path('results/aldi_5pages_eval.json')
with open(result_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

pages = [p for p in data['per_page'] if 'aldi_page' in p.get('page_id', '')][:5]

for page in pages:
    page_id = page['page_id']
    print(f"\n{'='*60}")
    print(f"{page_id.upper()}")
    print(f"{'='*60}")
    print(f"TP={page['true_positives']}, FP={page['false_positives']}, FN={page['false_negatives']}")
    print(f"P={page['precision']:.2f}, R={page['recall']:.2f}, F1={page['f1_score']:.2f}")
    
    print(f"\n📊 Ground Truth ({len(page['false_negatives_details']) + page['true_positives']} items):")
    # Show all ground truth items
    gt_file = Path(f'data/annotations/{page_id}.json')
    with open(gt_file, 'r', encoding='utf-8') as f:
        gt_data = json.load(f)
    for i, offer in enumerate(gt_data['offers'], 1):
        print(f"  {i}. {offer['product']:45s} @ {offer['price']}")
    
    print(f"\n🔍 Predictions ({len(page['false_positives_details']) + page['true_positives']} items):")
    # Show all predictions
    pred_file = Path(f'data/predictions/{page_id}.json')
    with open(pred_file, 'r', encoding='utf-8') as f:
        pred_data = json.load(f)
    for i, item in enumerate(pred_data[0]['items'], 1):
        product = item.get('product') or '(no product)'
        price = item.get('price', '?')
        print(f"  {i}. {product[:45]:45s} @ {price}")
    
    if page['false_positives'] > 0:
        print(f"\n❌ False Positives ({page['false_positives']}):")
        for fp in page['false_positives_details'][:3]:
            pred = fp['prediction']
            product = pred.get('product') or '(no product)'
            price = pred.get('price', '?')
            print(f"  - {product[:45]:45s} @ {price}")
    
    if page['false_negatives'] > 0:
        print(f"\n⚠️  False Negatives ({page['false_negatives']}):")
        for fn in page['false_negatives_details'][:3]:
            gt = fn['ground_truth']
            print(f"  - {gt['product'][:45]:45s} @ {gt['price']}")

print(f"\n{'='*60}")
print(f"OVERALL: P={data['overall']['precision']:.2f}, R={data['overall']['recall']:.2f}, F1={data['overall']['f1_score']:.2f}")
print(f"{'='*60}")
