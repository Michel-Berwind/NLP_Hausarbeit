"""Quick comparison of ground truth vs prediction for aldi_page01."""
import json

gt = json.load(open('data/annotations/aldi_page01.json'))
pred = json.load(open('data/predictions/aldi_page01.json'))[0]['items']

print("=" * 60)
print("GROUND TRUTH (5 items)")
print("=" * 60)
for o in gt['offers']:
    print(f"  {o['product']:45} @ {o['price']}")

print("\n" + "=" * 60)
print("PREDICTIONS (5 items)")
print("=" * 60)
for i in pred:
    product = i.get('product') or '(no product)'
    price = i.get('price')
    print(f"  {product:45} @ {price}")

print("\n" + "=" * 60)
print("MATCHES")
print("=" * 60)
from rapidfuzz import fuzz

gt_offers = [(o['product'], o['price']) for o in gt['offers']]
pred_offers = [(i.get('product') or '', i.get('price')) for i in pred]

matches = []
for gt_prod, gt_price in gt_offers:
    for pred_prod, pred_price in pred_offers:
        if gt_price == pred_price:
            similarity = fuzz.token_set_ratio(gt_prod.lower(), pred_prod.lower()) / 100
            if similarity >= 0.80:
                matches.append((gt_prod, pred_prod, gt_price, similarity))
                print(f"  ✓ {gt_prod[:30]:30} matched @ {gt_price} (sim={similarity:.2f})")

print(f"\nMatches: {len(matches)} / {len(gt_offers)} = {len(matches)/len(gt_offers)*100:.0f}% recall")
