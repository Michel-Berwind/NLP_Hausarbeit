"""
Evaluation Module - Offer-based Matching (No Bounding Boxes)

Calculates Precision, Recall, and F1 score by comparing pipeline predictions
against ground truth annotations using product+price matching.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False
    print("Warning: rapidfuzz not installed. Install with: pip install rapidfuzz")


def normalize_price(price_str: str) -> Optional[str]:
    """
    Normalize price string to standard format (e.g., "2.99").
    
    Handles formats like:
    - "2,99" → "2.99"
    - "2.99*" → "2.99"
    - "ab 1.59" → "1.59"
    - "1 59" → "1.59"
    - "€2,99" → "2.99"
    
    Args:
        price_str: Raw price string
        
    Returns:
        Normalized price string or None if parsing fails
    """
    if not price_str:
        return None
    
    # Remove common prefixes and symbols
    price = price_str.lower()
    price = re.sub(r'^(ab|ca\.?|circa|~)\s*', '', price)
    price = re.sub(r'[€$£*\s]', '', price)
    
    # Replace comma with dot
    price = price.replace(',', '.')
    
    # Extract first valid price pattern (digits with optional dot)
    match = re.search(r'(\d+\.?\d*)', price)
    if match:
        return match.group(1)
    
    return None


def normalize_product(product_str: str) -> str:
    """
    Normalize product string for comparison.
    
    - Convert to lowercase
    - Remove punctuation
    - Collapse whitespace
    
    Args:
        product_str: Raw product string
        
    Returns:
        Normalized product string
    """
    if not product_str:
        return ""
    
    # Lowercase
    text = product_str.lower()
    
    # Remove punctuation (keep alphanumeric, umlauts, and spaces)
    text = re.sub(r'[^\wäöüß\s]', ' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def load_ground_truth(annotation_path: Path) -> List[Dict]:
    """Load ground truth offers from JSON file."""
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("offers", [])


def load_predictions(prediction_path: Path) -> List[Dict]:
    """Load pipeline predictions from JSON file."""
    with open(prediction_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # Handle both single result and list of results
    if isinstance(data, list):
        if len(data) > 0:
            first = data[0]
            if "items" in first:
                return first.get("items", [])
            if "offers" in first:
                return first.get("offers", [])
        return []
    else:
        if "items" in data:
            return data.get("items", [])
        if "offers" in data:
            return data.get("offers", [])
        return []


def calculate_product_similarity(prod1: str, prod2: str) -> float:
    """
    Calculate product name similarity using RapidFuzz.
    
    Args:
        prod1: First product name (normalized)
        prod2: Second product name (normalized)
        
    Returns:
        Similarity score (0-1)
    """
    if not RAPIDFUZZ_AVAILABLE:
        # Fallback: exact match only
        return 1.0 if prod1 == prod2 else 0.0
    
    # token_set_ratio is more robust to word order
    score = fuzz.token_set_ratio(prod1, prod2)
    return score / 100.0


def match_offers(
    predictions: List[Dict],
    ground_truth: List[Dict],
    price_exact: bool = True,
    product_threshold: float = 0.80
) -> Tuple[int, int, int, List[Dict], List[Dict]]:
    """
    Match predicted offers with ground truth offers.
    
    Matching rule:
    - Price must match exactly (after normalization)
    - Product similarity must be >= product_threshold
    - Greedy matching (each GT offer matched at most once)
    
    Args:
        predictions: List of predicted offers
        ground_truth: List of ground truth offers
        price_exact: Whether price must match exactly
        product_threshold: Minimum product similarity (0-1)
        
    Returns:
        (true_positives, false_positives, false_negatives, fp_list, fn_list)
    """
    matched_gt = set()
    true_positives = 0
    false_positives_list = []
    
    # Normalize all offers first
    gt_normalized = []
    for gt in ground_truth:
        gt_normalized.append({
            "product": normalize_product(gt.get("product", "")),
            "price": normalize_price(gt.get("price", "")),
            "original": gt
        })
    
    pred_normalized = []
    for pred in predictions:
        # Handle different possible structures
        product = pred.get("product") or pred.get("product_nlp", {}).get("product_name", "")
        price = pred.get("price", "")
        
        pred_normalized.append({
            "product": normalize_product(product),
            "price": normalize_price(price),
            "original": pred
        })
    
    # Match each prediction
    for pred_norm in pred_normalized:
        pred_price = pred_norm["price"]
        pred_product = pred_norm["product"]
        
        if not pred_price:
            # No price = automatic FP
            false_positives_list.append({
                "prediction": pred_norm["original"],
                "reason": "no_price"
            })
            continue
        
        best_match_idx = -1
        best_similarity = 0.0
        
        # Find best matching GT offer
        for i, gt_norm in enumerate(gt_normalized):
            if i in matched_gt:
                continue
            
            gt_price = gt_norm["price"]
            gt_product = gt_norm["product"]
            
            # Check price match
            if price_exact and pred_price != gt_price:
                continue
            
            # Check product similarity
            similarity = calculate_product_similarity(pred_product, gt_product)
            
            if similarity >= product_threshold and similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
        
        if best_match_idx >= 0:
            # Match found
            true_positives += 1
            matched_gt.add(best_match_idx)
        else:
            # No match = FP
            false_positives_list.append({
                "prediction": pred_norm["original"],
                "reason": "no_match"
            })
    
    false_positives = len(false_positives_list)
    
    # Unmatched GT offers = FN
    false_negatives_list = []
    for i, gt_norm in enumerate(gt_normalized):
        if i not in matched_gt:
            false_negatives_list.append({
                "ground_truth": gt_norm["original"],
                "reason": "not_detected"
            })
    
    false_negatives = len(false_negatives_list)
    
    return true_positives, false_positives, false_negatives, false_positives_list, false_negatives_list


def calculate_metrics(tp: int, fp: int, fn: int) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        tp: True positives
        fp: False positives
        fn: False negatives
        
    Returns:
        Dictionary with precision, recall, and f1_score
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn
    }


def evaluate_single_page(
    prediction_path: Path,
    annotation_path: Path,
    product_threshold: float = 0.80
) -> Dict:
    """
    Evaluate predictions for a single page.
    
    Args:
        prediction_path: Path to prediction JSON file
        annotation_path: Path to annotation JSON file
        product_threshold: Minimum product similarity threshold
        
    Returns:
        Dictionary with metrics and error analysis
    """
    # Load data
    predictions = load_predictions(prediction_path)
    ground_truth = load_ground_truth(annotation_path)
    
    # Match offers
    tp, fp, fn, fp_list, fn_list = match_offers(
        predictions,
        ground_truth,
        price_exact=True,
        product_threshold=product_threshold
    )
    
    # Calculate metrics
    metrics = calculate_metrics(tp, fp, fn)
    
    # Add error analysis
    metrics["false_positives_details"] = fp_list
    metrics["false_negatives_details"] = fn_list
    metrics["page_id"] = annotation_path.stem
    
    return metrics


def evaluate_directory(
    predictions_dir: Path,
    annotations_dir: Path,
    product_threshold: float = 0.80,
    output_file: Optional[Path] = None
) -> Dict:
    """
    Evaluate all predictions in a directory.
    
    Args:
        predictions_dir: Directory containing prediction JSON files
        annotations_dir: Directory containing annotation JSON files
        product_threshold: Minimum product similarity threshold
        output_file: Optional path to save detailed results
        
    Returns:
        Dictionary with aggregated metrics and per-page results
    """
    predictions_dir = Path(predictions_dir)
    annotations_dir = Path(annotations_dir)
    
    # Find all annotation files
    annotation_files = list(annotations_dir.glob("*.json"))
    
    if not annotation_files:
        print(f"No annotation files found in {annotations_dir}")
        return {}
    
    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_page_results = []
    
    print(f"Evaluating {len(annotation_files)} pages...")
    
    for annotation_file in sorted(annotation_files):
        page_id = annotation_file.stem
        prediction_file = predictions_dir / f"{page_id}.json"
        
        if not prediction_file.exists():
            print(f"Warning: No prediction file for {page_id}")
            continue
        
        # Evaluate single page
        page_metrics = evaluate_single_page(
            prediction_file,
            annotation_file,
            product_threshold=product_threshold
        )
        
        # Accumulate totals
        total_tp += page_metrics["true_positives"]
        total_fp += page_metrics["false_positives"]
        total_fn += page_metrics["false_negatives"]
        
        per_page_results.append(page_metrics)
        
        print(f"  {page_id}: P={page_metrics['precision']:.2f} R={page_metrics['recall']:.2f} F1={page_metrics['f1_score']:.2f}")
    
    # Calculate overall metrics
    overall_metrics = calculate_metrics(total_tp, total_fp, total_fn)
    
    results = {
        "overall": overall_metrics,
        "per_page": per_page_results,
        "config": {
            "product_threshold": product_threshold
        }
    }
    
    print("\n" + "="*60)
    print("OVERALL METRICS:")
    print(f"  Precision: {overall_metrics['precision']:.3f}")
    print(f"  Recall:    {overall_metrics['recall']:.3f}")
    print(f"  F1 Score:  {overall_metrics['f1_score']:.3f}")
    print(f"  TP: {overall_metrics['true_positives']}, FP: {overall_metrics['false_positives']}, FN: {overall_metrics['false_negatives']}")
    print("="*60)
    
    # Save results if requested
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to {output_file}")
    
    return results


def main():
    """Main entry point for evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate product/price extraction results"
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Directory with prediction JSON files"
    )
    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory with ground truth annotation files"
    )
    parser.add_argument(
        "--product-threshold",
        type=float,
        default=0.80,
        help="Product similarity threshold (0-1, default: 0.80)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional: Save detailed results to JSON file"
    )
    
    args = parser.parse_args()
    
    # Run evaluation
    results = evaluate_directory(
        predictions_dir=args.predictions,
        annotations_dir=args.annotations,
        product_threshold=args.product_threshold,
        output_file=args.output
    )
    
    if not results:
        print("No results generated. Check input directories.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
