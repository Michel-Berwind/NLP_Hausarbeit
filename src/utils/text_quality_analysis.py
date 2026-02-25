"""
Text Quality Analysis for Product Name Extraction

Uses visual features to distinguish product titles from descriptions:
- Font weight (bold vs regular text)
- Font size (large vs small text)
- Stroke width (thick vs thin characters)
- Text density (compact vs sparse)

Product titles are typically:
- Bold/thick font
- Larger size
- Higher contrast
- Denser stroke width
"""

from __future__ import annotations

from typing import Tuple, Dict
import cv2
import numpy as np


class TextQualityAnalyzer:
    """Analyze visual text quality to prioritize product names."""
    
    def __init__(self):
        self.min_bold_ratio = 0.15  # Minimum pixel density for bold text
        self.min_large_height = 25  # Minimum height for large text (after scaling)
        
    def analyze_text_region(self, crop: np.ndarray) -> Dict[str, float]:
        """
        Analyze visual characteristics of a text region.
        
        Args:
            crop: BGR image crop containing text
        
        Returns:
            Dictionary with quality metrics:
            - boldness: 0-1, higher = bolder text
            - size_score: 0-1, higher = larger text
            - stroke_width: Average character thickness
            - is_title: True if likely a product title
            - confidence: Overall confidence this is a title
        """
        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            return self._empty_result()
        
        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()
        
        # Measure various text characteristics
        boldness = self._measure_boldness(gray)
        size_score = self._measure_size(gray)
        stroke_width = self._measure_stroke_width(gray)
        contrast = self._measure_contrast(gray)
        density = self._measure_text_density(gray)
        
        # Determine if this looks like a title
        is_title = (
            boldness > 0.15 or  # Bold text
            size_score > 0.6 or  # Large text
            stroke_width > 3.0  # Thick strokes
        )
        
        # Calculate confidence score
        confidence = self._calculate_title_confidence(
            boldness, size_score, stroke_width, contrast, density
        )
        
        return {
            "boldness": boldness,
            "size_score": size_score,
            "stroke_width": stroke_width,
            "contrast": contrast,
            "density": density,
            "is_title": is_title,
            "confidence": confidence
        }
    
    def _measure_boldness(self, gray: np.ndarray) -> float:
        """
        Measure text boldness/thickness using pixel density.
        
        Bold text has more black pixels relative to the region size.
        """
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Calculate black pixel ratio
        total_pixels = binary.size
        black_pixels = np.sum(binary > 0)
        
        if total_pixels == 0:
            return 0.0
        
        boldness = black_pixels / total_pixels
        return float(boldness)
    
    def _measure_size(self, gray: np.ndarray) -> float:
        """
        Estimate text size from connected components.
        
        Larger text = larger bounding boxes for characters.
        """
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find connected components (individual characters)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        
        if num_labels < 2:  # Only background
            return 0.0
        
        # Get heights of components (skip background at index 0)
        heights = []
        for i in range(1, num_labels):
            h = stats[i, cv2.CC_STAT_HEIGHT]
            w = stats[i, cv2.CC_STAT_WIDTH]
            area = stats[i, cv2.CC_STAT_AREA]
            
            # Filter noise (very small components)
            if area > 10 and h > 3 and w > 2:
                heights.append(h)
        
        if not heights:
            return 0.0
        
        # Use median height as size indicator
        median_height = np.median(heights)
        
        # Normalize to 0-1 range (assume max height ~100px for titles)
        size_score = min(median_height / 50.0, 1.0)
        return float(size_score)
    
    def _measure_stroke_width(self, gray: np.ndarray) -> float:
        """
        Estimate character stroke width using morphological operations.
        
        Thicker strokes = bolder font = likely title.
        """
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Use distance transform to measure stroke width
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
        
        # Average distance from edge = approximate stroke width
        text_pixels = binary > 0
        if not text_pixels.any():
            return 0.0
        
        avg_stroke_width = np.mean(dist_transform[text_pixels]) * 2
        return float(avg_stroke_width)
    
    def _measure_contrast(self, gray: np.ndarray) -> float:
        """
        Measure contrast between text and background.
        
        High contrast = clearer text = more likely to be important (title).
        """
        if gray.size == 0:
            return 0.0
        
        min_val = np.min(gray)
        max_val = np.max(gray)
        
        contrast = (max_val - min_val) / 255.0
        return float(contrast)
    
    def _measure_text_density(self, gray: np.ndarray) -> float:
        """
        Measure how compact the text is.
        
        Titles are usually more compact, descriptions more sparse.
        """
        # Binarize
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find text bounding box
        text_pixels = np.argwhere(binary > 0)
        if len(text_pixels) == 0:
            return 0.0
        
        # Calculate bounding box
        y_min, x_min = text_pixels.min(axis=0)
        y_max, x_max = text_pixels.max(axis=0)
        
        bbox_area = (y_max - y_min + 1) * (x_max - x_min + 1)
        text_area = np.sum(binary > 0)
        
        if bbox_area == 0:
            return 0.0
        
        density = text_area / bbox_area
        return float(density)
    
    def _calculate_title_confidence(
        self, 
        boldness: float, 
        size_score: float, 
        stroke_width: float,
        contrast: float,
        density: float
    ) -> float:
        """
        Calculate overall confidence that this is a product title.
        
        Weighted combination of visual features.
        """
        # Weights for each feature
        weights = {
            "boldness": 0.30,      # Most important
            "size_score": 0.25,    # Second most important
            "stroke_width": 0.20,  # Third
            "contrast": 0.15,      # Fourth
            "density": 0.10        # Fifth
        }
        
        # Normalize stroke width to 0-1 (assume max ~10px)
        stroke_normalized = min(stroke_width / 10.0, 1.0)
        
        # Calculate weighted score
        confidence = (
            weights["boldness"] * boldness +
            weights["size_score"] * size_score +
            weights["stroke_width"] * stroke_normalized +
            weights["contrast"] * contrast +
            weights["density"] * density
        )
        
        return float(min(confidence, 1.0))
    
    def _empty_result(self) -> Dict[str, float]:
        """Return empty result for invalid crops."""
        return {
            "boldness": 0.0,
            "size_score": 0.0,
            "stroke_width": 0.0,
            "contrast": 0.0,
            "density": 0.0,
            "is_title": False,
            "confidence": 0.0
        }


def is_likely_product_title(crop: np.ndarray) -> Tuple[bool, float]:
    """
    Quick check if a text region is likely a product title.
    
    Args:
        crop: BGR image crop
    
    Returns:
        (is_title, confidence)
    """
    analyzer = TextQualityAnalyzer()
    result = analyzer.analyze_text_region(crop)
    return result["is_title"], result["confidence"]


def filter_text_regions_by_boldness(
    regions: list, 
    crops: list
) -> Tuple[list, list]:
    """
    Filter text regions to keep only bold/large text (likely titles).
    
    Args:
        regions: List of text region coordinates
        crops: List of corresponding image crops
    
    Returns:
        (filtered_regions, filtered_crops) - only title-like text
    """
    analyzer = TextQualityAnalyzer()
    
    filtered_regions = []
    filtered_crops = []
    
    for region, crop in zip(regions, crops):
        analysis = analyzer.analyze_text_region(crop)
        
        # Keep if it looks like a title
        if analysis["is_title"] and analysis["confidence"] > 0.3:
            filtered_regions.append(region)
            filtered_crops.append(crop)
    
    return filtered_regions, filtered_crops
