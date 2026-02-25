"""
NLP-based Entity Extraction for Product Information

This module uses spaCy for Natural Language Processing to extract:
- Product names (using NER and POS tagging)
- Brand names (custom NER + entity recognition)
- Quantities and units (pattern matching + entity extraction)
- Product attributes (adjectives, compound nouns)

Techniques used:
- Named Entity Recognition (NER)
- Part-of-Speech (POS) tagging
- Dependency parsing
- Custom entity patterns
- Token-level linguistic features
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import re
import sys
import subprocess

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc, Span


class ProductNERExtractor:
    """NLP-based extractor for product information from OCR text."""
    
    def __init__(self, model_name: str = "de_core_news_sm"):
        """
        Initialize spaCy NLP pipeline.
        
        Args:
            model_name: spaCy German model to use
                - de_core_news_sm: Small, fast model
                - de_core_news_md: Medium model with word vectors
                - de_core_news_lg: Large model, best accuracy
        """
        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"Model '{model_name}' not found. Downloading...")
            subprocess.run([sys.executable, "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)
        
        # Add custom brand entity patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._add_brand_patterns()
        
        # Known brand names for entity recognition
        self.known_brands = {
            "BARISSIMO", "BARISTA", "NESCAFE", "NESCAFÉ", "JACOBS", "DALLMAYR", 
            "MELITTA", "LAVAZZA", "TCHIBO", "CREMESSO", "SENSEO", "DOLCE GUSTO",
            "TASSIMO", "MILKA", "LINDT", "RITTER SPORT", "HARIBO", "NUTELLA",
            "WORKZONE", "ALDI",
            "FERREX", "TOPCRAFT", "UPAFASHION", "UP2FASHION", "CRANE",
            # Aldi-specific brands found in test data:
            "RIDE", "GO", "HOME", "CREATION", "GARDENLINE", "ACTIV", "ENERGY",
            "LILY", "DAN", "EXPERTIZ", "BACK", "FAMILY", "FARMER", "NATURALS", "LIVE", "STYLE"
            # NOTE: ESPRESSO, LUNGO, RISTRETTO are product TYPES, not brands!
        }
        
        # Non-product stopwords to filter out
        # IMPORTANT: Only disclaimer/legal words, NOT spec words!
        self.non_product_words = {
            "AKTIONSARTIKEL", "SORTIMENT", "VORMITTAG", "NACHMITTAG", "AKTIONSTAG",
            "UNTERSCHIED", "DEKORATION", "ANGEBOT", "AKTION", "BEGRENZTER",
            "BEACHTUNG", "VERFÜGUNG", "ANZAHL"
        }
        
        # Product type keywords (strong indicators of actual product name)
        self.product_type_keywords = {
            "KAFFEE", "ESPRESSO", "CAPPUCCINO", "MAHLKAFFEE", "BOHNENKAFFEE",
            "KAFFEEKAPSELN", "KAPSELN", "CAPSULES", "CAPS", "BOHNEN", "LUNGO", "RISTRETTO",
            "SCHOKOLADE", "RIEGEL", "TAFEL", "PRALINE",
            "SAFT", "TEE", "GETRÄNK", "WASSER", "COLA", "LIMONADE",
            "MÜSLI", "CEREALIEN", "CORNFLAKES",
            "BROT", "BRÖTCHEN", "KUCHEN", "GEBÄCK",
            "WURST", "KÄSE", "BUTTER", "JOGHURT", "QUARK"
        }
        
        # Attribute keywords (descriptive, but NOT the core product name)
        # These should be EXCLUDED from the final product name
        self.attribute_keywords = {
            "CHARAKTERVOLL", "AROMATISCH", "MILD", "KRÄFTIG", "INTENSIV",
            "CREMIG", "VOLLMUNDIG", "AUSGEWOGEN", "SANFT", "FEIN",
            "LECKER", "KÖSTLICH", "FEINSTER", "PREMIUM",
            "WHOLE", "BEANS", "GANZE",  # English/parts of product
            "VERSCH", "VERSCHIEDENE", "SORTEN",  # Generic descriptors
            "ALUMINIUM", "PLASTIK", "METALL", "GLAS", "PAPIER",  # Materials
            "COMPATIBLE", "KOMPATIBEL"  # Technical specs
        }
    
    def _add_brand_patterns(self) -> None:
        """Add matcher patterns for brand recognition."""
        # Pattern: Capitalized words (brands are usually capitalized)
        brand_pattern = [{"IS_UPPER": True, "LENGTH": {">": 2}}]
        self.matcher.add("BRAND", [brand_pattern])
    
    def _find_product_type_tokens(self, doc: Doc) -> List[Tuple[int, str]]:
        """Find tokens that indicate product types (Kaffee, Espresso, Kaffeekapseln, etc.)"""
        product_type_positions = []
        for i, token in enumerate(doc):
            text_upper = token.text.upper()
            # Check if token or its lemma matches product type
            if text_upper in self.product_type_keywords:
                product_type_positions.append((i, token.text))
            # Check for compound words containing product type
            elif any(pt in text_upper for pt in self.product_type_keywords):
                product_type_positions.append((i, token.text))
        return product_type_positions
        
        # Pattern: Product types with adjectives
        product_pattern = [
            {"POS": "ADJ", "OP": "*"},  # Optional adjectives
            {"POS": "NOUN"},            # Main noun
            {"POS": "NOUN", "OP": "*"}  # Optional compound nouns
        ]
        self.matcher.add("PRODUCT", [product_pattern])
    
    def extract_entities(self, text: str) -> Dict[str, any]:
        """
        Extract structured entities from product text using NLP.
        
        Args:
            text: Raw OCR text
        
        Returns:
            Dictionary with extracted entities:
            - product_name: Main product name (cleaned)
            - brands: List of detected brands
            - entities: List of all named entities found
            - pos_filtered: Text filtered by POS tags (nouns/proper nouns only)
            - noun_chunks: Meaningful noun phrases
            - quantities: Extracted quantities with units
        """
        if not text or len(text.strip()) < 3:
            return self._empty_result()
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Extract various entity types
        result = {
            "product_name": self._extract_product_name(doc, debug=True),  # Temporarily enable debug
            "brands": self._extract_brands(doc),
            "entities": self._extract_named_entities(doc),
            "pos_filtered": self._pos_filter_text(doc),
            "noun_chunks": self._extract_noun_chunks(doc),
            "quantities": self._extract_quantities(doc),
            "attributes": self._extract_attributes(doc),
            "confidence": self._calculate_confidence(doc)
        }
        
        return result
    
    def _extract_product_name(self, doc: Doc, debug=False) -> str:
        """
        PHASE 2 SIMPLIFIED: Extract product name using simple, focused strategy.
        
        Strategy:
        1. Look for known brand in text
        2. If brand found: Brand + next 2-3 NOUNs/PROPNs
        3. If no brand: First 2-3 NOUNs/PROPNs from the text
        4. Stop at comma, period, or description words
        
        This avoids the complex scoring that was picking description details
        instead of the main product title.
        """
        # Early rejection for non-product text
        text_upper = doc.text.upper()
        
        # Reject disclaimer text
        disclaimer_phrases = ['WEITERE FARBE', 'IM UNTERSCHIED', 'STÄNDIG VORHANDENEN', 
                             'AM VORMITTAG', 'WIR BITTEN', 'SIEHE ABBILDUNG']
        for phrase in disclaimer_phrases:
            if text_upper.startswith(phrase):
                return ""
        
        # Reject if dominated by non-product words
        if any(non_prod in text_upper for non_prod in self.non_product_words):
            non_prod_count = sum(1 for np in self.non_product_words if np in text_upper)
            if non_prod_count >= 3:
                return ""
        
        # Strategy: Find BRAND + NOUNs OR just first NOUNs
        product_parts = []
        found_brand = False
        brand_position = -1
        
        # Find brand position
        for i, token in enumerate(doc):
            if token.text.upper() in self.known_brands:
                found_brand = True
                brand_position = i
                product_parts.append(token.text)
                break
        
        # Collect nouns after brand (or from start if no brand)
        start_pos = brand_position + 1 if found_brand else 0
        noun_count = 0
        max_nouns = 3
        
        for i in range(start_pos, len(doc)):
            token = doc[i]
            
            # Stop at punctuation marks
            if token.text in [',', '.', ':', ';']:
                break
            
            # Stop at description/attribute words
            if token.text.upper() in self.attribute_keywords:
                break
            
            # Stop at verbs (start of description)
            if token.pos_ in ["VERB", "AUX"]:
                break
            
            # Collect nouns and proper nouns
            if token.pos_ in ["NOUN", "PROPN"]:
                product_parts.append(token.text)
                noun_count += 1
                if noun_count >= max_nouns:
                    break
            
            # Allow numbers right after brand (e.g., "FERREX 20V")
            elif token.pos_ == "NUM" and i == start_pos:
                product_parts.append(token.text)
            
            # Allow connectors in compound names (e.g., "Fahrrad-Spiralkabelschloss")
            elif token.text in ['-', 'und', '/'] and len(product_parts) > 0 and noun_count < max_nouns:
                product_parts.append(token.text)
            
            # Allow adjectives before first noun
            elif token.pos_ == "ADJ" and noun_count == 0 and len(product_parts) < 3:
                product_parts.append(token.text)
            
            # Stop at anything else
            else:
                break
        
        # Build final product name
        if len(product_parts) == 0:
            return ""
        
        product_name = " ".join(product_parts)
        
        # Clean up the result
        cleaned = self._clean_product_name(product_name)
        if not cleaned:
            return ""

        tokens = [t for t in re.split(r"\s+", cleaned) if t]
        if not tokens:
            return ""

        generic_tokens = {"GERMANY", "MADE", "IN"}
        meaningful_tokens = []
        for token in tokens:
            token_norm = re.sub(r"[^A-Za-zÄÖÜäöüß]", "", token).upper()
            if not token_norm:
                continue
            if token_norm in generic_tokens:
                continue
            if token_norm in self.known_brands:
                continue
            meaningful_tokens.append(token_norm)

        # Reject brand-only / meta-only strings like "HOME CREATION" or "GERMANY RIDE"
        if not meaningful_tokens:
            return ""

        return cleaned
    
    def _extract_brands(self, doc: Doc) -> List[str]:
        """Extract brand names using NER and pattern matching."""
        brands = set()
        
        # Method 1: Known brands from list
        for token in doc:
            if token.text.upper() in self.known_brands:
                brands.add(token.text)
        
        # Method 2: Named entities (ORG)
        for ent in doc.ents:
            if ent.label_ == "ORG":
                brands.add(ent.text)
        
        # Method 3: Capitalized multi-char words (likely brands)
        for token in doc:
            if token.is_upper and len(token.text) > 2 and token.is_alpha:
                brands.add(token.text)
        
        return sorted(list(brands))
    
    def _extract_named_entities(self, doc: Doc) -> List[Dict[str, str]]:
        """Extract all named entities with their types."""
        entities = []
        for ent in doc.ents:
            entities.append({
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char
            })
        return entities
    
    def _pos_filter_text(self, doc: Doc) -> str:
        """
        Filter text by POS tags to remove OCR noise.
        
        Keep only:
        - NOUN: Common nouns (product types)
        - PROPN: Proper nouns (brands, names)
        - ADJ: Adjectives (product attributes)
        - NUM: Numbers (quantities)
        
        Remove:
        - VERB, ADP, CONJ, etc. (unlikely in product names)
        """
        relevant_pos = {"NOUN", "PROPN", "ADJ", "NUM"}
        filtered_tokens = [token.text for token in doc if token.pos_ in relevant_pos]
        return " ".join(filtered_tokens)
    
    def _extract_noun_chunks(self, doc: Doc) -> List[str]:
        """Extract meaningful noun phrases (product descriptions)."""
        chunks = []
        for chunk in doc.noun_chunks:
            # Filter out single-word chunks that are just determiners
            if len(chunk) > 1 or chunk.root.pos_ in ["NOUN", "PROPN"]:
                chunks.append(chunk.text)
        return chunks
    
    def _extract_quantities(self, doc: Doc) -> List[Dict[str, str]]:
        """
        Extract quantities with units using NER and patterns.
        
        Examples: "500g", "2x250g", "20 Kapseln"
        """
        quantities = []
        
        # Pattern 1: Number + unit (e.g., "500g", "20 Kapseln")
        quantity_pattern = r'(\d+(?:[,.]\d+)?)\s*([a-zA-ZäöüÄÖÜ]+)'
        for match in re.finditer(quantity_pattern, doc.text):
            quantities.append({
                "value": match.group(1),
                "unit": match.group(2),
                "full_text": match.group(0)
            })
        
        # Pattern 2: Use POS tags to find NUM + NOUN combinations
        for i, token in enumerate(doc):
            if token.pos_ == "NUM":
                # Check if next token is a unit
                if i < len(doc) - 1 and doc[i+1].pos_ == "NOUN":
                    quantities.append({
                        "value": token.text,
                        "unit": doc[i+1].text,
                        "full_text": f"{token.text} {doc[i+1].text}"
                    })
        
        return quantities
    
    def _extract_attributes(self, doc: Doc) -> List[str]:
        """Extract product attributes (adjectives describing the product)."""
        attributes = []
        for token in doc:
            if token.pos_ == "ADJ":
                attributes.append(token.text)
        return attributes
    
    def _calculate_confidence(self, doc: Doc) -> float:
        """
        Calculate confidence score for the extraction.
        
        Based on:
        - Number of entities found
        - POS tag distribution
        - Presence of known brands
        - Text quality indicators
        """
        score = 0.5  # Base score
        
        # Bonus for entities
        if doc.ents:
            score += 0.2
        
        # Bonus for proper nouns
        propn_count = sum(1 for token in doc if token.pos_ == "PROPN")
        if propn_count > 0:
            score += 0.15
        
        # Bonus for known brands
        text_upper = doc.text.upper()
        if any(brand in text_upper for brand in self.known_brands):
            score += 0.15
        
        # Penalty for too many numbers (OCR noise)
        digit_ratio = sum(c.isdigit() for c in doc.text) / max(len(doc.text), 1)
        if digit_ratio > 0.4:
            score -= 0.2
        
        return max(0.0, min(1.0, score))
    
    def _clean_product_name(self, text: str) -> str:
        """Clean up extracted product name."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        # Common OCR truncations/misspellings for frequent flyer brands
        ocr_word_fixes = {
            "CREAT": "CREATION",
            "UPAFASHION": "UP2FASHION",
        }
        for wrong, right in ocr_word_fixes.items():
            text = re.sub(rf'\b{wrong}\b', right, text, flags=re.IGNORECASE)
        
        # Remove leading/trailing punctuation
        text = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', text)
        
        return text
    
    def _empty_result(self) -> Dict[str, any]:
        """Return empty result structure."""
        return {
            "product_name": "",
            "brands": [],
            "entities": [],
            "pos_filtered": "",
            "noun_chunks": [],
            "quantities": [],
            "attributes": [],
            "confidence": 0.0
        }


# Singleton instance for pipeline integration
_extractor = None
_extractor_unavailable = False


def get_extractor(model_name: str = "de_core_news_sm") -> ProductNERExtractor:
    """Get or create the global NER extractor instance."""
    global _extractor, _extractor_unavailable
    if _extractor_unavailable:
        return None

    if _extractor is None:
        try:
            _extractor = ProductNERExtractor(model_name)
        except Exception as e:
            print(f"NLP extractor unavailable: {e}")
            _extractor_unavailable = True
            return None
    return _extractor


def extract_product_entities(text: str) -> Dict[str, any]:
    """
    Convenience function for extracting entities from text.
    
    Args:
        text: Raw OCR text
    
    Returns:
        Dictionary with extracted entities
    """
    extractor = get_extractor()
    if extractor is None:
        return {
            "product_name": "",
            "brands": [],
            "entities": [],
            "pos_filtered": "",
            "noun_chunks": [],
            "quantities": [],
            "attributes": [],
            "confidence": 0.0
        }
    return extractor.extract_entities(text)
