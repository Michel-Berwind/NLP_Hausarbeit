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
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", model_name], check=True)
            self.nlp = spacy.load(model_name)
        
        # Add custom brand entity patterns
        self.matcher = Matcher(self.nlp.vocab)
        self._add_brand_patterns()
        
        # Known brand names for entity recognition
        self.known_brands = {
            "BARISSIMO", "BARISTA", "NESCAFE", "NESCAFÉ", "JACOBS", "DALLMAYR", 
            "MELITTA", "LAVAZZA", "TCHIBO", "CREMESSO", "SENSEO", "DOLCE GUSTO",
            "TASSIMO", "MILKA", "LINDT", "RITTER SPORT", "HARIBO", "NUTELLA",
            "WORKZONE", "ALDI", "LIDL",
            "FERREX", "TOPCRAFT", "UPAFASHION", "CRANE"
            # NOTE: ESPRESSO, LUNGO, RISTRETTO are product TYPES, not brands!
        }
        
        # Non-product stopwords to filter out
        # IMPORTANT: Only disclaimer/legal words, NOT spec words!
        # Spec words (DREHZAHL, LEISTUNG, GARANTIE, STÜCK) appear in valid products
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
        Extract the main product name from potentially long text with descriptions.
        
        Strategy when text contains multiple sentences/phrases:
        1. Split into sentences and analyze each
        2. Prioritize earlier sentences (product name usually comes first)
        3. Look for patterns: BRAND + NOUN, Capitalized words
        4. Reject specification/disclaimer sentences
        5. Return the most product-like phrase
        """
        # Early rejection for non-product text
        text_upper = doc.text.upper()
        
        # Strong rejection if text STARTS with disclaimer phrases (not just contains them)
        disclaimer_phrases = ['WEITERE FARBE', 'IM UNTERSCHIED ZU UNSEREM', 'STÄNDIG VORHANDENEN SORTIMENT', 
                             'AM VORMITTAG DES ERSTEN', 'WIR BITTEN UM BEACHTUNG',
                             'DIESE AKTIONSARTIKEL', 'SIEHE ABBILDUNG']
        for phrase in disclaimer_phrases:
            if text_upper.startswith(phrase) or text_upper.startswith('WIR ' + phrase):
                if debug:
                    print(f"    DEBUG: Rejected - starts with disclaimer phrase: {phrase}")
                return ""
        
        # Check ratio of non-product words (only reject if heavily dominated by them)
        # BUT: If text is long and starts with a brand, try extraction first (disclaimer might be at end)
        has_brand_at_start = any(brand in text_upper[:50] for brand in self.known_brands)
        if any(non_prod in text_upper for non_prod in self.non_product_words):
            non_prod_count = sum(1 for np in self.non_product_words if np in text_upper)
            # Only reject if 4+ non-product words AND no brand at start (pure disclaimer text)
            if non_prod_count >= 4 and not (has_brand_at_start and len(doc.text) > 100):
                if debug:
                    matching_words = [np for np in self.non_product_words if np in text_upper]
                    print(f"    DEBUG: Rejected by non-product word count: {non_prod_count}, words: {matching_words}")
                return ""
        
        candidates = []
        
        # Strategy -1: Find product TYPE first (Mahlkaffee, Espresso, Kaffeekapseln)
        # This is the MOST important - the actual product type should be prioritized
        # BUT: Prioritize product types NEAR known brands (not text in product images!)
        product_type_positions = self._find_product_type_tokens(doc)
        if product_type_positions and debug:
            print(f"    Found product type tokens: {product_type_positions}")
        
        # Find positions of known brands in the text
        brand_positions = []
        for i, token in enumerate(doc):
            if token.text.upper() in self.known_brands:
                brand_positions.append(i)
                if debug:
                    print(f"    Found brand '{token.text}' at position {i}")
        
        for pos, type_word in product_type_positions:
            # Build product name around the type word
            product_parts = []
            
            # Look back for brand/qualifiers (max 3 tokens)
            start_pos = max(0, pos - 3)
            for i in range(start_pos, pos):
                token = doc[i]
                # Skip attribute words (materials, descriptors)
                if token.text.upper() in self.attribute_keywords:
                    continue
                # Include brand names, proper nouns, quoted text
                if (token.text.upper() in self.known_brands or 
                    token.pos_ == "PROPN" or
                    token.text.startswith('"') or
                    (token.pos_ in ["NOUN", "ADJ"] and token.text.upper() not in self.attribute_keywords)):
                    product_parts.append(token.text)
            
            # Add the product type itself
            product_parts.append(type_word)
            
            # Look ahead for additional qualifiers (max 2 tokens)
            # BUT skip attribute words (charaktervoll, aromatisch, etc.)
            end_pos = min(len(doc), pos + 3)
            for i in range(pos + 1, end_pos):
                token = doc[i]
                # Stop at attributes or connectors
                if token.text.upper() in self.attribute_keywords:
                    if debug:
                        print(f"      Stopped at attribute: {token.text}")
                    break
                if token.text in ["&", "und", ",", "."]:
                    break
                # Skip very short tokens (likely OCR noise)
                if len(token.text) < 3 and token.pos_ in ["PROPN", "NOUN"]:
                    if debug:
                        print(f"      Skipped short token: {token.text}")
                    continue  # Skip but keep going
                # Include relevant modifiers
                if token.pos_ in ["NOUN", "PROPN", "ADJ"] or token.text.startswith('"'):
                    product_parts.append(token.text)
                elif token.text == "oder":  # Handle OR constructions
                    # Check if there's another product after "oder"
                    if i + 1 < len(doc):
                        next_token = doc[i + 1]
                        if next_token.pos_ in ["NOUN", "PROPN"]:
                            product_parts.extend(["oder", next_token.text])
                            # Check for one more word after
                            if i + 2 < len(doc) and doc[i + 2].pos_ in ["NOUN", "PROPN"]:
                                product_parts.append(doc[i + 2].text)
                    break
            
            if product_parts:
                product_name = " ".join(product_parts)
                
                # Base score for product type
                score = 25.0  # HIGHEST PRIORITY - product type is key!
                
                # HUGE BONUS: If this product type is NEAR a known brand (within 5 tokens)
                # This filters out text that's IN the product image vs. the actual product name
                min_distance_to_brand = float('inf')
                if brand_positions:
                    min_distance_to_brand = min(abs(pos - bp) for bp in brand_positions)
                    if min_distance_to_brand <= 5:
                        score += 15.0  # HUGE bonus for being near a brand!
                        if debug:
                            print(f"      Near brand bonus: +15.0 (distance={min_distance_to_brand})")
                    elif min_distance_to_brand <= 10:
                        score += 5.0  # Smaller bonus
                        if debug:
                            print(f"      Near brand bonus: +5.0 (distance={min_distance_to_brand})")
                    else:
                        # Far from any brand - likely text IN product image
                        score -= 10.0  # PENALTY!
                        if debug:
                            print(f"      Far from brand penalty: -10.0 (distance={min_distance_to_brand})")
                else:
                    # No brand in text at all - uncertain
                    score -= 5.0
                
                # BONUS: Specific product types (Kaffeekapseln, Mahlkaffee) get extra points
                # vs generic coffee types (Espresso, Lungo, Ristretto)
                specific_types = {"KAFFEEKAPSELN", "KAPSELN", "CAPSULES", "MAHLKAFFEE", 
                                 "BOHNENKAFFEE", "SCHOKOLADE", "RIEGEL"}
                generic_types = {"ESPRESSO", "LUNGO", "RISTRETTO", "CAPPUCCINO", "KAFFEE"}
                
                type_word_upper = type_word.upper()
                if any(st in type_word_upper for st in specific_types):
                    score += 5.0  # Bonus for specific product type
                    if debug:
                        print(f"      Specific product type bonus: +5.0")
                elif any(gt in type_word_upper for gt in generic_types):
                    score -= 2.0  # Slight penalty for generic type
                    if debug:
                        print(f"      Generic product type penalty: -2.0")
                
                candidates.append((product_name, score, "product_type"))
                if debug:
                    print(f"      Added product type candidate: {product_name} (score={score})")
        
        # Strategy 0A: For most product text, analyze as whole (don't trust sentence segmentation)
        # Sentence segmentation often breaks on capital letters, numbers, which are common in products
        # Only use sentence-based for VERY long text (> 400 chars) with many separators
        comma_count = doc.text.count(',')
        use_simple_strategy = len(doc.text) < 400  # Most product text is under 400 chars
        
        if use_simple_strategy:
            if debug:
                print(f"    Using whole-text strategy (len={len(doc.text)}, commas={comma_count})")
            # Treat whole text as potential product name
            for i, token in enumerate(doc):
                is_brand = token.text.upper() in self.known_brands
                is_likely_brand = (token.pos_ == "PROPN" and 
                                 sum(c.isupper() for c in token.text) > len(token.text) * 0.5 and
                                 len(token.text) >= 4)
                
                if is_brand or is_likely_brand:
                    if debug:
                        print(f"      Found brand: {token.text} at position {i}")
                    product_parts = [token.text]
                    noun_count = 0
                    for j in range(i+1, min(i+7, len(doc))):  # Look ahead up to 7 tokens
                        next_token = doc[j]
                        # Reduced debug - only print first few
                        if debug and j < i+3:
                            print(f"        Checking: {next_token.text} (POS={next_token.pos_})")
                        if next_token.pos_ in ["NOUN", "PROPN"]:
                            product_parts.append(next_token.text)
                            noun_count += 1
                            if noun_count >= 3:  # Stop after 2-3 nouns (for compound names like "Asche- und Grobschmutzsauger")
                                break
                        elif next_token.pos_ == "NUM" and j == i+1:
                            # Numbers right after brand (e.g., "FERREX 20V")
                            product_parts.append(next_token.text)
                        elif next_token.text in ['-', 'und', '/'] and len(product_parts) < 5:
                            # Continue through connectors for compound names
                            product_parts.append(next_token.text)
                            # Don't count 'und' as breaking the noun sequence
                            continue
                        elif next_token.pos_ in ["ADJ"] and len(product_parts) < 3 and noun_count == 0:
                            # Allow adjectives before first noun
                            product_parts.append(next_token.text)
                        elif next_token.text == "," or next_token.pos_ in ["VERB", "ADP", "ADV"]:
                            # Stop at comma, verb, preposition, adverb (start of description)
                            break
                        elif next_token.text.endswith('.') and len(next_token.text) <= 8:
                            # Stop at abbreviations like "Versch."
                            break
                        else:
                            break
                    
                    if len(product_parts) > 1:
                        product_name = " ".join(product_parts)
                        score = 20.0 if is_brand else 15.0  # HIGHEST PRIORITY
                        candidates.append((product_name, score, "short_text_brand_noun"))
                        if debug:
                            print(f"        Added: {product_name} (score={score})")
                    break  # Only take first brand
        
        # Strategy 0B: Split by sentence and prioritize first sentence (likely title)
        # Only use if short_text strategy didn't find anything
        if not candidates:
            if debug:
                print(f"    Using sentence-based strategy")
        sentences = list(doc.sents)
        for sent_idx, sent in enumerate(sentences[:3]):  # Check first 3 sentences
            sent_text = sent.text.strip()
            
            # Skip sentences with disclaimer words
            if any(word in sent_text.upper() for word in ['BEACHTUNG', 'UNTERSCHIED', 'SORTIMENT', 'VORMITTAG', 'VERFÜGUNG', 'ANZAHL', 'BITTEN']):
                continue
            
            # Skip sentences that are mostly specs (lots of numbers/units)
            if sent_text.count('/') > 2 or sent_text.count(',') > 4:
                continue
            
            # Higher score for earlier sentences (product name typically first)
            position_bonus = 5.0 if sent_idx == 0 else (3.0 if sent_idx == 1 else 1.0)
            
            # Look for BRAND + NOUN pattern in first sentence
            sent_tokens = list(sent)
            for i, token in enumerate(sent_tokens):
                # Find known brands (handle OCR errors by checking if word is mostly uppercase)
                is_brand = token.text.upper() in self.known_brands
                is_likely_brand = (token.pos_ == "PROPN" and 
                                 sum(c.isupper() for c in token.text) > len(token.text) * 0.5 and
                                 len(token.text) >= 4)
                
                if debug and (is_brand or is_likely_brand):
                    print(f"      Found brand/propn: {token.text} (is_brand={is_brand}, POS={token.pos_}, i={i}, sent_len={len(sent_tokens)})")
                
                if is_brand or is_likely_brand:
                    # Build product name: BRAND + following NOUNs
                    product_parts = [token.text]
                    
                    # Look ahead for nouns (product type)
                    lookahead_range = range(i+1, min(i+5, len(sent_tokens)))
                    if debug:
                        print(f"        Lookahead range: {list(lookahead_range)}")
                    for j in range(i+1, min(i+5, len(sent_tokens))):
                        next_token = sent_tokens[j]
                        if debug:
                            print(f"        Next token: {next_token.text} (POS={next_token.pos_})")
                        if next_token.pos_ in ["NOUN", "PROPN"]:
                            product_parts.append(next_token.text)
                        elif next_token.pos_ == "NUM" and j == i+1:  # e.g., "FERREX 20V"
                            product_parts.append(next_token.text)
                        elif next_token.pos_ in ["ADJ"] and len(product_parts) < 3:
                            # Allow adjectives in product name
                            product_parts.append(next_token.text)
                        elif next_token.text in ['-', 'und']:  # Compound names
                            product_parts.append(next_token.text)
                        else:
                            # Stop at punctuation or non-product words
                            break
                    
                    if debug:
                        print(f"        Built product_parts: {product_parts}")
                    
                    if len(product_parts) > 1:
                        product_name = " ".join(product_parts)
                        # HIGHEST PRIORITY: BRAND + NOUN in first sentence gets best score
                        # This should beat everything else
                        score = 18.0 + position_bonus if is_brand else 12.0 + position_bonus
                        candidates.append((product_name, score, f"brand_noun_sent{sent_idx}"))
                        if debug:
                            print(f"        Added candidate: {product_name} (score={score})")
            
            # Extract entities from this sentence
            for ent in sent.ents:
                if ent.label_ in ["PRODUCT", "ORG"]:
                    candidates.append((ent.text, 5.0 + position_bonus, f"entity_sent{sent_idx}"))
            
            # Extract noun chunks from first sentence
            for chunk in sent.noun_chunks:
                # Prefer chunks with proper nouns or brands
                has_propn = any(t.pos_ == "PROPN" for t in chunk)
                has_brand = any(t.text.upper() in self.known_brands for t in chunk)
                
                if has_brand:
                    candidates.append((chunk.text, 6.0 + position_bonus, f"brand_chunk_sent{sent_idx}"))
                elif has_propn and sent_idx == 0:
                    candidates.append((chunk.text, 4.0 + position_bonus, f"propn_chunk_sent{sent_idx}"))
        
        # Strategy 1: Named entities (PRODUCT, ORG, MISC) from full text
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG", "MISC"]:
                candidates.append((ent.text, 4.0, "entity"))
        
        # Strategy 2: Proper nouns and their context
        for token in doc:
            if token.pos_ == "PROPN":
                # Get the token and its right neighbors (compound names)
                span_tokens = [token]
                for right in token.rights:
                    if right.pos_ in ["PROPN", "NOUN"]:
                        span_tokens.append(right)
                span_text = " ".join([t.text for t in span_tokens])
                candidates.append((span_text, 3.5, "propn"))
        
        # Strategy 3: Brand-adjacent nouns (e.g., "WORKZONE Arbeitsleuchte")
        for i, token in enumerate(doc):
            if token.text.upper() in self.known_brands or token.pos_ == "PROPN":
                # Get surrounding nouns to form product name
                context = []
                if i > 0 and doc[i-1].pos_ in ["NOUN", "ADJ"]:
                    context.insert(0, doc[i-1].text)
                context.append(token.text)
                if i < len(doc) - 1 and doc[i+1].pos_ in ["NOUN", "ADJ"]:
                    context.append(doc[i+1].text)
                if i < len(doc) - 2 and doc[i+2].pos_ == "NOUN":
                    context.append(doc[i+2].text)
                if len(context) > 1:
                    candidates.append((" ".join(context), 5.5, "brand_context"))
        
        # Score and rank candidates
        if not candidates:
            if debug:
                print(f"    DEBUG: No candidates found!")
            return ""
        
        # Add bonuses for specific features
        scored_candidates = []
        for text, base_score, source in candidates:
            # Filter out non-product words
            text_upper = text.upper()
            if any(non_prod in text_upper for non_prod in self.non_product_words):
                continue
            
            # Remove trailing attribute words from candidate
            # e.g., "BARISSIMO UNSER BESTER CHARAKTERVOLL" -> "BARISSIMO UNSER BESTER"
            words = text.split()
            while words and words[-1].upper() in self.attribute_keywords:
                words.pop()
            if words:
                text = " ".join(words)
            else:
                continue
            
            # Filter out generic parts
            if text_upper in ["ALUMINIUM-KLINGE", "KLINGE", "STECKDOSE", "SCHALTER"]:
                continue
            
            score = base_score
            
            # Bonus for length (good product names are 8-50 chars)
            if 10 <= len(text) <= 50:
                score += 3.0
            elif 6 <= len(text) <= 60:
                score += 1.0
            
            # Strong bonus for containing known brands
            if any(brand in text_upper for brand in self.known_brands):
                score += 5.0
            
            # Strong bonus for containing product type keywords
            if any(pt.lower() in text.upper() for pt in self.product_type_keywords):
                score += 8.0  # VERY HIGH BONUS for product type
                if debug:
                    print(f"        Product type bonus for: {text}")
            
            # Penalty for very short text
            if len(text) < 5:
                score -= 3.0
            
            # Penalty for digits (product names rarely have many numbers)
            digit_ratio = sum(c.isdigit() for c in text) / max(len(text), 1)
            if digit_ratio > 0.3:
                score -= 5.0
            
            # Penalty for too many special characters
            special_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / max(len(text), 1)
            if special_ratio > 0.25:
                score -= 3.0
            
            # Require minimum score threshold
            if score < 3.0:
                continue
            
            scored_candidates.append((text, score, source))
            
            # Penalty for digits (product names rarely have many numbers)
            digit_ratio = sum(c.isdigit() for c in text) / len(text)
            if digit_ratio > 0.3:
                score -= 3.0
            
            scored_candidates.append((text, score, source))
        
        if debug:
            print(f"    DEBUG: {len(scored_candidates)} candidates after scoring")
            for text, score, source in scored_candidates[:5]:
                print(f"      - {text[:50]} (score={score:.1f}, source={source})")
        
        # Sort by score and return best
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        if not scored_candidates:
            if debug:
                print(f"    DEBUG: No scored candidates passed threshold!")
            return ""
        
        best_text = scored_candidates[0][0]
        
        # Clean up the result
        return self._clean_product_name(best_text)
    
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


def get_extractor(model_name: str = "de_core_news_sm") -> ProductNERExtractor:
    """Get or create the global NER extractor instance."""
    global _extractor
    if _extractor is None:
        _extractor = ProductNERExtractor(model_name)
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
    return extractor.extract_entities(text)
