"""
Enhanced NLP Utilities for Sinhala Text Processing

This module provides advanced NLP capabilities for Sinhala language:
- POS Tagging (Part-of-Speech)
- Named Entity Recognition (NER)
- Stemming / Lemmatization
- Word Disambiguation
- Morphological Analysis
"""

import re
from typing import List, Dict, Tuple, Optional

# Try to import sinling components
try:
    from sinling import SinhalaTokenizer, POSTagger, SinhalaStemmer
    SINLING_AVAILABLE = True
except ImportError:
    SINLING_AVAILABLE = False
    print("Warning: sinling library not available. Using fallback methods.")


class SinhalaNLP:
    """
    Comprehensive Sinhala NLP processor with POS tagging, NER, and more.
    """
    
    def __init__(self):
        if SINLING_AVAILABLE:
            self.tokenizer = SinhalaTokenizer()
            self.pos_tagger = POSTagger()
            self.stemmer = SinhalaStemmer()
        else:
            self.tokenizer = None
            self.pos_tagger = None
            self.stemmer = None
        
        # Named Entity patterns for Sinhala (Rule-based)
        self.entity_patterns = {
            'PERSON': [
                r'මහතා', r'මහත්මිය', r'ජනාධිපති', r'අගමැති', r'මැති', 
                r'ඇමති', r'හිටපු', r'ආචාර්ය', r'මහාචාර්ය', r'මහතුන්'
            ],
            'LOCATION': [
                r'ප්‍රදේශය', r'නගරය', r'දිස්ත්‍රික්ක', r'පළාත', r'රට',
                r'කොළඹ', r'ගම්පහ', r'කළුතර', r'මහනුවර', r'ගාල්ල',
                r'මාතර', r'යාපනය', r'බත්තරමුල්ල', r'ශ්‍රී ලංකා'
            ],
            'ORGANIZATION': [
                r'සංවිධානය', r'සමාගම', r'බැංකුව', r'විශ්ව විද්‍යාලය',
                r'රජය', r'අමාත්‍යාංශය', r'දෙපාර්තමේන්තුව', r'පොලිසිය'
            ],
            'DATE': [
                r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
                r'\d{1,2}[-/]\d{1,2}[-/]\d{4}',
                r'ජනවාරි|පෙබරවාරි|මාර්තු|අප්‍රේල්|මැයි|ජූනි|ජූලි|අගෝස්තු|සැප්තැම්බර්|ඔක්තෝබර්|නොවැම්බර්|දෙසැම්බර්'
            ],
            'NUMBER': [
                r'\d+(?:\.\d+)?%?',
                r'මිලියන|බිලියන|දහස්|ලක්ෂ|කෝටි'
            ]
        }
        
        # Claim indicator keywords (for fact-check relevance)
        self.claim_indicators = [
            'ප්‍රකාශ කළේය', 'සඳහන් කළේය', 'පවසයි', 'කියා', 'අනුව',
            'වාර්තා', 'අනාවරණය', 'තහවුරු', 'ප්‍රකාශයට', 'හෙළි'
        ]
        
        # Negation words
        self.negation_words = [
            'නැත', 'නොවේ', 'නෑ', 'එපා', 'බැහැ', 'නොමැත'
        ]
        
    def tokenize(self, text: str) -> List[str]:
        """Tokenize Sinhala text into words."""
        if self.tokenizer:
            return self.tokenizer.tokenize(text)
        else:
            # Fallback: Simple regex-based tokenization
            return re.findall(r'[\u0D80-\u0DFF]+|[a-zA-Z]+|\d+', text)
    
    def pos_tag(self, text: str) -> List[Tuple[str, str]]:
        """
        Perform Part-of-Speech tagging on Sinhala text.
        Returns list of (word, tag) tuples.
        
        POS Tags:
        - NN: Noun
        - VB: Verb
        - JJ: Adjective
        - RB: Adverb
        - PP: Postposition
        - CC: Conjunction
        - PRP: Pronoun
        - CD: Cardinal number
        - NNP: Proper noun
        """
        if self.pos_tagger:
            tokens = self.tokenize(text)
            return self.pos_tagger.predict(tokens)
        else:
            # Fallback: Simple rule-based tagging
            tokens = self.tokenize(text)
            tagged = []
            for token in tokens:
                tag = self._rule_based_pos(token)
                tagged.append((token, tag))
            return tagged
    
    def _rule_based_pos(self, word: str) -> str:
        """Simple rule-based POS fallback."""
        # Numbers
        if re.match(r'\d+', word):
            return 'CD'
        # Common verb endings in Sinhala
        if word.endswith(('යි', 'නවා', 'මින්', 'ලා')):
            return 'VB'
        # Common adjective endings
        if word.endswith(('ම', 'ක්', 'ය')):
            return 'JJ'
        # Default to noun
        return 'NN'
    
    def stem(self, word: str) -> str:
        """Get the stem/root form of a Sinhala word."""
        if self.stemmer:
            return self.stemmer.stem(word)
        else:
            # Fallback: Remove common suffixes
            suffixes = ['ය', 'ක්', 'ලා', 'නවා', 'යි', 'වල', 'ගේ', 'ට', 'න්']
            for suffix in suffixes:
                if word.endswith(suffix) and len(word) > len(suffix) + 2:
                    return word[:-len(suffix)]
            return word
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from Sinhala text.
        Returns dict with entity types as keys and lists of entities as values.
        """
        entities = {etype: [] for etype in self.entity_patterns.keys()}
        
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text)
                entities[entity_type].extend(matches)
        
        # Additional: Look for capitalized words (likely proper nouns)
        # In mixed Sinhala-English text
        english_proper = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        entities['PERSON'].extend(english_proper)
        
        # Deduplicate
        for entity_type in entities:
            entities[entity_type] = list(set(entities[entity_type]))
        
        return entities
    
    def detect_claim_indicators(self, text: str) -> List[str]:
        """Find phrases that indicate factual claims."""
        found = []
        for indicator in self.claim_indicators:
            if indicator in text:
                found.append(indicator)
        return found
    
    def detect_negation(self, text: str) -> bool:
        """Check if text contains negation."""
        for neg in self.negation_words:
            if neg in text:
                return True
        return False
    
    def analyze_sentence(self, sentence: str) -> Dict:
        """
        Comprehensive analysis of a single sentence.
        Returns structured information about the sentence.
        """
        tokens = self.tokenize(sentence)
        pos_tags = self.pos_tag(sentence)
        entities = self.extract_entities(sentence)
        
        # Count POS categories
        pos_counts = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        
        # Extract nouns and verbs
        nouns = [word for word, tag in pos_tags if tag in ['NN', 'NNP']]
        verbs = [word for word, tag in pos_tags if tag == 'VB']
        
        # Stem important words
        stemmed_nouns = [self.stem(n) for n in nouns]
        
        return {
            'tokens': tokens,
            'token_count': len(tokens),
            'pos_tags': pos_tags,
            'pos_distribution': pos_counts,
            'entities': entities,
            'nouns': nouns,
            'verbs': verbs,
            'stemmed_nouns': stemmed_nouns,
            'has_claim_indicator': len(self.detect_claim_indicators(sentence)) > 0,
            'has_negation': self.detect_negation(sentence),
            'claim_indicators': self.detect_claim_indicators(sentence)
        }
    
    def process_document(self, text: str) -> Dict:
        """
        Process a full document with comprehensive NLP analysis.
        """
        # Split into sentences
        sentences = re.split(r'[.!?။]', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Analyze each sentence
        sentence_analyses = [self.analyze_sentence(s) for s in sentences]
        
        # Aggregate entities
        all_entities = {etype: [] for etype in self.entity_patterns.keys()}
        for analysis in sentence_analyses:
            for etype, ents in analysis['entities'].items():
                all_entities[etype].extend(ents)
        
        # Deduplicate
        for etype in all_entities:
            all_entities[etype] = list(set(all_entities[etype]))
        
        # Collect all nouns and verbs
        all_nouns = []
        all_verbs = []
        for analysis in sentence_analyses:
            all_nouns.extend(analysis['nouns'])
            all_verbs.extend(analysis['verbs'])
        
        # Count claim sentences
        claim_sentences = [s for s, a in zip(sentences, sentence_analyses) 
                          if a['has_claim_indicator']]
        
        return {
            'sentence_count': len(sentences),
            'sentences': sentences,
            'sentence_analyses': sentence_analyses,
            'entities': all_entities,
            'entity_count': sum(len(v) for v in all_entities.values()),
            'all_nouns': list(set(all_nouns)),
            'all_verbs': list(set(all_verbs)),
            'claim_sentences': claim_sentences,
            'has_claims': len(claim_sentences) > 0
        }


# Singleton instance for easy use
_nlp_instance = None

def get_sinhala_nlp() -> SinhalaNLP:
    """Get or create the Sinhala NLP processor instance."""
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = SinhalaNLP()
    return _nlp_instance


# Convenience functions
def pos_tag(text: str) -> List[Tuple[str, str]]:
    """POS tag Sinhala text."""
    return get_sinhala_nlp().pos_tag(text)

def extract_entities(text: str) -> Dict[str, List[str]]:
    """Extract named entities from Sinhala text."""
    return get_sinhala_nlp().extract_entities(text)

def stem_word(word: str) -> str:
    """Stem a Sinhala word."""
    return get_sinhala_nlp().stem(word)

def analyze_document(text: str) -> Dict:
    """Full NLP analysis of a document."""
    return get_sinhala_nlp().process_document(text)
