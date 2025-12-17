"""
wikidata_client.py

Wikidata SPARQL Client for structured fact verification.
Queries Wikidata knowledge graph for factual claims verification.
FREE API - no authentication required.
"""
import requests
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import re


class ClaimType(Enum):
    """Types of factual claims we can verify via Wikidata."""
    CAPITAL = "capital"
    CURRENCY = "currency"
    PRESIDENT = "president"
    LANGUAGE = "language"
    LOCATION = "location"
    LARGEST = "largest"
    LONGEST = "longest"
    HIGHEST = "highest"
    INDEPENDENCE_DAY = "independence_day"
    POPULATION = "population"
    UNKNOWN = "unknown"


@dataclass
class WikidataResult:
    """Result from Wikidata query."""
    claim_type: ClaimType
    expected_value: str  # What the claim says
    actual_value: str    # What Wikidata says
    is_correct: bool
    confidence: float
    source_url: str
    evidence: str


class WikidataClient:
    """
    Client for verifying factual claims against Wikidata.
    
    Uses SPARQL to query structured knowledge for exact fact verification.
    This is critical for claims like "X is the capital of Y" where
    Wikipedia might just mention the topic but not verify the specific fact.
    """
    
    ENDPOINT = "https://query.wikidata.org/sparql"
    
    # Entity IDs for common countries (can be expanded)
    COUNTRY_ENTITIES = {
        "sri lanka": "Q854",
        "india": "Q668",
        "usa": "Q30",
        "united states": "Q30",
        "america": "Q30",
        "china": "Q148",
        "japan": "Q17",
        "australia": "Q408",
        "uk": "Q145",
        "united kingdom": "Q145",
        "nepal": "Q837",
        "pakistan": "Q843",
        "bangladesh": "Q902"
    }
    
    # Claim type detection patterns
    CLAIM_PATTERNS = {
        ClaimType.CAPITAL: [
            r"(?:capital|අගනුවර|राजधानी)",
            r"(?:capital city|administrative capital|legislative capital)"
        ],
        ClaimType.CURRENCY: [
            r"(?:currency|මුදල්|மुद்ரை|rupee|dollar|euro|yen)"
        ],
        ClaimType.PRESIDENT: [
            r"(?:president|ජනාධිපති|prime minister|leader)"
        ],
        ClaimType.LANGUAGE: [
            r"(?:language|භාෂා|official language|national language)"
        ],
        ClaimType.LOCATION: [
            r"(?:located|located in|situated|පිහිටා|lies in)"
        ],
        ClaimType.LARGEST: [
            r"(?:largest|biggest|විශාලතම)"
        ],
        ClaimType.LONGEST: [
            r"(?:longest|දිගම)"
        ],
        ClaimType.INDEPENDENCE_DAY: [
            r"(?:independence|නිදහස්|national day)"
        ]
    }
    
    # SPARQL query templates
    SPARQL_TEMPLATES = {
        ClaimType.CAPITAL: """
            SELECT ?capitalLabel WHERE {{
                wd:{entity} wdt:P36 ?capital.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,si". }}
            }}
        """,
        ClaimType.CURRENCY: """
            SELECT ?currencyLabel WHERE {{
                wd:{entity} wdt:P38 ?currency.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,si". }}
            }}
        """,
        ClaimType.PRESIDENT: """
            SELECT ?headLabel WHERE {{
                wd:{entity} wdt:P35 ?head.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,si". }}
            }}
        """,
        ClaimType.LANGUAGE: """
            SELECT ?languageLabel WHERE {{
                wd:{entity} wdt:P37 ?language.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,si". }}
            }}
        """,
        ClaimType.LOCATION: """
            SELECT ?continentLabel WHERE {{
                wd:{entity} wdt:P30 ?continent.
                SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
            }}
        """,
        ClaimType.INDEPENDENCE_DAY: """
            SELECT ?date WHERE {{
                wd:{entity} wdt:P31 wd:Q6256.
                wd:{entity} wdt:P571 ?date.
            }}
        """
    }
    
    def __init__(self):
        """Initialize the Wikidata client."""
        self.headers = {
            "Accept": "application/sparql-results+json",
            "User-Agent": "SinhalaFakeNewsDetector/1.0"
        }
        print("[WikidataClient] Initialized")
    
    def verify_claim(
        self, 
        claim: str, 
        translated_claim: str
    ) -> Optional[WikidataResult]:
        """
        Verify a factual claim against Wikidata.
        
        Args:
            claim: Original Sinhala claim
            translated_claim: English translation
            
        Returns:
            WikidataResult if verifiable, None otherwise
        """
        print(f"[WikidataClient] Verifying: {translated_claim[:60]}...")
        
        # 1. Detect claim type
        claim_type = self._detect_claim_type(translated_claim)
        if claim_type == ClaimType.UNKNOWN:
            print("[WikidataClient] Claim type not recognized")
            return None
        
        print(f"[WikidataClient] Claim type: {claim_type.value}")
        
        # 2. Extract entity from claim
        entity = self._extract_entity(translated_claim)
        if not entity:
            print("[WikidataClient] No entity found")
            return None
        
        print(f"[WikidataClient] Entity: {entity}")
        
        # 3. Query Wikidata
        actual_value = self._query_wikidata(claim_type, entity)
        if not actual_value:
            print("[WikidataClient] No Wikidata result")
            return None
        
        print(f"[WikidataClient] Wikidata says: {actual_value}")
        
        # 4. Extract claimed value
        claimed_value = self._extract_claimed_value(translated_claim, claim_type)
        print(f"[WikidataClient] Claim says: {claimed_value}")
        
        # 5. Compare
        is_correct = self._compare_values(claimed_value, actual_value, claim_type)
        
        return WikidataResult(
            claim_type=claim_type,
            expected_value=claimed_value,
            actual_value=actual_value,
            is_correct=is_correct,
            confidence=0.95 if is_correct else 0.90,  # High confidence from structured data
            source_url=f"https://www.wikidata.org/wiki/{self.COUNTRY_ENTITIES.get(entity.lower(), '')}",
            evidence=f"According to Wikidata: {actual_value}"
        )
    
    def _detect_claim_type(self, claim: str) -> ClaimType:
        """Detect the type of factual claim."""
        claim_lower = claim.lower()
        
        for claim_type, patterns in self.CLAIM_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, claim_lower, re.IGNORECASE):
                    return claim_type
        
        return ClaimType.UNKNOWN
    
    # Additional entity classes for generic queries
    ENTITY_CLASSES = {
        "river": "Q4022",
        "animal": "Q729",
        "country": "Q6256",
        "city": "Q515",
        "mountain": "Q8502",
        "planet": "Q634",
        "star": "Q523",
        "building": "Q41176",
        "island": "Q23442"
    }

    def _extract_entity(self, claim: str) -> Optional[str]:
        """Extract the main entity from the claim."""
        claim_lower = claim.lower()
        
        # 1. Check for specific countries first (Highest priority)
        for country, entity_id in self.COUNTRY_ENTITIES.items():
            if country in claim_lower:
                return country
                
        # 2. Check for generic entity classes (for 'largest', 'longest' queries)
        for entity_class, entity_id in self.ENTITY_CLASSES.items():
            if entity_class in claim_lower:
                return entity_id  # Return ID directly for classes
                
        return None
    
    def _query_wikidata(self, claim_type: ClaimType, entity: str) -> Optional[str]:
        """Query Wikidata SPARQL endpoint."""
        if claim_type not in self.SPARQL_TEMPLATES:
            return None
        
        # Determine Entity ID
        # If it's a known country name, get ID. If it looks like Q-ID, use as is.
        entity_id = self.COUNTRY_ENTITIES.get(entity.lower())
        if not entity_id and entity.startswith("Q"):
            entity_id = entity
            
        if not entity_id:
            return None
        
        query = self.SPARQL_TEMPLATES[claim_type].format(entity=entity_id)
        
        try:
            response = requests.get(
                self.ENDPOINT,
                params={"query": query},
                headers=self.headers,
                timeout=15
            )
            
            if response.status_code != 200:
                print(f"[WikidataClient] Query failed: {response.status_code}")
                return None
            
            data = response.json()
            bindings = data.get("results", {}).get("bindings", [])
            
            if not bindings:
                return None
            
            # Get all values (for languages, there might be multiple)
            values = []
            for binding in bindings:
                for key in binding:
                    if "Label" in key or key == "date":
                        values.append(binding[key]["value"])
            
            return ", ".join(values) if values else None
            
        except Exception as e:
            print(f"[WikidataClient] Query error: {e}")
            return None
    
    def _extract_claimed_value(self, claim: str, claim_type: ClaimType) -> str:
        """Extract what the claim is asserting."""
        claim_lower = claim.lower()
        
        # Pattern matching for different claim types
        if claim_type == ClaimType.CAPITAL:
            # Look for "X is the capital" pattern
            patterns = [
                r"capital.*?(?:is|वन्नेे)\s+(.+?)(?:\.|$)",
                r"(.+?)\s+(?:is|are)\s+(?:the\s+)?capital",
                r"capital.*?(?:ශ්‍රී ජයවර්ධනපුර කෝට්ටේ|colombo|kotte)"
            ]
            for pattern in patterns:
                match = re.search(pattern, claim_lower)
                if match:
                    return match.group(1).strip() if match.lastindex else claim
        
        elif claim_type == ClaimType.CURRENCY:
            # Look for currency mentions
            currencies = ["rupee", "dollar", "euro", "yen", "yuan", "rupees"]
            for curr in currencies:
                if curr in claim_lower:
                    return curr
        
        elif claim_type == ClaimType.LARGEST:
            # "largest X is Y"
            match = re.search(r"largest.*?(?:is|are)\s+(?:the\s+)?(.+?)(?:\.|$)", claim_lower)
            if match:
                return match.group(1).strip()
        
        # Default - return cleaned claim
        return claim
    
    def _compare_values(self, claimed: str, actual: str, claim_type: ClaimType) -> bool:
        """Compare claimed value with actual Wikidata value."""
        claimed_lower = claimed.lower().strip()
        actual_lower = actual.lower().strip()
        
        # Direct match
        if claimed_lower in actual_lower or actual_lower in claimed_lower:
            return True
        
        # Capital-specific matching
        if claim_type == ClaimType.CAPITAL:
            # Sri Lanka capital variations
            kotte_variants = ["kotte", "jayawardenepura", "sri jayawardenepura"]
            colombo_variants = ["colombo"]
            
            actual_is_kotte = any(v in actual_lower for v in kotte_variants)
            claimed_is_kotte = any(v in claimed_lower for v in kotte_variants)
            claimed_is_colombo = any(v in claimed_lower for v in colombo_variants)
            
            if actual_is_kotte and claimed_is_kotte:
                return True
            if actual_is_kotte and claimed_is_colombo:
                return False  # Colombo is NOT the capital
        
        # Currency matching
        if claim_type == ClaimType.CURRENCY:
            rupee_variants = ["rupee", "rupees", "lkr", "sri lankan rupee"]
            euro_variants = ["euro", "eur"]
            
            if any(v in actual_lower for v in rupee_variants):
                if any(v in claimed_lower for v in euro_variants):
                    return False  # Claiming Euro when it's Rupee
                if any(v in claimed_lower for v in rupee_variants):
                    return True
        
        # Language matching
        if claim_type == ClaimType.LANGUAGE:
            # Check if claimed languages are in actual
            claimed_langs = set(re.findall(r'\b(?:sinhala|tamil|english|sinhalese)\b', claimed_lower))
            actual_langs = set(re.findall(r'\b(?:sinhala|tamil|english|sinhalese)\b', actual_lower))
            
            if claimed_langs and claimed_langs.issubset(actual_langs):
                return True
        
        # Fuzzy match - check word overlap
        claimed_words = set(claimed_lower.split())
        actual_words = set(actual_lower.split())
        overlap = len(claimed_words & actual_words)
        
        return overlap >= min(2, len(claimed_words) * 0.5)


# Singleton instance
_client = None

def get_wikidata_client() -> WikidataClient:
    """Get or create Wikidata client instance."""
    global _client
    if _client is None:
        _client = WikidataClient()
    return _client
