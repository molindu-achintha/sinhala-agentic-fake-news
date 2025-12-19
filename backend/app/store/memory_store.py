"""
memory_store.py

Memory management for the verification system.
Long term memory: PostgreSQL (persistent claims and verdicts)
Short term memory: Redis (fast cache for recent queries)
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, Optional, List

# Redis for short term cache
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("[MemoryStore] Redis not installed, using in memory cache")

# PostgreSQL for long term storage
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
    POSTGRES_AVAILABLE = True
except ImportError:
    POSTGRES_AVAILABLE = False
    print("[MemoryStore] PostgreSQL not installed, using in memory storage")


class ShortTermMemory:
    """
    Redis based short term cache.
    Stores recent queries and results for fast retrieval.
    Default TTL: 1 hour
    """
    
    def __init__(self):
        """Initialize Redis connection."""
        self.redis_url = os.getenv("REDIS_URL")
        self.redis_host = os.getenv("REDIS_HOST", "localhost")
        self.redis_port = int(os.getenv("REDIS_PORT", "6379"))
        self.redis_password = os.getenv("REDIS_PASSWORD", "")
        self.ttl = 3600  # 1 hour default
        self.client = None
        self.fallback_cache = {}  # In memory fallback
        
        if REDIS_AVAILABLE:
            try:
                # Try URL first
                if self.redis_url:
                    self.client = redis.from_url(self.redis_url, decode_responses=True)
                else:
                    # Use host/port/password
                    self.client = redis.Redis(
                        host=self.redis_host,
                        port=self.redis_port,
                        password=self.redis_password if self.redis_password else None,
                        decode_responses=True
                    )
                self.client.ping()
                print("[ShortTermMemory] Redis connected")
            except Exception as e:
                print("[ShortTermMemory] Redis connection failed:", str(e))
                self.client = None
        else:
            print("[ShortTermMemory] Using in memory fallback")
    
    def _get_key(self, claim: str) -> str:
        """Generate cache key from claim text."""
        return "claim:" + hashlib.md5(claim.encode()).hexdigest()
    
    def get(self, claim: str) -> Optional[Dict]:
        """Get cached result for a claim."""
        key = self._get_key(claim)
        
        if self.client:
            try:
                data = self.client.get(key)
                if data:
                    print("[ShortTermMemory] Cache hit for claim")
                    return json.loads(data)
            except Exception as e:
                print("[ShortTermMemory] Redis get error:", str(e))
        else:
            if key in self.fallback_cache:
                print("[ShortTermMemory] Fallback cache hit")
                return self.fallback_cache[key]
        
        return None
    
    def set(self, claim: str, result: Dict, ttl: int = None) -> bool:
        """Cache a verification result."""
        key = self._get_key(claim)
        ttl = ttl or self.ttl
        
        if self.client:
            try:
                self.client.setex(key, ttl, json.dumps(result, default=str))
                print("[ShortTermMemory] Cached result, TTL:", ttl)
                return True
            except Exception as e:
                print("[ShortTermMemory] Redis set error:", str(e))
        else:
            self.fallback_cache[key] = result
            print("[ShortTermMemory] Stored in fallback cache")
            return True
        
        return False
    
    def delete(self, claim: str) -> bool:
        """Delete cached result."""
        key = self._get_key(claim)
        
        if self.client:
            try:
                self.client.delete(key)
                return True
            except:
                pass
        else:
            if key in self.fallback_cache:
                del self.fallback_cache[key]
                return True
        
        return False
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text."""
        key = "emb:" + hashlib.md5(text.encode()).hexdigest()
        
        if self.client:
            try:
                data = self.client.get(key)
                if data:
                    return json.loads(data)
            except:
                pass
        
        return None
    
    def set_embedding(self, text: str, embedding: List[float]) -> bool:
        """Cache embedding for text."""
        key = "emb:" + hashlib.md5(text.encode()).hexdigest()
        ttl = 86400  # 24 hours for embeddings
        
        if self.client:
            try:
                self.client.setex(key, ttl, json.dumps(embedding))
                return True
            except:
                pass
        
        return False


class LongTermMemory:
    """
    PostgreSQL based long term memory.
    Stores verified claims, verdicts, and user feedback.
    """
    
    CREATE_TABLE_SQL = '''
    CREATE TABLE IF NOT EXISTS verified_claims (
        id SERIAL PRIMARY KEY,
        claim_hash VARCHAR(64) UNIQUE NOT NULL,
        claim_text TEXT NOT NULL,
        verdict VARCHAR(32) NOT NULL,
        confidence FLOAT,
        reasoning TEXT,
        evidence JSONB,
        source VARCHAR(128),
        verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        access_count INTEGER DEFAULT 1
    );
    
    CREATE INDEX IF NOT EXISTS idx_claim_hash ON verified_claims(claim_hash);
    CREATE INDEX IF NOT EXISTS idx_verdict ON verified_claims(verdict);
    CREATE INDEX IF NOT EXISTS idx_verified_at ON verified_claims(verified_at);
    '''
    
    def __init__(self):
        """Initialize PostgreSQL connection."""
        self.database_url = os.getenv("DATABASE_URL", "postgresql://localhost:5432/fakenews")
        self.conn = None
        self.fallback_storage = {}  # In memory fallback
        
        if POSTGRES_AVAILABLE:
            self._connect()
        else:
            print("[LongTermMemory] Using in memory fallback")
    
    def _connect(self):
        """Establish PostgreSQL connection."""
        try:
            self.conn = psycopg2.connect(self.database_url)
            self._create_tables()
            print("[LongTermMemory] PostgreSQL connected")
        except Exception as e:
            print("[LongTermMemory] PostgreSQL connection failed:", str(e))
            self.conn = None
    
    def _ensure_connection(self):
        """Ensure connection is alive, reconnect if closed."""
        if not POSTGRES_AVAILABLE:
            return
            
        if self.conn is None or self.conn.closed != 0:
            print("[LongTermMemory] Connection lost, reconnecting...")
            self._connect()
    
    def _create_tables(self):
        """Create tables if not exist."""
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute(self.CREATE_TABLE_SQL)
                    self.conn.commit()
            except Exception as e:
                print("[LongTermMemory] Table creation error:", str(e))
    
    def _get_hash(self, claim: str) -> str:
        """Generate hash from claim text."""
        return hashlib.sha256(claim.encode()).hexdigest()
    
    def get(self, claim: str) -> Optional[Dict]:
        """Get stored verification result for a claim."""
        claim_hash = self._get_hash(claim)
        
        self._ensure_connection()
        if self.conn:
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('''
                        UPDATE verified_claims 
                        SET access_count = access_count + 1, updated_at = NOW()
                        WHERE claim_hash = %s
                        RETURNING *
                    ''', (claim_hash,))
                    self.conn.commit()
                    row = cur.fetchone()
                    if row:
                        print("[LongTermMemory] Found stored claim")
                        return dict(row)
            except Exception as e:
                print("[LongTermMemory] Get error:", str(e))
        else:
            if claim_hash in self.fallback_storage:
                return self.fallback_storage[claim_hash]
        
        return None
    
    def store(self, claim: str, result: Dict) -> bool:
        """Store a verification result."""
        claim_hash = self._get_hash(claim)
        
        self._ensure_connection()
        if self.conn:
            try:
                with self.conn.cursor() as cur:
                    cur.execute('''
                        INSERT INTO verified_claims 
                        (claim_hash, claim_text, verdict, confidence, reasoning, evidence, source)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (claim_hash) DO UPDATE SET
                            verdict = EXCLUDED.verdict,
                            confidence = EXCLUDED.confidence,
                            reasoning = EXCLUDED.reasoning,
                            evidence = EXCLUDED.evidence,
                            updated_at = NOW(),
                            access_count = verified_claims.access_count + 1
                    ''', (
                        claim_hash,
                        claim,
                        result.get("verdict", {}).get("label", "unknown"),
                        result.get("verdict", {}).get("confidence", 0),
                        result.get("reasoning", {}).get("cot_reasoning", ""),
                        json.dumps(result.get("evidence", {})),
                        "hybrid_verifier"
                    ))
                    self.conn.commit()
                    print("[LongTermMemory] Stored verification result")
                    return True
            except Exception as e:
                print("[LongTermMemory] Store error:", str(e))
        else:
            self.fallback_storage[claim_hash] = {
                "claim_text": claim,
                "verdict": result.get("verdict", {}).get("label", "unknown"),
                "confidence": result.get("verdict", {}).get("confidence", 0),
                "verified_at": datetime.now().isoformat()
            }
            return True
        
        return False
    
    def get_similar_verdicts(self, verdict: str, limit: int = 10) -> List[Dict]:
        """Get recent claims with same verdict."""
        self._ensure_connection()
        if self.conn:
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('''
                        SELECT claim_text, verdict, confidence, verified_at
                        FROM verified_claims
                        WHERE verdict = %s
                        ORDER BY verified_at DESC
                        LIMIT %s
                    ''', (verdict, limit))
                    return [dict(row) for row in cur.fetchall()]
            except:
                pass
        return []
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        self._ensure_connection()
        if self.conn:
            try:
                with self.conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute('''
                        SELECT 
                            COUNT(*) as total_claims,
                            COUNT(DISTINCT verdict) as verdict_types,
                            AVG(confidence) as avg_confidence,
                            MAX(verified_at) as last_verified
                        FROM verified_claims
                    ''')
                    return dict(cur.fetchone())
            except:
                pass
        
        return {
            "total_claims": len(self.fallback_storage),
            "verdict_types": 0,
            "storage": "in_memory"
        }


class MemoryManager:
    """
    Unified memory manager.
    Coordinates short term (Redis) and long term (PostgreSQL) memory.
    """
    
    def __init__(self):
        """Initialize both memory stores."""
        print("[MemoryManager] Initializing memory stores")
        self.short_term = ShortTermMemory()
        self.long_term = LongTermMemory()
        print("[MemoryManager] Ready")
    
    def get_cached_result(self, claim: str) -> Optional[Dict]:
        """
        Check both caches for existing result.
        Returns cached result if found, None otherwise.
        """
        # Check short term first (faster)
        result = self.short_term.get(claim)
        if result:
            print("[MemoryManager] Found in short term cache")
            return result
        
        # Check long term
        result = self.long_term.get(claim)
        if result:
            print("[MemoryManager] Found in long term memory")
            # Populate short term cache
            self.short_term.set(claim, result)
            return result
        
        return None
    
    def store_result(self, claim: str, result: Dict):
        """Store result in both caches."""
        # Store in short term (fast access)
        self.short_term.set(claim, result)
        
        # Store in long term (persistent)
        self.long_term.store(claim, result)
        
        print("[MemoryManager] Result stored in both memories")
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding."""
        return self.short_term.get_embedding(text)
    
    def cache_embedding(self, text: str, embedding: List[float]):
        """Cache embedding for reuse."""
        self.short_term.set_embedding(text, embedding)
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "long_term": self.long_term.get_stats(),
            "short_term": {
                "type": "redis" if self.short_term.client else "in_memory"
            }
        }


# Singleton instance
_memory_manager = None

def get_memory_manager() -> MemoryManager:
    """Get or create MemoryManager instance."""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager
