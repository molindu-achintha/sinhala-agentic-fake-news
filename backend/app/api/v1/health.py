"""
Health check endpoint with memory status.
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
def health_check():
    """
    Health check endpoint.
    Returns status of all services including memory stores.
    """
    print("[Health] Health check requested")
    
    # Check Redis
    redis_status = "unknown"
    try:
        from ...store.memory_store import get_memory_manager
        memory = get_memory_manager()
        if memory.short_term.client:
            memory.short_term.client.ping()
            redis_status = "connected"
        else:
            redis_status = "in_memory"
    except Exception as e:
        redis_status = "error: " + str(e)[:50]
    
    # Check PostgreSQL
    postgres_status = "unknown"
    try:
        if memory.long_term.conn:
            with memory.long_term.conn.cursor() as cur:
                cur.execute("SELECT 1")
            postgres_status = "connected"
        else:
            postgres_status = "in_memory"
    except Exception as e:
        postgres_status = "error: " + str(e)[:50]
    
    # Get memory stats
    memory_stats = None
    try:
        memory_stats = memory.get_stats()
    except:
        pass
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "redis": redis_status,
        "postgres": postgres_status,
        "memory_stats": memory_stats,
        "services": {
            "hybrid_verifier": "ready",
            "embedding_model": "multilingual-e5-large",
            "vector_db": "pinecone"
        }
    }


@router.get("/health/detailed")
def detailed_health():
    """
    Detailed health check with all component status.
    """
    print("[Health] Detailed health check requested")
    
    components = {
        "api": {"status": "ok", "message": "FastAPI running"},
        "memory": {"status": "unknown", "message": ""},
        "pinecone": {"status": "unknown", "message": ""},
        "openrouter": {"status": "unknown", "message": ""}
    }
    
    # Check memory
    try:
        from ...store.memory_store import get_memory_manager
        memory = get_memory_manager()
        stats = memory.get_stats()
        components["memory"] = {
            "status": "ok",
            "short_term": "redis" if memory.short_term.client else "in_memory",
            "long_term": "postgres" if memory.long_term.conn else "in_memory",
            "stats": stats
        }
    except Exception as e:
        components["memory"] = {"status": "error", "message": str(e)}
    
    # Check Pinecone
    try:
        from ...store.pinecone_store import get_pinecone_store
        store = get_pinecone_store()
        components["pinecone"] = {"status": "ok", "message": "Connected"}
    except Exception as e:
        components["pinecone"] = {"status": "error", "message": str(e)}
    
    # Overall status
    overall = "ok"
    for comp in components.values():
        if comp.get("status") == "error":
            overall = "degraded"
            break
    
    return {
        "status": overall,
        "timestamp": datetime.now().isoformat(),
        "components": components
    }
