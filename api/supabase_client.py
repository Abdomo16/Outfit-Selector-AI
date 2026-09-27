import os
from typing import Optional

from supabase import Client, create_client


_supabase: Optional[Client] = None


def get_supabase() -> Client:
    global _supabase
    if _supabase is None:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_SERVICE_KEY")
        if not url or not key:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in the environment"
            )
        _supabase = create_client(url, key)
    return _supabase


def reset_supabase() -> None:
    global _supabase
    _supabase = None
