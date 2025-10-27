"""
STRYK MIGRATION STEP 1: EXPORT OLD DATA
========================================

Exports all data from your current backend to migration_data.json

Usage:
    python 01_export_old_data.py
"""

import httpx
import json
import sys
from datetime import datetime
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================

OLD_BACKEND_URL = "http://localhost:8001"  # Change to your backend URL

ENDPOINTS = {
    "live_games": "/api/live-games",
    "opportunities": "/api/opportunities",
    "mamba_performance": "/api/mamba-performance",
    "mamba_scores": "/api/mamba-scores",
    # Add any other endpoints you want to export
}

OUTPUT_DIR = Path("../export")
OUTPUT_FILE = OUTPUT_DIR / "migration_data.json"

# =====================================================
# EXPORT FUNCTIONS
# =====================================================

def fetch_endpoint(client: httpx.Client, endpoint: str) -> dict:
    """Fetch data from an endpoint"""
    try:
        url = f"{OLD_BACKEND_URL}{endpoint}"
        print(f"📡 Fetching {url}...")
        response = client.get(url, timeout=30.0)
        response.raise_for_status()
        data = response.json()
        print(f"✅ Got {len(data) if isinstance(data, list) else 1} items")
        return data
    except Exception as e:
        print(f"❌ Error fetching {endpoint}: {e}")
        return None

def export_all_data():
    """Export all data from old backend"""
    print("=" * 60)
    print("🚀 STRYK MIGRATION: EXPORTING OLD DATA")
    print("=" * 60)
    
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Create HTTP client
    client = httpx.Client()
    
    # Export data from all endpoints
    exported_data = {
        "export_timestamp": datetime.now().isoformat(),
        "source_backend": OLD_BACKEND_URL,
        "data": {}
    }
    
    for name, endpoint in ENDPOINTS.items():
        print(f"\n📦 Exporting {name}...")
        data = fetch_endpoint(client, endpoint)
        if data is not None:
            exported_data["data"][name] = data
        else:
            print(f"⚠️  Skipping {name} (error)")
    
    # Close client
    client.close()
    
    # Save to JSON
    print(f"\n💾 Saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        json.dump(exported_data, f, indent=2)
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ EXPORT COMPLETE")
    print("=" * 60)
    print(f"📁 File: {OUTPUT_FILE}")
    print(f"📊 Size: {OUTPUT_FILE.stat().st_size / 1024:.2f} KB")
    print(f"🎯 Endpoints exported: {len(exported_data['data'])}")
    
    for name, data in exported_data["data"].items():
        count = len(data) if isinstance(data, list) else "N/A"
        print(f"   - {name}: {count} items")
    
    print("\n🎯 Next step: Run 02_migrate_to_stryk.py")
    print("=" * 60)

# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    try:
        export_all_data()
    except KeyboardInterrupt:
        print("\n\n⚠️  Export interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

