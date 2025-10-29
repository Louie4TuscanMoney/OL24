"""
NBA TRANSACTIONS TRACKER
Populates player_transactions table with trades, waivers, signings

Data source: ESPN API (nba_api doesn't have transactions endpoint)
"""

import os
import psycopg2
import requests
from datetime import datetime, timedelta


def fetch_espn_transactions(days_back=30):
    """
    Fetch recent NBA transactions from ESPN
    
    Returns list of transactions with:
    - player_name
    - transaction_type
    - transaction_date
    - from_team
    - to_team
    - description
    """
    
    transactions = []
    
    try:
        # ESPN transactions endpoint
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/transactions"
        
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'items' in data:
            for item in data['items']:
                trans_date = item.get('date', '')
                description = item.get('description', '')
                
                # Parse transaction type from description
                trans_type = 'Trade'
                if 'waived' in description.lower():
                    trans_type = 'Waiver'
                elif 'signed' in description.lower():
                    trans_type = 'Signing'
                elif 'released' in description.lower():
                    trans_type = 'Release'
                elif 'two-way' in description.lower():
                    trans_type = 'Two-Way'
                
                # Extract player name (first name in description)
                player_name = description.split()[0] if description else 'Unknown'
                
                # Extract team info (simplified - would need more parsing)
                from_team = None
                to_team = None
                
                transactions.append({
                    'player_name': player_name,
                    'transaction_type': trans_type,
                    'transaction_date': trans_date,
                    'from_team': from_team,
                    'to_team': to_team,
                    'description': description
                })
        
        print(f"✅ Fetched {len(transactions)} transactions from ESPN")
        
    except Exception as e:
        print(f"❌ Error fetching ESPN transactions: {e}")
    
    return transactions


def populate_transactions():
    """Populate player_transactions table"""
    
    print("="*80)
    print("🔄 NBA TRANSACTIONS POPULATION")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Database connection
    DATABASE_URL = os.getenv('DATABASE_URL')
    if not DATABASE_URL:
        print("❌ DATABASE_URL not set")
        return
    
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        print("✅ Connected to PostgreSQL")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return
    
    # Fetch transactions
    transactions = fetch_espn_transactions(days_back=30)
    
    # Insert into database
    inserted = 0
    for trans in transactions:
        try:
            cur.execute("""
                INSERT INTO player_transactions (
                    player_name, transaction_type, transaction_date,
                    from_team_id, to_team_id, trade_description
                )
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                trans['player_name'],
                trans['transaction_type'],
                trans['transaction_date'],
                trans['from_team'],
                trans['to_team'],
                trans['description']
            ))
            inserted += 1
        except Exception as e:
            print(f"   ⚠️  Error inserting transaction: {e}")
            conn.rollback()
            continue
    
    conn.commit()
    print(f"✅ Inserted {inserted} transactions")
    
    # Verification
    cur.execute("SELECT COUNT(*) FROM player_transactions WHERE transaction_date >= CURRENT_DATE - INTERVAL '30 days'")
    count = cur.fetchone()[0]
    print(f"   Recent transactions (last 30 days): {count}")
    
    cur.close()
    conn.close()
    
    print()
    print("="*80)
    print("✅ TRANSACTIONS POPULATED")
    print("="*80)


if __name__ == "__main__":
    populate_transactions()

