"""
USER AUTH MANAGER

Purpose: Manage user access requests and approved passwords
Author: Ontologic XYZ
Date: October 20, 2025

This stores:
- Access requests (phone + password)
- Approval status
- Approved passwords
"""

import sqlite3
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class UserAuthManager:
    """
    Manage user authentication and access requests
    """
    
    def __init__(self, db_path: str = "data/user_auth.db"):
        """
        Initialize user auth manager
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Access requests table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS access_requests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                phone TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                requested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                approved_at TEXT,
                notes TEXT
            )
        ''')
        
        # Approved passwords table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS approved_passwords (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                password_hash TEXT UNIQUE NOT NULL,
                phone TEXT,
                approved_at TEXT DEFAULT CURRENT_TIMESTAMP,
                last_login TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
        
        print(f"✅ User auth database initialized: {self.db_path}")
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def submit_access_request(self, phone: str, password: str) -> bool:
        """
        Submit a new access request
        
        Args:
            phone: User's phone number
            password: Desired password
            
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self._hash_password(password)
        
        try:
            cursor.execute('''
                INSERT INTO access_requests (phone, password_hash, status)
                VALUES (?, ?, 'pending')
            ''', (phone, password_hash))
            
            conn.commit()
            print(f"✅ Access request submitted: {phone}")
            return True
        except Exception as e:
            print(f"❌ Error submitting access request: {e}")
            return False
        finally:
            conn.close()
    
    def check_password(self, password: str) -> bool:
        """
        Check if a password is approved
        
        Args:
            password: Password to check
            
        Returns:
            True if approved
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self._hash_password(password)
        
        cursor.execute('''
            SELECT id FROM approved_passwords
            WHERE password_hash = ?
        ''', (password_hash,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result is not None
    
    def get_pending_requests(self) -> List[Dict]:
        """
        Get all pending access requests
        
        Returns:
            List of pending requests
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, phone, requested_at, notes
            FROM access_requests
            WHERE status = 'pending'
            ORDER BY requested_at DESC
        ''')
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def approve_request(self, request_id: int) -> bool:
        """
        Approve an access request
        
        Args:
            request_id: ID of the request to approve
            
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            # Get request details
            cursor.execute('''
                SELECT phone, password_hash FROM access_requests
                WHERE id = ?
            ''', (request_id,))
            
            result = cursor.fetchone()
            if not result:
                return False
            
            phone, password_hash = result
            
            # Add to approved passwords
            cursor.execute('''
                INSERT OR IGNORE INTO approved_passwords (password_hash, phone)
                VALUES (?, ?)
            ''', (password_hash, phone))
            
            # Update request status
            cursor.execute('''
                UPDATE access_requests
                SET status = 'approved', approved_at = ?
                WHERE id = ?
            ''', (datetime.now().isoformat(), request_id))
            
            conn.commit()
            print(f"✅ Request #{request_id} approved for {phone}")
            return True
        except Exception as e:
            print(f"❌ Error approving request: {e}")
            return False
        finally:
            conn.close()
    
    def reject_request(self, request_id: int, reason: str = None) -> bool:
        """
        Reject an access request
        
        Args:
            request_id: ID of the request to reject
            reason: Optional reason for rejection
            
        Returns:
            True if successful
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                UPDATE access_requests
                SET status = 'rejected', notes = ?
                WHERE id = ?
            ''', (reason, request_id))
            
            conn.commit()
            print(f"✅ Request #{request_id} rejected")
            return True
        except Exception as e:
            print(f"❌ Error rejecting request: {e}")
            return False
        finally:
            conn.close()
    
    def update_last_login(self, password: str):
        """Update last login time for a password"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        password_hash = self._hash_password(password)
        
        cursor.execute('''
            UPDATE approved_passwords
            SET last_login = ?
            WHERE password_hash = ?
        ''', (datetime.now().isoformat(), password_hash))
        
        conn.commit()
        conn.close()


# Example usage
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔐 USER AUTH MANAGER - EXAMPLE")
    print("="*80 + "\n")
    
    # Initialize
    manager = UserAuthManager()
    
    # Submit a test request
    manager.submit_access_request("+1 (555) 123-4567", "test password 123")
    
    # Get pending requests
    pending = manager.get_pending_requests()
    print(f"\n📋 Pending requests: {len(pending)}")
    for req in pending:
        print(f"  - ID: {req['id']}, Phone: {req['phone']}, Date: {req['requested_at']}")
    
    # Approve first request
    if pending:
        manager.approve_request(pending[0]['id'])
        
        # Check if password now works
        approved = manager.check_password("test password 123")
        print(f"\n✅ Password approved: {approved}")
    
    print("\n" + "="*80)
    print("✅ USER AUTH MANAGER READY")
    print("="*80)

