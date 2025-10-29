"""
MAMBA LIVE WEBSOCKET - Real-Time Play-by-Play Streaming

Streams minute-by-minute scoring patterns to frontend
Shows countdown to Q2 6:00 Mamba trigger
Displays prediction in real-time

Usage:
    WebSocket: wss://your-app.railway.app/ws/mamba/{game_id}
"""

import asyncio
import json
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, List
import psycopg2
import os
import requests
from datetime import datetime

DATABASE_URL = os.getenv('DATABASE_URL')

class MambaLiveManager:
    """Manages WebSocket connections for live Mamba updates"""
    
    def __init__(self):
        # game_id -> List[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, game_id: str):
        await websocket.accept()
        if game_id not in self.active_connections:
            self.active_connections[game_id] = []
        self.active_connections[game_id].append(websocket)
        
    def disconnect(self, websocket: WebSocket, game_id: str):
        if game_id in self.active_connections:
            self.active_connections[game_id].remove(websocket)
            if not self.active_connections[game_id]:
                del self.active_connections[game_id]
    
    async def broadcast_to_game(self, game_id: str, message: dict):
        """Broadcast message to all clients watching this game"""
        if game_id not in self.active_connections:
            return
        
        dead_connections = []
        for connection in self.active_connections[game_id]:
            try:
                await connection.send_json(message)
            except:
                dead_connections.append(connection)
        
        # Clean up dead connections
        for conn in dead_connections:
            self.disconnect(conn, game_id)


manager = MambaLiveManager()


async def mamba_websocket_handler(websocket: WebSocket, game_id: str):
    """
    WebSocket handler for live Mamba updates
    
    Sends:
    - Play-by-play events (every 30 seconds)
    - Countdown to Mamba trigger
    - Feature extraction progress
    - Mamba prediction result
    - Performance tracking
    """
    await manager.connect(websocket, game_id)
    
    try:
        # Send initial game state
        initial_state = get_game_state(game_id)
        await websocket.send_json({
            'type': 'initial_state',
            'data': initial_state
        })
        
        # Keep connection alive and send updates
        while True:
            # Get current game state
            game_state = get_game_state(game_id)
            
            # Send update
            await websocket.send_json({
                'type': 'game_update',
                'data': game_state,
                'timestamp': datetime.now().isoformat()
            })
            
            # Check if Mamba should trigger
            if should_trigger_mamba(game_state):
                # Send trigger notification
                await websocket.send_json({
                    'type': 'mamba_trigger',
                    'message': '⚡ MAMBA TRIGGER AT Q2 6:00!',
                    'timestamp': datetime.now().isoformat()
                })
                
                # Get Mamba prediction
                prediction = get_mamba_prediction(game_id)
                
                if prediction:
                    await websocket.send_json({
                        'type': 'mamba_prediction',
                        'data': prediction,
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Wait before next update
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        manager.disconnect(websocket, game_id)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket, game_id)


def get_game_state(game_id: str) -> dict:
    """Get current game state with play-by-play data"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        # Get play-by-play count and latest events
        cur.execute("""
            SELECT 
                COUNT(*) as total_events,
                MAX(time_elapsed_seconds) as latest_time,
                MAX(period) as current_period,
                MAX(clock) as current_clock
            FROM play_by_play
            WHERE game_id = %s
        """, (game_id,))
        
        stats = cur.fetchone()
        
        # Get last 10 scoring events
        cur.execute("""
            SELECT 
                event_num,
                period,
                clock,
                home_score,
                away_score,
                score_margin,
                description
            FROM play_by_play
            WHERE game_id = %s
            ORDER BY event_num DESC
            LIMIT 10
        """, (game_id,))
        
        recent_events = []
        for row in cur.fetchall():
            recent_events.append({
                'event_num': row[0],
                'period': row[1],
                'clock': row[2],
                'home_score': row[3],
                'away_score': row[4],
                'margin': row[5],
                'description': row[6]
            })
        
        # Get scoring pattern for last 18 minutes
        cur.execute("""
            SELECT score_margin, time_elapsed_seconds
            FROM play_by_play
            WHERE game_id = %s
            AND time_elapsed_seconds <= 1080
            ORDER BY time_elapsed_seconds ASC
        """, (game_id,))
        
        pattern_data = [{'margin': row[0], 'time': row[1]} for row in cur.fetchall()]
        
        # Check if Mamba triggered
        cur.execute("""
            SELECT prediction, confidence, triggered_at
            FROM mamba_game_cache
            WHERE game_id = %s
        """, (game_id,))
        
        mamba_result = cur.fetchone()
        mamba_prediction = None
        if mamba_result:
            mamba_prediction = {
                'prediction': float(mamba_result[0]),
                'confidence': float(mamba_result[1]),
                'triggered_at': mamba_result[2].isoformat()
            }
        
        cur.close()
        conn.close()
        
        return {
            'game_id': game_id,
            'total_events': stats[0] or 0,
            'latest_time': stats[1] or 0,
            'current_period': stats[2] or 1,
            'current_clock': stats[3] or '12:00',
            'recent_events': recent_events,
            'pattern_data': pattern_data,
            'mamba_prediction': mamba_prediction,
            'countdown_to_trigger': get_countdown_to_trigger(stats[2], stats[3]),
            'trigger_ready': is_trigger_ready(stats[2], stats[3])
        }
        
    except Exception as e:
        print(f"Error getting game state: {e}")
        return {
            'game_id': game_id,
            'error': str(e)
        }


def should_trigger_mamba(game_state: dict) -> bool:
    """Check if Mamba should trigger (Q2 6:00)"""
    period = game_state.get('current_period', 0)
    clock = game_state.get('current_clock', '')
    
    return period == 2 and clock.startswith('6:0')


def get_countdown_to_trigger(period: int, clock: str) -> dict:
    """Calculate countdown to Q2 6:00"""
    if not period or not clock:
        return {'message': 'Waiting for game to start', 'seconds': None}
    
    if period == 1:
        # In Q1 - show time until Q2 6:00
        try:
            parts = clock.split(':')
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            
            # Time left in Q1
            q1_remaining = mins * 60 + secs
            
            # Time from Q2 start to 6:00
            q2_to_trigger = 6 * 60
            
            total_seconds = q1_remaining + q2_to_trigger
            
            return {
                'message': f'Mamba triggers in Q2 at 6:00',
                'seconds': total_seconds,
                'status': 'Q1 - Building pattern data'
            }
        except:
            return {'message': 'Q1 in progress', 'seconds': None}
    
    elif period == 2:
        # In Q2 - show time until 6:00
        try:
            parts = clock.split(':')
            mins = int(parts[0])
            secs = int(parts[1]) if len(parts) > 1 else 0
            
            current_seconds = mins * 60 + secs
            trigger_seconds = 6 * 60
            
            if current_seconds > trigger_seconds:
                diff = current_seconds - trigger_seconds
                return {
                    'message': f'Mamba triggers in {mins}:{secs:02d}',
                    'seconds': diff,
                    'status': 'COUNTDOWN'
                }
            elif current_seconds == trigger_seconds or abs(current_seconds - trigger_seconds) < 10:
                return {
                    'message': '⚡ MAMBA TRIGGERING NOW!',
                    'seconds': 0,
                    'status': 'TRIGGERED'
                }
            else:
                return {
                    'message': 'Mamba already triggered',
                    'seconds': 0,
                    'status': 'COMPLETE'
                }
        except:
            return {'message': 'Q2 in progress', 'seconds': None}
    
    else:
        return {
            'message': 'Mamba trigger window passed',
            'seconds': 0,
            'status': 'PASSED'
        }


def is_trigger_ready(period: int, clock: str) -> bool:
    """Check if we're at the trigger point"""
    return period == 2 and clock and clock.startswith('6:0')


def get_mamba_prediction(game_id: str) -> dict:
    """Get Mamba prediction from cache"""
    try:
        conn = psycopg2.connect(DATABASE_URL)
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                prediction,
                confidence,
                features,
                home_score,
                away_score,
                current_margin,
                triggered_at
            FROM mamba_game_cache
            WHERE game_id = %s
        """, (game_id,))
        
        result = cur.fetchone()
        
        if not result:
            return None
        
        cur.close()
        conn.close()
        
        return {
            'prediction': float(result[0]),
            'confidence': float(result[1]),
            'features': json.loads(result[2]) if result[2] else None,
            'game_state': {
                'home_score': result[3],
                'away_score': result[4],
                'margin': result[5]
            },
            'triggered_at': result[6].isoformat() if result[6] else None
        }
        
    except Exception as e:
        print(f"Error getting prediction: {e}")
        return None


# Add this to trading_dashboard_api.py:
"""
from mamba_live_websocket import mamba_websocket_handler

@app.websocket("/ws/mamba/{game_id}")
async def websocket_mamba_endpoint(websocket: WebSocket, game_id: str):
    await mamba_websocket_handler(websocket, game_id)
"""

