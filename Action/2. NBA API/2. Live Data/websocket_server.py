"""
WebSocket Server for Real-Time Updates
Broadcasts NBA scores and ML predictions to SolidJS frontend

Following: NBA_API/LIVE_DATA_INTEGRATION.md
Speed: Updates every 5 seconds
"""

import asyncio
import json
from datetime import datetime
from typing import Set, Dict
import websockets
from websockets.server import WebSocketServerProtocol

class LiveDataBroadcaster:
    """
    WebSocket server that broadcasts live game updates
    """
    
    def __init__(self, host='0.0.0.0', port=8765):
        """
        Args:
            host: Server host
            port: WebSocket port
        """
        self.host = host
        self.port = port
        self.clients: Set[WebSocketServerProtocol] = set()
        self.game_states: Dict = {}
        self.predictions: Dict = {}
        
    async def register_client(self, websocket: WebSocketServerProtocol):
        """Register new client connection"""
        self.clients.add(websocket)
        print(f"✅ Client connected. Total clients: {len(self.clients)}")
        
        # Send current state to new client
        if self.game_states:
            await websocket.send(json.dumps({
                'type': 'initial_state',
                'games': self.game_states,
                'predictions': self.predictions
            }))
    
    async def unregister_client(self, websocket: WebSocketServerProtocol):
        """Unregister client connection"""
        self.clients.remove(websocket)
        print(f"❌ Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast(self, message: Dict):
        """
        Broadcast message to all connected clients
        
        Args:
            message: Dict to send (will be JSON serialized)
        """
        if not self.clients:
            return
        
        message_str = json.dumps(message)
        
        # Send to all clients
        disconnected = set()
        for client in self.clients:
            try:
                await client.send(message_str)
            except websockets.exceptions.ConnectionClosed:
                disconnected.add(client)
        
        # Remove disconnected clients
        for client in disconnected:
            self.clients.discard(client)
    
    async def broadcast_score_update(self, game_id: str, game_data: Dict):
        """
        Broadcast score update
        
        Args:
            game_id: NBA game ID
            game_data: Current game state
        """
        self.game_states[game_id] = game_data
        
        await self.broadcast({
            'type': 'score_update',
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'data': game_data
        })
    
    async def broadcast_prediction(self, game_id: str, prediction: Dict):
        """
        Broadcast ML prediction
        
        Args:
            game_id: NBA game ID
            prediction: ML model prediction
        """
        self.predictions[game_id] = prediction
        
        await self.broadcast({
            'type': 'ml_prediction',
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'prediction': prediction
        })
        
        print(f"📡 Broadcast prediction for {game_id} to {len(self.clients)} clients")
    
    async def broadcast_pattern_progress(self, game_id: str, minutes_collected: int):
        """
        Broadcast pattern collection progress
        
        Args:
            game_id: Game ID
            minutes_collected: How many minutes buffered
        """
        await self.broadcast({
            'type': 'pattern_progress',
            'timestamp': datetime.now().isoformat(),
            'game_id': game_id,
            'minutes_collected': minutes_collected,
            'minutes_needed': 18,
            'progress_percent': min(100, (minutes_collected / 18) * 100)
        })
    
    async def handler(self, websocket: WebSocketServerProtocol, path: str):
        """
        WebSocket connection handler
        
        Args:
            websocket: Client connection
            path: Connection path
        """
        await self.register_client(websocket)
        
        try:
            # Keep connection alive
            async for message in websocket:
                # Handle client messages (e.g., subscriptions)
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'ping':
                        await websocket.send(json.dumps({'type': 'pong'}))
                    
                    elif data.get('type') == 'subscribe':
                        game_id = data.get('game_id')
                        # Client subscribed to specific game
                        pass
                        
                except json.JSONDecodeError:
                    pass
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            await self.unregister_client(websocket)
    
    async def start(self):
        """Start WebSocket server"""
        print("="*80)
        print("WEBSOCKET SERVER STARTING")
        print("="*80)
        print(f"Host: {self.host}")
        print(f"Port: {self.port}")
        print(f"Waiting for clients...")
        
        async with websockets.serve(self.handler, self.host, self.port):
            await asyncio.Future()  # Run forever


# Standalone server
if __name__ == "__main__":
    broadcaster = LiveDataBroadcaster(host='0.0.0.0', port=8765)
    asyncio.run(broadcaster.start())

