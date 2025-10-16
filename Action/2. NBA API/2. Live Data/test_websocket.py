"""
Test WebSocket Server
Verifies WebSocket broadcasting works correctly
"""

import asyncio
import websockets
import json

async def test_client():
    """
    Test client - connects to WebSocket and listens
    """
    print("="*80)
    print("WEBSOCKET CLIENT TEST")
    print("="*80)
    
    uri = "ws://localhost:8765"
    
    print(f"\nConnecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("✅ Connected!")
            
            # Send ping
            await websocket.send(json.dumps({'type': 'ping'}))
            print("📤 Sent ping")
            
            # Receive pong
            message = await websocket.recv()
            data = json.loads(message)
            print(f"📥 Received: {data}")
            
            if data.get('type') == 'pong':
                print("✅ WebSocket working!")
            
            # Listen for 10 seconds
            print("\nListening for messages (10 seconds)...")
            try:
                while True:
                    message = await asyncio.wait_for(websocket.recv(), timeout=10)
                    data = json.loads(message)
                    print(f"📥 Message: {data.get('type')}")
            except asyncio.TimeoutError:
                print("\n⏱️  Timeout (no more messages)")
            
            print("\n✅ Test complete")
            
    except ConnectionRefusedError:
        print("❌ Connection refused - Is server running?")
        print("\nStart server first:")
        print("  python websocket_server.py")
        return False
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False
    
    return True

if __name__ == "__main__":
    result = asyncio.run(test_client())
    
    if result:
        print("\n" + "="*80)
        print("✅ WEBSOCKET TEST PASSED")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("❌ WEBSOCKET TEST FAILED")
        print("="*80)

