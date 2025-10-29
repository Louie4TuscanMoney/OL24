# 🎯 FIND ESPN'S REAL-TIME WEBSOCKET (5 MINUTES)

**Do this RIGHT NOW while the game is live!**

---

## 🚀 STEP-BY-STEP GUIDE:

### **1. Open ESPN (In Chrome/Brave):**

```
https://www.espn.com/nba/scoreboard
```

---

### **2. Open DevTools:**

**Mac:** `Cmd + Option + I`  
**Windows:** `F12` or `Ctrl + Shift + I`

---

### **3. Click "Network" Tab:**

![Network Tab Location]

---

### **4. Filter by "WS" (WebSocket):**

At the top of the Network tab, you'll see filter buttons:
```
All | Fetch/XHR | JS | CSS | Img | Media | Font | Doc | WS | Wasm | Manifest | Other
```

**Click "WS"** (WebSocket)

---

### **5. Refresh the Page:**

Press `Cmd+R` (Mac) or `Ctrl+R` (Windows)

---

### **6. LOOK FOR WEBSOCKET CONNECTIONS:**

You should see entries like:
```
Name                          Status    Type        
streaming                     101       websocket
push                          101       websocket
realtime-scores              101       websocket
```

**Click on any WebSocket entry!**

---

### **7. INSPECT THE WEBSOCKET:**

Once clicked, you'll see tabs:
- **Headers** - Shows the URL and connection info
- **Messages** - Shows live data flowing in
- **Timing** - Connection performance

**Click "Messages" tab!**

---

### **8. WATCH THE LIVE DATA:**

You'll see messages coming in every 1-2 seconds:

```
↓ {"type":"score","game":"401585370","home":60,"away":58}
↓ {"type":"score","game":"401585370","home":62,"away":58}
↓ {"type":"clock","game":"401585370","time":"PT05M01S"}
```

---

### **9. COPY THE WEBSOCKET URL:**

**In the "Headers" tab, find:**
```
Request URL: wss://push.api.espn.com/streaming/nba/scores
```

**COPY THIS ENTIRE URL!**

---

### **10. CHECK FOR AUTHENTICATION:**

**In the "Headers" tab, look for:**
- Cookies
- Authorization headers
- API keys in URL parameters

**Example:**
```
Cookie: ESPN_S2=ABC123...
Authorization: Bearer XYZ789...
```

---

## 📋 SEND ME THIS INFO:

**Once you find the WebSocket, send me:**

1. **WebSocket URL:**
   ```
   wss://[COPY THE FULL URL HERE]
   ```

2. **Any query parameters:**
   ```
   ?sport=basketball&league=nba&...
   ```

3. **Required headers (if any):**
   ```
   Cookie: ...
   Authorization: ...
   ```

4. **Example message:**
   ```json
   {
     "type": "score",
     "game": "401585370",
     ...
   }
   ```

---

## ⚡ I'LL IMMEDIATELY INTEGRATE IT!

**Once you send me the WebSocket URL and format:**

1. I'll write the Python WebSocket client (5 min)
2. Connect to ESPN's real-time feed
3. Parse the messages
4. Push to our WebSocket (1-second updates!)
5. Frontend displays instantly (<2s lag!)

---

## 🔧 IF YOU CAN'T FIND A WEBSOCKET:

**Then look for API calls in "Fetch/XHR" filter:**

```
Look for URLs like:
- /apis/site/v2/sports/basketball/nba/scoreboard/live
- /live/v1/nba/scores
- /realtime/scores/nba
```

**These might be polled every second instead of WebSocket!**

---

## 🎮 DO IT NOW:

**Game is LIVE right now - perfect time to inspect!**

**Open ESPN.com, open DevTools, find the WebSocket, and send me the URL!** 🕵️

I'll have it integrated in 10 minutes! 🚀

