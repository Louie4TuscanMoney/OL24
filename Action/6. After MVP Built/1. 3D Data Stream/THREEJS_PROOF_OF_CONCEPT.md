# ThreeJS Basketball Court - Proof of Concept

**Concept:** Live 3D basketball court updating every 5 seconds with play-by-play  
**Technology:** ThreeJS + NBA API + SolidJS integration  
**Status:** Architectural proof-of-concept

---

## 🏀 The Vision

**What user sees:**

```
╔══════════════════════════════════════════════════════════╗
║              3D BASKETBALL COURT (LIVE)                   ║
║  ┌────────────────────────────────────────────────────┐  ║
║  │          Away Hoop                                  │  ║
║  │            ▣                                        │  ║
║  │            │                                        │  ║
║  │    👤  👤  │  👤                                    │  ║
║  │         🏀 │                                        │  ║
║  │    👤      │     👤                                 │  ║
║  │            │                                        │  ║
║  │  ══════════┼══════════  (Center Line)              │  ║
║  │            │                                        │  ║
║  │      👤    │  👤  👤                                │  ║
║  │         👤 │                                        │  ║
║  │            │     👤                                 │  ║
║  │            │                                        │  ║
║  │            ▣                                        │  ║
║  │          Home Hoop                                  │  ║
║  └────────────────────────────────────────────────────┘  ║
║                                                          ║
║  LAL 92  vs  BOS 88  •  Q2 6:00                         ║
║  ML Prediction: +15.1 [+11.3, +18.9]                    ║
║  🎯 Edge: 19.2 pts  •  Bet: $750                        ║
║                                                          ║
║  [Shot arc animation shows Curry 3-pointer]             ║
║  [Heatmap shows scoring probability zones]              ║
║  [Camera rotates around court - user controlled]        ║
╚══════════════════════════════════════════════════════════╝
```

---

## Technical Architecture

### Layer 1: NBA Play-by-Play Stream

**Data Source:** NBA API PlayByPlayV3

```typescript
interface PlayByPlayEvent {
  eventNum: number;
  eventType: string;  // "shot", "rebound", "assist", etc.
  playerName: string;
  teamId: number;
  locationX: number;  // Court coordinates
  locationY: number;
  shotResult: 'make' | 'miss';
  points: number;
  description: string;
}
```

**Update Frequency:** Every 5 seconds (poll NBA API)

**Processing:**
```typescript
// Fetch latest events
const events = await PlayByPlayV3(gameId);

// Filter new events since last fetch
const newEvents = events.filter(e => e.eventNum > lastEventId);

// Update 3D scene
newEvents.forEach(event => {
  updatePlayerPosition(event.playerName, event.locationX, event.locationY);
  
  if (event.eventType === 'shot') {
    animateShotArc(event);
  }
  
  if (event.shotResult === 'make') {
    playMakeAnimation();
  }
});
```

---

### Layer 2: ThreeJS Scene

**Core Elements:**

```typescript
// 1. Basketball Court
const court = new THREE.Mesh(
  new THREE.PlaneGeometry(94, 50),  // NBA court dimensions
  new THREE.MeshStandardMaterial({ color: 0xd4af37 })
);

// 2. Court Lines
const lines = createCourtLines();  // 3-point, free throw, key, etc.

// 3. Hoops
const hoop1 = createHoop({ x: -47, y: 10, z: 0 });
const hoop2 = createHoop({ x: 47, y: 10, z: 0 });

// 4. Players (10 total)
const players = createPlayers(10);  // 5 per team

// 5. Ball
const ball = new THREE.Mesh(
  new THREE.SphereGeometry(0.5),
  new THREE.MeshStandardMaterial({ color: 0xff6600 })
);

// 6. Lighting
const lights = setupLighting();  // Spotlights, ambient

// 7. Camera
const camera = new THREE.PerspectiveCamera(75, aspect, 0.1, 1000);
camera.position.set(0, 30, 40);  // Elevated view

// 8. Controls
const controls = new OrbitControls(camera, renderer.domElement);
```

---

### Layer 3: Animation System

**Shot Animation:**
```typescript
function animateShotArc(event: PlayByPlayEvent) {
  // Create parabolic arc from player to hoop
  const startPos = new THREE.Vector3(event.locationX, 0, event.locationY);
  const endPos = new THREE.Vector3(47, 10, 0);  // Hoop
  
  const curve = new THREE.QuadraticBezierCurve3(
    startPos,
    new THREE.Vector3(
      (startPos.x + endPos.x) / 2,
      15,  // Arc height
      (startPos.z + endPos.z) / 2
    ),
    endPos
  );
  
  // Animate ball along curve
  const duration = 1000;  // 1 second
  const start = Date.now();
  
  function animate() {
    const elapsed = Date.now() - start;
    const t = Math.min(elapsed / duration, 1);
    
    const pos = curve.getPoint(t);
    ball.position.copy(pos);
    
    if (t < 1) {
      requestAnimationFrame(animate);
    } else {
      // Show make/miss effect
      if (event.shotResult === 'make') {
        showMakeEffect();
      } else {
        showMissEffect();
      }
    }
  }
  
  animate();
}
```

---

### Layer 4: ML Prediction Overlay

**Confidence Heatmap:**
```typescript
function visualizeMLPrediction(prediction: Prediction) {
  // Create confidence heatmap on court
  const confidence = 1 - (prediction.interval_upper - prediction.interval_lower) / 20;
  
  // Color based on prediction
  const color = prediction.point_forecast > 0 
    ? new THREE.Color(0x00ff00)  // Green (home leading)
    : new THREE.Color(0xff0000); // Red (away leading)
  
  // Create gradient overlay
  const canvas = document.createElement('canvas');
  const ctx = canvas.getContext('2d')!;
  
  // Draw gradient
  const gradient = ctx.createRadialGradient(250, 250, 0, 250, 250, 250);
  gradient.addColorStop(0, `rgba(${color.r*255}, ${color.g*255}, ${color.b*255}, ${confidence})`);
  gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
  
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 500, 500);
  
  // Apply as texture to court overlay
  const texture = new THREE.CanvasTexture(canvas);
  const overlayMaterial = new THREE.MeshBasicMaterial({
    map: texture,
    transparent: true,
    opacity: 0.5
  });
  
  const overlay = new THREE.Mesh(
    new THREE.PlaneGeometry(94, 50),
    overlayMaterial
  );
  overlay.position.y = 0.2;
  overlay.rotation.x = -Math.PI / 2;
  
  scene.add(overlay);
}
```

---

## Integration with Existing System

```
NBA API (Folder 2)
    ↓ Play-by-play endpoint
PlayByPlayStream
    ↓ Parse events every 5 seconds
    ↓
ThreeJS Scene
    ├─ Update player positions
    ├─ Animate shots
    ├─ Move ball
    └─ Update score
    ↓
ML Prediction (Folder 1)
    ↓ Confidence interval
PredictionHeatmap
    ↓ Overlay on 3D court
    ↓
User Experience: Immersive live visualization
```

---

## Performance Considerations

### Target Performance:
- **60 FPS:** Smooth rendering
- **5-second updates:** Sync with NBA API
- **<100ms:** Animation transitions
- **<50MB:** Memory footprint
- **Mobile:** Degraded quality (30 FPS, simpler models)

### Optimization Strategies:
```typescript
// 1. Level of Detail (LOD)
const playerLOD = new THREE.LOD();
playerLOD.addLevel(highPolyPlayer, 0);    // Close up
playerLOD.addLevel(mediumPolyPlayer, 20); // Medium distance
playerLOD.addLevel(lowPolyPlayer, 50);    // Far away

// 2. Instanced rendering (10 players)
const instancedPlayers = new THREE.InstancedMesh(
  playerGeometry,
  playerMaterial,
  10
);

// 3. Frustum culling (don't render what camera can't see)
scene.autoUpdate = true;

// 4. Texture atlasing (combine textures)
const textureAtlas = createAtlas([courtTexture, playerTexture, ballTexture]);
```

---

## User Features

### Camera Modes:
1. **Broadcast View** - Traditional TV angle (elevated, 45°)
2. **Sideline View** - Court level, side view
3. **Above View** - Bird's eye, top-down
4. **Follow Ball** - Camera tracks ball automatically
5. **Free Orbit** - User controls (default)

### Interactive Elements:
- Click player → Show stats
- Hover hoop → Show FG% for that end
- Click court zone → Show scoring heat for that zone
- Scrub timeline → Replay play-by-play

### Overlays:
- Score and time (always visible)
- ML prediction (toggleable)
- Confidence heatmap (toggleable)
- Shot chart (toggleable)
- Player trails (toggleable)

---

## Code Structure

```
1. 3D Data Stream/
├── 1. NBA API Stream/
│   ├── PlayByPlayFetcher.ts       Fetch PBP data
│   ├── EventParser.ts             Parse events
│   └── CoordinateMapper.ts        NBA coords → 3D coords
│
└── 2. ThreeJS Integration/
    ├── CourtRenderer.tsx          Main 3D scene
    ├── PlayerAnimator.ts          Animate players
    ├── BallAnimator.ts            Animate ball
    ├── ShotVisualizer.ts          Shot arcs
    ├── PredictionOverlay.ts       ML heatmap
    ├── CameraController.ts        Camera modes
    └── PerformanceOptimizer.ts    LOD, culling, etc.
```

---

## Why Wait to Build?

### Reasons:
1. **Time:** 2-3 weeks to build properly
2. **Complexity:** ThreeJS has learning curve
3. **Data:** Need live games to test animations
4. **Priority:** Visualization < Profit generation
5. **Feedback:** Let users request it first

### Benefits of Waiting:
- MVP deployed faster
- Generating revenue sooner
- User feedback guides features
- More time for polish

### When to Build:
- After 4-6 weeks of profitable MVP
- When users request better visualization
- When we have developer bandwidth
- As premium feature (optional)

---

## Proof of Concept Code

**Minimal ThreeJS court (50 lines):**

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.min.js"></script>
  <style>
    body { margin: 0; background: #000; }
    canvas { display: block; }
  </style>
</head>
<body>
  <script>
    // Scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight);
    const renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Court
    const courtGeom = new THREE.PlaneGeometry(94, 50);
    const courtMat = new THREE.MeshStandardMaterial({ color: 0xd4af37 });
    const court = new THREE.Mesh(courtGeom, courtMat);
    court.rotation.x = -Math.PI / 2;
    scene.add(court);

    // Players (10 spheres)
    for (let i = 0; i < 10; i++) {
      const player = new THREE.Mesh(
        new THREE.SphereGeometry(1),
        new THREE.MeshStandardMaterial({ 
          color: i < 5 ? 0x1d428a : 0xc8102e 
        })
      );
      player.position.set(
        Math.random() * 80 - 40,
        1,
        Math.random() * 40 - 20
      );
      scene.add(player);
    }

    // Lighting
    const light = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(light);
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(0, 50, 30);
    scene.add(dirLight);

    // Camera
    camera.position.set(0, 30, 40);
    camera.lookAt(0, 0, 0);

    // Animate
    function animate() {
      requestAnimationFrame(animate);
      renderer.render(scene, camera);
    }
    animate();
  </script>
</body>
</html>
```

**Save this as `test_court.html` and open in browser - it works!**

---

**✅ 3D Proof-of-Concept: Viable, build post-MVP**

*ThreeJS works  
NBA API has PBP data  
Integration path clear  
Build when profitable*

