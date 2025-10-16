# 3D Data Stream - Architecture & Proof of Concept

**Purpose:** Immersive 3D basketball court with live play-by-play visualization  
**Technology:** ThreeJS + NBA API play-by-play data  
**Timeline:** Post-MVP (Month 3 of NBA season)  
**Priority:** ENHANCEMENT (not MVP-critical)

---

## 🎯 The Vision

**3D Basketball Court that updates in real-time with play-by-play data:**

```
┌──────────────────────────────────────────────────────────────┐
│                  3D BASKETBALL COURT                          │
│  ┌────────────────────────────────────────────────────────┐ │
│  │                                                         │ │
│  │    [Player dots moving]                                │ │
│  │    [Ball tracking]                                     │ │
│  │    [Shot arcs visualized]                              │ │
│  │    [Heatmap of scoring zones]                          │ │
│  │                                                         │ │
│  │    LAL 92  vs  BOS 88                                  │ │
│  │    Q2 • 6:00                                           │ │
│  │                                                         │ │
│  │    ML Prediction: +15.1 [+11.3, +18.9]                │ │
│  │    [Confidence heatmap overlay]                        │ │
│  │                                                         │ │
│  └────────────────────────────────────────────────────────┘ │
│                                                              │
│  Updates: Every 5 seconds via NBA play-by-play API          │
│  Camera: User-controlled rotation and zoom                  │
│  Data: Full integration with existing pipeline              │
└──────────────────────────────────────────────────────────────┘
```

---

## Why ThreeJS?

**ThreeJS = 3D graphics library for web**

**Perfect for:**
- Real-time 3D visualization
- GPU-accelerated rendering
- Smooth animations
- Interactive camera controls
- WebGL-based (works in browser)

**Integrates with SolidJS:**
```typescript
// SolidJS component wrapping ThreeJS
const CourtVisualization: Component = () => {
  const [scene] = createSignal(new THREE.Scene());
  
  // ThreeJS runs inside SolidJS
  // SolidJS signals update ThreeJS objects
  // Perfect combination!
}
```

---

## Architecture

### Component 1: ThreeJS Court Renderer

**File:** `CourtRenderer.tsx`

```typescript
/**
 * 3D Basketball Court Renderer
 * ThreeJS scene inside SolidJS component
 */

import { Component, onMount, onCleanup } from 'solid-js';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

const CourtRenderer: Component = () => {
  let container: HTMLDivElement;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let controls: OrbitControls;

  onMount(() => {
    initScene();
    createCourt();
    animate();
  });

  const initScene = () => {
    // Create scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Create camera
    camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 30, 40);
    camera.lookAt(0, 0, 0);

    // Create renderer
    renderer = new THREE.WebGLRenderer({ antialias: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    container.appendChild(renderer.domElement);

    // Add controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
  };

  const createCourt = () => {
    // Court floor (94 x 50 feet NBA)
    const courtGeometry = new THREE.PlaneGeometry(94, 50);
    const courtMaterial = new THREE.MeshStandardMaterial({
      color: 0xd4af37, // Court wood color
      roughness: 0.7
    });
    const court = new THREE.Mesh(courtGeometry, courtMaterial);
    court.rotation.x = -Math.PI / 2;
    scene.add(court);

    // Court lines (white paint)
    addCourtLines();

    // 3-point lines
    addThreePointLines();

    // Hoops
    addHoop(-47, 0, 0); // Away hoop
    addHoop(47, 0, 0);  // Home hoop

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(0, 50, 30);
    scene.add(directionalLight);
  };

  const addCourtLines = () => {
    // Center circle
    // Free throw lines
    // Sidelines
    // Baseline
    // Key (paint)
    // etc.
  };

  const animate = () => {
    requestAnimationFrame(animate);
    controls.update();
    renderer.render(scene, camera);
  };

  return <div ref={container!} class="w-full h-full" />;
};

export default CourtRenderer;
```

---

### Component 2: Play-by-Play Data Stream

**File:** `PlayByPlayStream.ts`

```typescript
/**
 * NBA Play-by-Play Data Stream
 * Fetches and processes live PBP data every 5 seconds
 */

import { nba_api } from 'nba_api';

interface PlayEvent {
  timestamp: number;
  event_type: 'shot' | 'rebound' | 'assist' | 'turnover' | 'foul';
  player: string;
  team: string;
  location_x: number;  // Court coordinates
  location_y: number;
  result: 'make' | 'miss' | null;
  points: number;
}

export class PlayByPlayStream {
  private gameId: string;
  private lastEventId: number = 0;
  private events: PlayEvent[] = [];

  constructor(gameId: string) {
    this.gameId = gameId;
  }

  async fetchLatestEvents(): Promise<PlayEvent[]> {
    // Fetch from NBA API
    const pbp = await nba_api.playbyplayv3.PlayByPlayV3(
      game_id=this.gameId
    );

    // Parse events since last fetch
    const newEvents = pbp.play_by_play.get_dict()
      .filter(e => e.eventNum > this.lastEventId)
      .map(e => this.parseEvent(e));

    this.lastEventId = newEvents[newEvents.length - 1]?.eventNum || this.lastEventId;
    this.events.push(...newEvents);

    return newEvents;
  }

  private parseEvent(raw: any): PlayEvent {
    // Parse NBA API event to our format
    return {
      timestamp: Date.now(),
      event_type: this.getEventType(raw.eventType),
      player: raw.playerName,
      team: raw.teamId,
      location_x: raw.locationX || 0,
      location_y: raw.locationY || 0,
      result: raw.shotResult,
      points: raw.pointsScored || 0
    };
  }

  private getEventType(rawType: number): PlayEvent['event_type'] {
    // NBA API event type mapping
    // 1 = shot, 2 = miss, 3 = free throw, etc.
    // Map to our simplified types
    return 'shot'; // Simplified
  }

  // Get all events for replay
  getEventHistory(): PlayEvent[] {
    return this.events;
  }
}
```

---

### Component 3: Player Position Animator

**File:** `PlayerAnimator.ts`

```typescript
/**
 * Animate player positions on 3D court
 * Updates every 5 seconds with new play-by-play data
 */

import * as THREE from 'three';
import type { PlayEvent } from './PlayByPlayStream';

export class PlayerAnimator {
  private scene: THREE.Scene;
  private players: Map<string, THREE.Mesh> = new Map();

  constructor(scene: THREE.Scene) {
    this.scene = scene;
    this.createPlayers();
  }

  private createPlayers() {
    // Create 10 player meshes (5 per team)
    for (let i = 0; i < 10; i++) {
      const geometry = new THREE.SphereGeometry(1, 16, 16);
      const material = new THREE.MeshStandardMaterial({
        color: i < 5 ? 0x1d428a : 0xc8102e, // Team colors
        emissive: i < 5 ? 0x0a1a3a : 0x5a0816,
        metalness: 0.5,
        roughness: 0.5
      });

      const player = new THREE.Mesh(geometry, material);
      player.position.y = 1; // 1 unit above court
      
      this.scene.add(player);
      this.players.set(`player_${i}`, player);
    }
  }

  updatePlayerPositions(event: PlayEvent) {
    // Animate player to event location
    const player = this.players.get(event.player);
    if (player) {
      // Smooth animation to new position
      this.animateToPosition(
        player,
        event.location_x,
        event.location_y,
        1000 // 1 second animation
      );
    }
  }

  private animateToPosition(
    mesh: THREE.Mesh,
    targetX: number,
    targetZ: number,
    duration: number
  ) {
    // Use GSAP or manual animation
    const startX = mesh.position.x;
    const startZ = mesh.position.z;
    const startTime = Date.now();

    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);

      // Ease function
      const eased = this.easeInOutCubic(progress);

      mesh.position.x = startX + (targetX - startX) * eased;
      mesh.position.z = startZ + (targetZ - startZ) * eased;

      if (progress < 1) {
        requestAnimationFrame(animate);
      }
    };

    animate();
  }

  private easeInOutCubic(t: number): number {
    return t < 0.5
      ? 4 * t * t * t
      : 1 - Math.pow(-2 * t + 2, 3) / 2;
  }

  // Add shot arc visualization
  visualizeShot(event: PlayEvent) {
    if (event.event_type === 'shot') {
      // Create arc from player to hoop
      const arc = this.createShotArc(
        event.location_x,
        event.location_y,
        event.result === 'make' ? 0x00ff00 : 0xff0000
      );

      this.scene.add(arc);

      // Remove after animation
      setTimeout(() => {
        this.scene.remove(arc);
      }, 2000);
    }
  }

  private createShotArc(x: number, z: number, color: number): THREE.Line {
    // Create parabolic arc for shot trajectory
    const points = [];
    const hoopX = x > 0 ? 47 : -47; // Nearest hoop

    for (let i = 0; i <= 20; i++) {
      const t = i / 20;
      const posX = x + (hoopX - x) * t;
      const posY = 8 * t * (1 - t); // Parabola (max height 2 units)
      const posZ = z * (1 - t);
      
      points.push(new THREE.Vector3(posX, posY, posZ));
    }

    const geometry = new THREE.BufferGeometry().setFromPoints(points);
    const material = new THREE.LineBasicMaterial({ color });

    return new THREE.Line(geometry, material);
  }
}
```

---

### Component 4: ML Prediction Heatmap

**File:** `PredictionHeatmap.ts`

```typescript
/**
 * Overlay ML prediction confidence as heatmap on court
 */

import * as THREE from 'three';

export class PredictionHeatmap {
  private scene: THREE.Scene;
  private heatmapMesh: THREE.Mesh | null = null;

  constructor(scene: THREE.Scene) {
    this.scene = scene;
  }

  visualizePrediction(
    forecast: number,
    intervalLower: number,
    intervalUpper: number
  ) {
    // Create heatmap based on confidence
    const confidence = 1 - (intervalUpper - intervalLower) / 20;

    // Green gradient for home team lead
    // Red gradient for away team lead
    const color = forecast > 0 ? 0x00ff00 : 0xff0000;
    const opacity = confidence * 0.3;

    // Create plane overlay on court
    const geometry = new THREE.PlaneGeometry(94, 50);
    const material = new THREE.MeshBasicMaterial({
      color,
      transparent: true,
      opacity,
      side: THREE.DoubleSide
    });

    if (this.heatmapMesh) {
      this.scene.remove(this.heatmapMesh);
    }

    this.heatmapMesh = new THREE.Mesh(geometry, material);
    this.heatmapMesh.rotation.x = -Math.PI / 2;
    this.heatmapMesh.position.y = 0.1; // Just above court

    this.scene.add(this.heatmapMesh);

    // Add prediction text floating above court
    this.addPredictionText(forecast);
  }

  private addPredictionText(forecast: number) {
    // Use TextGeometry or HTML overlay
    // Show "+15.1" floating above center court
  }
}
```

---

## Integration with Existing System

```
NBA API (Folder 2)
    ↓ Play-by-play data
PlayByPlayStream
    ↓ Parse events
PlayerAnimator
    ↓ Update 3D positions
    ↓
ThreeJS Court
    ↓ Render
    ↓
ML Prediction (Folder 1)
    ↓ Confidence interval
PredictionHeatmap
    ↓ Overlay on court
    ↓
User sees: Live 3D court with ML prediction visualization
```

---

## Technical Requirements

### Dependencies:
```json
{
  "three": "^0.160.0",
  "@types/three": "^0.160.0",
  "gsap": "^3.12.0"
}
```

### Performance Targets:
- 60 FPS rendering
- 5-second update intervals
- <100ms animation transitions
- Mobile support (degraded quality)

### Data Sources:
1. NBA API play-by-play (live)
2. ML predictions (from Folder 1)
3. WebSocket stream (from Folder 2)

---

## User Experience

### Camera Controls:
- Orbit: Click + drag
- Zoom: Scroll wheel
- Pan: Right-click + drag
- Reset: Double-click

### Overlays:
- Score display
- ML prediction
- Confidence heatmap
- Shot trajectories
- Player stats on hover

### Performance Modes:
- **High:** Full 3D, all animations
- **Medium:** Simplified models, reduced particles
- **Low:** 2D top-down view (fallback)

---

## Why Wait?

**Reasons to build AFTER MVP:**

1. **Time Investment:** 2-3 weeks to build properly
2. **MVP Blocker:** Delays profitable betting system
3. **Data Requirement:** Need live games to test properly
4. **Priority:** Visualization < Profit generation
5. **User Feedback:** Let users tell us if they want it

**Better Strategy:**
- Deploy MVP first
- Generate profit
- Collect user feedback
- Build 3D if users request it
- Build 3D if we have bandwidth

**Current 2D dashboard is sufficient for MVP!**

---

## Cost-Benefit Analysis

### If We Build Now:
- Cost: 2-3 weeks delay
- Benefit: Cool visualization
- Risk: Delays revenue generation

### If We Build Later:
- Cost: Same 2-3 weeks (when we have time)
- Benefit: Cool visualization + we already have revenue
- Risk: None (MVP already deployed)

**Decision: Build later ✅**

---

## Implementation Phases (When We Build It)

### Phase 1: Basic Court (Week 1)
- ThreeJS scene setup
- Court geometry
- Camera controls
- Static player positions

### Phase 2: Play-by-Play Integration (Week 2)
- NBA API play-by-play parser
- Player position updates
- Shot visualization
- Score updates

### Phase 3: ML Integration (Week 3)
- Prediction heatmap overlay
- Confidence visualization
- Edge indicators
- Risk layer display

### Phase 4: Polish (Week 4)
- Smooth animations
- Mobile optimization
- Performance tuning
- User controls

---

## Proof of Concept

**Minimal 3D court in ThreeJS:**

```html
<!DOCTYPE html>
<html>
<head>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r160/three.min.js"></script>
</head>
<body>
  <script>
    // Scene
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    const renderer = new THREE.WebGLRenderer();
    renderer.setSize(window.innerWidth, window.innerHeight);
    document.body.appendChild(renderer.domElement);

    // Court
    const geometry = new THREE.PlaneGeometry(94, 50);
    const material = new THREE.MeshBasicMaterial({ color: 0xd4af37 });
    const court = new THREE.Mesh(geometry, material);
    court.rotation.x = -Math.PI / 2;
    scene.add(court);

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

**This works in browser - proof ThreeJS is viable!**

---

**✅ 3D Data Stream: Architecture defined, ready to build post-MVP**

*ThreeJS basketball court  
Live play-by-play integration  
ML prediction heatmap  
Build timeline: Month 3 of NBA season*

