import { onMount, onCleanup, createSignal } from 'solid-js';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';

interface Player {
  player_id: string;
  x: number;
  y: number;
  z: number;
  team: string;
}

interface Court3DFrame {
  timestamp: string;
  period: number;
  event_type: string;
  home_players: Player[];
  away_players: Player[];
  ball_position: { x: number; y: number; z: number };
}

interface Props {
  gameData?: Court3DFrame;
}

export default function BasketballCourt3D(props: Props) {
  let canvasRef: HTMLCanvasElement;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let controls: OrbitControls;
  let playerMeshes: Map<string, THREE.Mesh> = new Map();
  let ballMesh: THREE.Mesh;

  onMount(() => {
    // Initialize Three.js scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);

    // Camera
    camera = new THREE.PerspectiveCamera(
      60,
      canvasRef.clientWidth / canvasRef.clientHeight,
      0.1,
      1000
    );
    camera.position.set(47, 80, 70);
    camera.lookAt(47, 0, 25);

    // Renderer
    renderer = new THREE.WebGLRenderer({ 
      canvas: canvasRef, 
      antialias: true,
      alpha: true 
    });
    renderer.setSize(canvasRef.clientWidth, canvasRef.clientHeight);
    renderer.shadowMap.enabled = true;

    // Controls
    controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;

    // Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(50, 100, 50);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Build court
    buildCourt();

    // Initial players
    createPlayers();

    // Animation loop
    const animate = () => {
      requestAnimationFrame(animate);
      controls.update();
      
      // Update player positions from gameData
      if (props.gameData) {
        updatePlayerPositions(props.gameData);
      }
      
      renderer.render(scene, camera);
    };
    animate();

    // Handle resize
    const handleResize = () => {
      camera.aspect = canvasRef.clientWidth / canvasRef.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(canvasRef.clientWidth, canvasRef.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    onCleanup(() => {
      window.removeEventListener('resize', handleResize);
      renderer.dispose();
      controls.dispose();
    });
  });

  function buildCourt() {
    // Court floor
    const courtGeometry = new THREE.PlaneGeometry(94, 50);
    const courtMaterial = new THREE.MeshStandardMaterial({ 
      color: 0xd2691e,
      roughness: 0.8
    });
    const court = new THREE.Mesh(courtGeometry, courtMaterial);
    court.rotation.x = -Math.PI / 2;
    court.receiveShadow = true;
    scene.add(court);

    // Court lines
    const lineMaterial = new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 });

    // Boundary
    const boundary = new THREE.EdgesGeometry(courtGeometry);
    const boundaryLine = new THREE.LineSegments(boundary, lineMaterial);
    boundaryLine.position.y = 0.1;
    boundaryLine.rotation.x = -Math.PI / 2;
    scene.add(boundaryLine);

    // Center circle
    const circleGeometry = new THREE.RingGeometry(5.9, 6, 64);
    const circleMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, side: THREE.DoubleSide });
    const centerCircle = new THREE.Mesh(circleGeometry, circleMaterial);
    centerCircle.rotation.x = -Math.PI / 2;
    centerCircle.position.set(47, 0.1, 25);
    scene.add(centerCircle);

    // 3-point arcs (simplified)
    const arcCurve = new THREE.EllipseCurve(
      0, 0,
      23.75, 23.75,
      0, Math.PI,
      false,
      0
    );
    const arcPoints = arcCurve.getPoints(50);
    const arcGeometry = new THREE.BufferGeometry().setFromPoints(arcPoints);
    
    // Home 3-point arc
    const homeArc = new THREE.Line(arcGeometry, lineMaterial);
    homeArc.rotation.x = -Math.PI / 2;
    homeArc.position.set(5.25, 0.1, 25);
    scene.add(homeArc);

    // Away 3-point arc
    const awayArc = new THREE.Line(arcGeometry, lineMaterial);
    awayArc.rotation.x = -Math.PI / 2;
    awayArc.rotation.z = Math.PI;
    awayArc.position.set(88.75, 0.1, 25);
    scene.add(awayArc);

    // Hoops
    createHoop(5.25, 25);
    createHoop(88.75, 25);
  }

  function createHoop(x: number, y: number) {
    // Rim
    const rimGeometry = new THREE.TorusGeometry(0.75, 0.05, 16, 100);
    const rimMaterial = new THREE.MeshStandardMaterial({ color: 0xff6600 });
    const rim = new THREE.Mesh(rimGeometry, rimMaterial);
    rim.rotation.x = Math.PI / 2;
    rim.position.set(x, 10, y);
    scene.add(rim);

    // Backboard
    const backboardGeometry = new THREE.BoxGeometry(0.5, 3.5, 6);
    const backboardMaterial = new THREE.MeshStandardMaterial({ 
      color: 0xffffff,
      transparent: true,
      opacity: 0.3
    });
    const backboard = new THREE.Mesh(backboardGeometry, backboardMaterial);
    backboard.position.set(x - 1, 11, y);
    scene.add(backboard);
  }

  function createPlayers() {
    // Create 10 player meshes (5 per team)
    for (let i = 0; i < 5; i++) {
      // Home team (Lakers gold)
      const homeGeometry = new THREE.CylinderGeometry(1, 1, 6, 32);
      const homeMaterial = new THREE.MeshStandardMaterial({ color: 0xfdb927 });
      const homePlayer = new THREE.Mesh(homeGeometry, homeMaterial);
      homePlayer.castShadow = true;
      homePlayer.position.set(20 + i * 5, 3, 20);
      scene.add(homePlayer);
      playerMeshes.set(`home_${i}`, homePlayer);

      // Away team (Celtics green)
      const awayGeometry = new THREE.CylinderGeometry(1, 1, 6, 32);
      const awayMaterial = new THREE.MeshStandardMaterial({ color: 0x007a33 });
      const awayPlayer = new THREE.Mesh(awayGeometry, awayMaterial);
      awayPlayer.castShadow = true;
      awayPlayer.position.set(20 + i * 5, 3, 30);
      scene.add(awayPlayer);
      playerMeshes.set(`away_${i}`, awayPlayer);
    }

    // Ball
    const ballGeometry = new THREE.SphereGeometry(0.5, 32, 32);
    const ballMaterial = new THREE.MeshStandardMaterial({ color: 0xff6600 });
    ballMesh = new THREE.Mesh(ballGeometry, ballMaterial);
    ballMesh.castShadow = true;
    ballMesh.position.set(47, 5, 25);
    scene.add(ballMesh);
  }

  function updatePlayerPositions(data: Court3DFrame) {
    // Update home players
    data.home_players.forEach((player, i) => {
      const mesh = playerMeshes.get(`home_${i}`);
      if (mesh) {
        mesh.position.x = player.x;
        mesh.position.z = player.y;
        mesh.position.y = 3; // Player height
      }
    });

    // Update away players
    data.away_players.forEach((player, i) => {
      const mesh = playerMeshes.get(`away_${i}`);
      if (mesh) {
        mesh.position.x = player.x;
        mesh.position.z = player.y;
        mesh.position.y = 3;
      }
    });

    // Update ball
    if (ballMesh && data.ball_position) {
      ballMesh.position.x = data.ball_position.x;
      ballMesh.position.y = data.ball_position.z;
      ballMesh.position.z = data.ball_position.y;
    }
  }

  return (
    <div class="w-full h-full">
      <canvas 
        ref={canvasRef!} 
        class="w-full h-full rounded-2xl"
        style={{ "min-height": "500px" }}
      />
    </div>
  );
}

