import { onMount, onCleanup, createEffect } from 'solid-js';
import * as THREE from 'three';

interface ModelVisualizationProps {
  prediction?: {
    current_diff: number;
    features: number[];
    prediction: number;
    confidence: number;
    edge: number;
  };
}

export default function MLModelVisualization3D(props: ModelVisualizationProps) {
  let container: HTMLDivElement;
  let scene: THREE.Scene;
  let camera: THREE.PerspectiveCamera;
  let renderer: THREE.WebGLRenderer;
  let animationId: number;
  
  // Neural network visualization elements
  let inputNodes: THREE.Mesh[] = [];
  let hiddenNodes: THREE.Mesh[] = [];
  let outputNode: THREE.Mesh;
  let connections: THREE.Line[] = [];
  let particles: THREE.Points;

  onMount(() => {
    // Scene setup
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0a1a);
    scene.fog = new THREE.Fog(0x0a0a1a, 10, 50);

    // Camera
    camera = new THREE.PerspectiveCamera(
      75,
      container.clientWidth / container.clientHeight,
      0.1,
      1000
    );
    camera.position.set(0, 0, 25);

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(renderer.domElement);

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.3);
    scene.add(ambientLight);

    const pointLight1 = new THREE.PointLight(0x9333ea, 2, 100);
    pointLight1.position.set(10, 10, 10);
    scene.add(pointLight1);

    const pointLight2 = new THREE.PointLight(0xec4899, 2, 100);
    pointLight2.position.set(-10, -10, 10);
    scene.add(pointLight2);

    // Create neural network visualization
    createNeuralNetwork();

    // Add particle system for "thinking" effect
    createParticleSystem();

    // Animation loop
    const animate = () => {
      animationId = requestAnimationFrame(animate);

      // Rotate the whole scene slowly
      scene.rotation.y += 0.002;

      // Pulse nodes
      const time = Date.now() * 0.001;
      inputNodes.forEach((node, i) => {
        node.scale.setScalar(1 + Math.sin(time + i * 0.5) * 0.1);
      });
      
      hiddenNodes.forEach((node, i) => {
        node.scale.setScalar(1 + Math.sin(time + i * 0.3) * 0.15);
      });

      if (outputNode) {
        outputNode.scale.setScalar(1.5 + Math.sin(time * 2) * 0.2);
      }

      // Animate particles
      if (particles) {
        particles.rotation.y += 0.001;
        particles.rotation.x += 0.0005;
      }

      // Update connection colors based on activity
      connections.forEach((conn, i) => {
        const material = conn.material as THREE.LineBasicMaterial;
        const intensity = 0.5 + Math.sin(time * 2 + i * 0.5) * 0.5;
        material.opacity = intensity * 0.6;
      });

      renderer.render(scene, camera);
    };
    animate();

    // Handle window resize
    const handleResize = () => {
      camera.aspect = container.clientWidth / container.clientHeight;
      camera.updateProjectionMatrix();
      renderer.setSize(container.clientWidth, container.clientHeight);
    };
    window.addEventListener('resize', handleResize);

    onCleanup(() => {
      window.removeEventListener('resize', handleResize);
      cancelAnimationFrame(animationId);
      renderer.dispose();
    });
  });

  const createNeuralNetwork = () => {
    // Input layer (18 features)
    const inputCount = 18;
    for (let i = 0; i < inputCount; i++) {
      const geometry = new THREE.SphereGeometry(0.3, 16, 16);
      const material = new THREE.MeshPhongMaterial({
        color: 0x9333ea,
        emissive: 0x9333ea,
        emissiveIntensity: 0.5,
        transparent: true,
        opacity: 0.8
      });
      const node = new THREE.Mesh(geometry, material);
      
      // Arrange in a circle
      const angle = (i / inputCount) * Math.PI * 2;
      const radius = 8;
      node.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        -5
      );
      
      scene.add(node);
      inputNodes.push(node);

      // Add label
      const sprite = createTextSprite(`F${i + 1}`, 0.5);
      sprite.position.copy(node.position);
      sprite.position.z += 1;
      scene.add(sprite);
    }

    // Hidden layer (ensemble of models)
    const hiddenCount = 6;
    const hiddenLabels = ['XGB', 'LGBM', 'RF', 'NN', 'Ridge', 'ET'];
    for (let i = 0; i < hiddenCount; i++) {
      const geometry = new THREE.SphereGeometry(0.5, 16, 16);
      const material = new THREE.MeshPhongMaterial({
        color: 0xfbbf24,
        emissive: 0xfbbf24,
        emissiveIntensity: 0.7,
        transparent: true,
        opacity: 0.9
      });
      const node = new THREE.Mesh(geometry, material);
      
      // Arrange in a circle
      const angle = (i / hiddenCount) * Math.PI * 2;
      const radius = 5;
      node.position.set(
        Math.cos(angle) * radius,
        Math.sin(angle) * radius,
        0
      );
      
      scene.add(node);
      hiddenNodes.push(node);

      // Add label
      const sprite = createTextSprite(hiddenLabels[i], 0.6);
      sprite.position.copy(node.position);
      sprite.position.z += 1.5;
      scene.add(sprite);
    }

    // Output node (prediction)
    const outputGeometry = new THREE.SphereGeometry(1, 32, 32);
    const outputMaterial = new THREE.MeshPhongMaterial({
      color: 0xec4899,
      emissive: 0xec4899,
      emissiveIntensity: 1,
      transparent: true,
      opacity: 1
    });
    outputNode = new THREE.Mesh(outputGeometry, outputMaterial);
    outputNode.position.set(0, 0, 5);
    scene.add(outputNode);

    // Add label
    const predSprite = createTextSprite('PREDICTION', 1);
    predSprite.position.set(0, 0, 7);
    scene.add(predSprite);

    // Create connections
    // Input to Hidden
    inputNodes.forEach(inputNode => {
      hiddenNodes.forEach(hiddenNode => {
        const points = [inputNode.position, hiddenNode.position];
        const geometry = new THREE.BufferGeometry().setFromPoints(points);
        const material = new THREE.LineBasicMaterial({
          color: 0x9333ea,
          transparent: true,
          opacity: 0.3
        });
        const line = new THREE.Line(geometry, material);
        scene.add(line);
        connections.push(line);
      });
    });

    // Hidden to Output
    hiddenNodes.forEach(hiddenNode => {
      const points = [hiddenNode.position, outputNode.position];
      const geometry = new THREE.BufferGeometry().setFromPoints(points);
      const material = new THREE.LineBasicMaterial({
        color: 0xfbbf24,
        transparent: true,
        opacity: 0.4
      });
      const line = new THREE.Line(geometry, material);
      scene.add(line);
      connections.push(line);
    });
  };

  const createTextSprite = (text: string, scale: number) => {
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d')!;
    canvas.width = 256;
    canvas.height = 128;
    
    context.fillStyle = 'rgba(255, 255, 255, 0.9)';
    context.font = 'bold 48px Arial';
    context.textAlign = 'center';
    context.fillText(text, 128, 80);
    
    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({ map: texture, transparent: true });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(scale * 4, scale * 2, 1);
    
    return sprite;
  };

  const createParticleSystem = () => {
    const particleCount = 1000;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);

    for (let i = 0; i < particleCount * 3; i++) {
      positions[i] = (Math.random() - 0.5) * 50;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    const material = new THREE.PointsMaterial({
      color: 0x9333ea,
      size: 0.1,
      transparent: true,
      opacity: 0.6,
      blending: THREE.AdditiveBlending
    });

    particles = new THREE.Points(geometry, material);
    scene.add(particles);
  };

  // Update visualization when prediction changes
  createEffect(() => {
    if (props.prediction && outputNode) {
      const pred = props.prediction;
      
      // Update output node color based on prediction confidence
      const material = outputNode.material as THREE.MeshPhongMaterial;
      const confidence = pred.confidence || 0.5;
      
      // Green for high confidence, red for low
      const color = confidence > 0.7 ? 0x10b981 : confidence > 0.5 ? 0xfbbf24 : 0xef4444;
      material.color.setHex(color);
      material.emissive.setHex(color);
      material.emissiveIntensity = confidence;

      // Pulse animation on new prediction
      const originalScale = outputNode.scale.x;
      outputNode.scale.setScalar(originalScale * 1.5);
      setTimeout(() => {
        if (outputNode) {
          outputNode.scale.setScalar(originalScale);
        }
      }, 300);

      // Activate connections based on feature importance
      if (pred.features) {
        connections.forEach((conn, i) => {
          const material = conn.material as THREE.LineBasicMaterial;
          const featureIndex = i % pred.features.length;
          const featureValue = Math.abs(pred.features[featureIndex]);
          const maxValue = Math.max(...pred.features.map(f => Math.abs(f)));
          const intensity = maxValue > 0 ? featureValue / maxValue : 0.5;
          material.opacity = intensity * 0.8;
        });
      }
    }
  });

  return (
    <div class="relative w-full h-full">
      <div ref={container!} class="w-full h-full" />
      
      {/* Info overlay */}
      <div class="absolute top-4 left-4 right-4 pointer-events-none">
        <div class="bg-black/50 backdrop-blur-md rounded-xl p-4 border border-purple-500/30">
          <div class="text-white/90 text-sm space-y-2">
            <div class="font-bold text-lg mb-2 text-purple-400">
              🧠 MAMBA MENTALITY SYSTEM
            </div>
            
            <div class="grid grid-cols-2 gap-4">
              <div>
                <div class="text-white/50 text-xs">INPUT LAYER</div>
                <div class="text-white font-semibold">18 Features</div>
              </div>
              
              <div>
                <div class="text-white/50 text-xs">ENSEMBLE</div>
                <div class="text-white font-semibold">6 Models</div>
              </div>
              
              <div>
                <div class="text-white/50 text-xs">PREDICTION</div>
                <div class="text-white font-semibold">
                  {props.prediction ? `${props.prediction.prediction > 0 ? '+' : ''}${props.prediction.prediction.toFixed(1)}` : 'Waiting...'}
                </div>
              </div>
              
              <div>
                <div class="text-white/50 text-xs">CONFIDENCE</div>
                <div class="text-white font-semibold">
                  {props.prediction ? `${(props.prediction.confidence * 100).toFixed(1)}%` : '-'}
                </div>
              </div>
            </div>

            <div class="mt-3 pt-3 border-t border-white/10">
              <div class="text-white/50 text-xs mb-1">EDGE</div>
              <div class={`text-lg font-bold ${props.prediction && props.prediction.edge > 0 ? 'text-green-400' : 'text-red-400'}`}>
                {props.prediction ? `${props.prediction.edge > 0 ? '+' : ''}${props.prediction.edge.toFixed(1)} pts` : '-'}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Legend */}
      <div class="absolute bottom-4 left-4 right-4 pointer-events-none">
        <div class="bg-black/50 backdrop-blur-md rounded-xl p-3 border border-purple-500/30">
          <div class="flex justify-around text-xs">
            <div class="flex items-center gap-2">
              <div class="w-3 h-3 rounded-full bg-purple-500"></div>
              <span class="text-white/70">Input Features</span>
            </div>
            <div class="flex items-center gap-2">
              <div class="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span class="text-white/70">Ensemble Models</span>
            </div>
            <div class="flex items-center gap-2">
              <div class="w-3 h-3 rounded-full bg-pink-500"></div>
              <span class="text-white/70">Prediction Output</span>
            </div>
          </div>
        </div>
      </div>

      {/* Status */}
      <div class="absolute top-4 right-4 pointer-events-none">
        <div class="bg-green-500/20 backdrop-blur-md rounded-full px-4 py-2 border border-green-500/30">
          <div class="flex items-center gap-2">
            <div class="w-2 h-2 rounded-full bg-green-400 animate-pulse"></div>
            <span class="text-green-300 text-xs font-semibold">PROCESSING</span>
          </div>
        </div>
      </div>
    </div>
  );
}

