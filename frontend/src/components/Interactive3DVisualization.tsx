import React, { useRef, useEffect, useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  SelectChangeEvent,
  IconButton,
  Tooltip,
  Grid,
  Paper,
  Chip,
} from '@mui/material';
import { 
  Refresh, 
  ThreeDRotation, 
  ZoomIn, 
  ZoomOut, 
  CenterFocusStrong,
  ViewInAr 
} from '@mui/icons-material';
import * as THREE from 'three';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || '';

interface VisualizationProps {
  width?: number;
  height?: number;
}

interface RiskData {
  symbol: string;
  volatility: number;
  returns: number;
  sharpe: number;
  correlation: number;
}

const Interactive3DVisualization: React.FC<VisualizationProps> = ({ 
  width = 800, 
  height = 500 
}) => {
  const mountRef = useRef<HTMLDivElement>(null);
  const sceneRef = useRef<THREE.Scene | null>(null);
  const rendererRef = useRef<THREE.WebGLRenderer | null>(null);
  const cameraRef = useRef<THREE.PerspectiveCamera | null>(null);
  const controlsRef = useRef<any>(null);
  
  // Calculate responsive dimensions
  const [containerWidth, setContainerWidth] = useState(800);
  const [containerHeight, setContainerHeight] = useState(400);
  
  const [visualizationType, setVisualizationType] = useState<string>('risk_surface');
  const [loading, setLoading] = useState(false);
  const [riskData, setRiskData] = useState<RiskData[]>([]);
  const [correlationData, setCorrelationData] = useState<any>(null);

  // Update container dimensions based on parent
  useEffect(() => {
    const updateDimensions = () => {
      if (mountRef.current) {
        const rect = mountRef.current.getBoundingClientRect();
        const newWidth = Math.max(400, rect.width || 800);
        const newHeight = Math.max(300, 400);
        setContainerWidth(newWidth);
        setContainerHeight(newHeight);
      }
    };

    updateDimensions();
    window.addEventListener('resize', updateDimensions);
    return () => window.removeEventListener('resize', updateDimensions);
  }, []);

  // Initialize Three.js scene
  useEffect(() => {
    if (!mountRef.current || containerWidth === 0 || containerHeight === 0) return;

    // Clear any existing content
    if (mountRef.current.firstChild) {
      mountRef.current.removeChild(mountRef.current.firstChild);
    }

    // Scene setup
    const scene = new THREE.Scene();
    scene.background = new THREE.Color(0x1a1a2e);
    sceneRef.current = scene;

    // Camera setup
    const camera = new THREE.PerspectiveCamera(
      75,
      containerWidth / containerHeight,
      0.1,
      1000
    );
    camera.position.set(10, 10, 10);
    camera.lookAt(0, 0, 0);
    cameraRef.current = camera;

    // Renderer setup
    const renderer = new THREE.WebGLRenderer({ 
      antialias: true, 
      alpha: true,
      preserveDrawingBuffer: true 
    });
    renderer.setSize(containerWidth, containerHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.setClearColor(0x1a1a2e, 1);
    rendererRef.current = renderer;

    try {
      mountRef.current.appendChild(renderer.domElement);
    } catch (error) {
      console.error('Failed to append renderer to DOM:', error);
      return;
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x404040, 0.6);
    scene.add(ambientLight);

    const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    directionalLight.position.set(10, 10, 5);
    directionalLight.castShadow = true;
    scene.add(directionalLight);

    // Add some basic 3D objects to ensure rendering works
    const geometry = new THREE.BoxGeometry(2, 2, 2);
    const material = new THREE.MeshPhongMaterial({ color: 0x00ff00 });
    const cube = new THREE.Mesh(geometry, material);
    cube.position.set(0, 0, 0);
    scene.add(cube);

    // Add a grid for reference
    const gridHelper = new THREE.GridHelper(20, 20, 0x444444, 0x222222);
    scene.add(gridHelper);

    // Add axes helper
    const axesHelper = new THREE.AxesHelper(5);
    scene.add(axesHelper);

    // Controls (simplified - you'd need to install three/examples/jsm/controls/OrbitControls)
    // For now, we'll add basic mouse interaction
    let mouseX = 0;
    let mouseY = 0;
    let isMouseDown = false;

    const handleMouseDown = () => { isMouseDown = true; };
    const handleMouseUp = () => { isMouseDown = false; };
    const handleMouseMove = (event: MouseEvent) => {
      if (!isMouseDown) return;
      
      mouseX = (event.clientX / containerWidth) * 2 - 1;
      mouseY = -(event.clientY / containerHeight) * 2 + 1;
      
      camera.position.x = Math.sin(mouseX * Math.PI) * 15;
      camera.position.z = Math.cos(mouseX * Math.PI) * 15;
      camera.position.y = mouseY * 10 + 5;
      camera.lookAt(0, 0, 0);
    };

    renderer.domElement.addEventListener('mousedown', handleMouseDown);
    renderer.domElement.addEventListener('mouseup', handleMouseUp);
    renderer.domElement.addEventListener('mousemove', handleMouseMove);

    // Animation loop
    let animationId: number;
    const animate = () => {
      animationId = requestAnimationFrame(animate);
      
      // Rotate the cube for visual feedback
      if (cube) {
        cube.rotation.x += 0.01;
        cube.rotation.y += 0.01;
      }
      
      try {
        renderer.render(scene, camera);
      } catch (error) {
        console.error('Render error:', error);
      }
    };
    animate();

    // Cleanup
    return () => {
      if (animationId) {
        cancelAnimationFrame(animationId);
      }
      if (mountRef.current && renderer.domElement && mountRef.current.contains(renderer.domElement)) {
        mountRef.current.removeChild(renderer.domElement);
      }
      renderer.dispose();
      renderer.domElement.removeEventListener('mousedown', handleMouseDown);
      renderer.domElement.removeEventListener('mouseup', handleMouseUp);
      renderer.domElement.removeEventListener('mousemove', handleMouseMove);
    };
  }, [containerWidth, containerHeight]);

  // Fetch data for visualizations
  const fetchVisualizationData = async () => {
    setLoading(true);
    try {
      // Fetch correlation data
      const corrResponse = await axios.get(`${API_BASE_URL}/api/correlation`);
      setCorrelationData(corrResponse.data);

      // Generate mock risk data (in real app, this would come from API)
      const mockRiskData: RiskData[] = corrResponse.data.symbols.map((symbol: string, index: number) => ({
        symbol,
        volatility: Math.random() * 0.5 + 0.1, // 0.1 to 0.6
        returns: (Math.random() - 0.5) * 0.4, // -0.2 to 0.2
        sharpe: Math.random() * 3 - 1, // -1 to 2
        correlation: Math.random() * 2 - 1 // -1 to 1
      }));
      
      setRiskData(mockRiskData);
    } catch (error) {
      console.error('Error fetching visualization data:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchVisualizationData();
  }, []);

  // Create 3D visualizations
  useEffect(() => {
    if (!sceneRef.current || !riskData.length) return;

    // Clear existing objects
    const objectsToRemove = sceneRef.current.children.filter(
      child => child.type === 'Mesh' || child.type === 'Points'
    );
    objectsToRemove.forEach(obj => sceneRef.current!.remove(obj));

    if (visualizationType === 'risk_surface') {
      createRiskSurface();
    } else if (visualizationType === 'portfolio_sphere') {
      createPortfolioSphere();
    } else if (visualizationType === 'correlation_network') {
      createCorrelationNetwork();
    }
  }, [visualizationType, riskData, correlationData]);

  const createRiskSurface = () => {
    if (!sceneRef.current) return;

    // Create a risk surface using PlaneGeometry with height variations
    const geometry = new THREE.PlaneGeometry(20, 20, 32, 32);
    const positions = geometry.attributes.position;

    // Modify vertices based on risk data
    for (let i = 0; i < positions.count; i++) {
      const x = positions.getX(i);
      const z = positions.getZ(i);
      
      // Create a risk surface based on volatility and returns
      const riskHeight = Math.sin(x * 0.3) * Math.cos(z * 0.3) * 3 +
                        Math.random() * 2 - 1;
      
      positions.setY(i, riskHeight);
    }

    geometry.computeVertexNormals();

    // Create material with color gradient
    const material = new THREE.MeshLambertMaterial({
      color: 0x4fc3f7,
      wireframe: false,
      transparent: true,
      opacity: 0.8
    });

    const surface = new THREE.Mesh(geometry, material);
    surface.rotation.x = -Math.PI / 2;
    sceneRef.current.add(surface);

    // Add data points as spheres
    riskData.forEach((data, index) => {
      const sphereGeometry = new THREE.SphereGeometry(0.3, 16, 16);
      const sphereMaterial = new THREE.MeshLambertMaterial({
        color: data.returns > 0 ? 0x4caf50 : 0xf44336
      });
      
      const sphere = new THREE.Mesh(sphereGeometry, sphereMaterial);
      sphere.position.set(
        (data.volatility - 0.35) * 20,
        Math.abs(data.sharpe) * 2,
        (data.returns + 0.2) * 25
      );
      
      sceneRef.current!.add(sphere);
    });
  };

  const createPortfolioSphere = () => {
    if (!sceneRef.current) return;

    // Create a sphere with portfolio allocations
    const sphereGeometry = new THREE.SphereGeometry(8, 32, 32);
    
    // Create different colored segments for different stocks
    riskData.forEach((data, index) => {
      const segmentGeometry = new THREE.SphereGeometry(
        6 + data.volatility * 4, 
        8, 
        8,
        (index / riskData.length) * Math.PI * 2,
        Math.PI * 2 / riskData.length
      );
      
      const hue = (index / riskData.length) * 360;
      const color = new THREE.Color(`hsl(${hue}, 70%, 60%)`);
      
      const material = new THREE.MeshLambertMaterial({
        color: color,
        transparent: true,
        opacity: 0.7
      });
      
      const segment = new THREE.Mesh(segmentGeometry, material);
      sceneRef.current!.add(segment);
    });
  };

  const createCorrelationNetwork = () => {
    if (!sceneRef.current || !correlationData) return;

    const symbols = correlationData.symbols;
    const matrix = correlationData.matrix;
    
    // Create nodes for each stock
    const nodes: THREE.Mesh[] = [];
    symbols.forEach((symbol: string, index: number) => {
      const nodeGeometry = new THREE.SphereGeometry(0.5, 16, 16);
      const nodeMaterial = new THREE.MeshLambertMaterial({
        color: new THREE.Color(`hsl(${(index / symbols.length) * 360}, 70%, 60%)`)
      });
      
      const node = new THREE.Mesh(nodeGeometry, nodeMaterial);
      
      // Position nodes in a circle
      const angle = (index / symbols.length) * Math.PI * 2;
      node.position.set(
        Math.cos(angle) * 8,
        0,
        Math.sin(angle) * 8
      );
      
      nodes.push(node);
      sceneRef.current!.add(node);
    });

    // Create connections based on correlation strength
    for (let i = 0; i < symbols.length; i++) {
      for (let j = i + 1; j < symbols.length; j++) {
        const correlation = Math.abs(matrix[i][j]);
        
        if (correlation > 0.3) { // Only show strong correlations
          const points = [nodes[i].position, nodes[j].position];
          const lineGeometry = new THREE.BufferGeometry().setFromPoints(points);
          
          const lineMaterial = new THREE.LineBasicMaterial({
            color: correlation > 0.7 ? 0xff4444 : 0x44ff44,
            opacity: correlation,
            transparent: true
          });
          
          const line = new THREE.Line(lineGeometry, lineMaterial);
          sceneRef.current!.add(line);
        }
      }
    }
  };

  const handleVisualizationChange = (event: SelectChangeEvent<string>) => {
    setVisualizationType(event.target.value);
  };

  const handleRefresh = () => {
    fetchVisualizationData();
  };

  const resetCamera = () => {
    if (cameraRef.current) {
      cameraRef.current.position.set(10, 10, 10);
      cameraRef.current.lookAt(0, 0, 0);
    }
  };

  return (
    <Card>
      <CardContent>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6">
            🎮 Interactive 3D Visualizations
          </Typography>
          
          <Box display="flex" alignItems="center" gap={1}>
            <FormControl size="small" sx={{ minWidth: 180 }}>
              <InputLabel>Visualization Type</InputLabel>
              <Select
                value={visualizationType}
                label="Visualization Type"
                onChange={handleVisualizationChange}
              >
                <MenuItem value="risk_surface">Risk Surface</MenuItem>
                <MenuItem value="portfolio_sphere">Portfolio Sphere</MenuItem>
                <MenuItem value="correlation_network">Correlation Network</MenuItem>
              </Select>
            </FormControl>
            
            <Tooltip title="Reset Camera">
              <IconButton size="small" onClick={resetCamera}>
                <CenterFocusStrong />
              </IconButton>
            </Tooltip>
            
            <Tooltip title="Refresh Data">
              <IconButton size="small" onClick={handleRefresh} disabled={loading}>
                <Refresh />
              </IconButton>
            </Tooltip>
          </Box>
        </Box>

        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: { xs: '1fr', md: '2fr 1fr' },
          gap: 2,
          width: '100%'
        }}>
          <Paper sx={{ p: 1, bgcolor: 'background.default', width: '100%', overflow: 'hidden' }}>
            <div 
              ref={mountRef} 
              style={{ 
                width: '100%', 
                height: '400px',
                minHeight: '400px',
                border: '1px solid #333',
                borderRadius: '4px',
                overflow: 'hidden',
                backgroundColor: '#1a1a2e'
              }} 
            />
          </Paper>
          
          <Paper sx={{ p: 2, height: { xs: 'auto', md: containerHeight }, overflow: 'auto' }}>
            <Typography variant="subtitle2" gutterBottom>
              Visualization Info
            </Typography>
              
              {visualizationType === 'risk_surface' && (
                <Box>
                  <Typography variant="body2" paragraph>
                    <strong>Risk Surface:</strong> 3D surface showing risk-return relationships. 
                    Height represents risk level, colors indicate performance.
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={1}>
                    <Chip label="Green = Positive Returns" color="success" size="small" />
                    <Chip label="Red = Negative Returns" color="error" size="small" />
                  </Box>
                </Box>
              )}
              
              {visualizationType === 'portfolio_sphere' && (
                <Box>
                  <Typography variant="body2" paragraph>
                    <strong>Portfolio Sphere:</strong> 3D representation of portfolio allocation. 
                    Each segment represents a stock, size indicates volatility.
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                    Larger segments = Higher volatility
                  </Typography>
                </Box>
              )}
              
              {visualizationType === 'correlation_network' && (
                <Box>
                  <Typography variant="body2" paragraph>
                    <strong>Correlation Network:</strong> 3D network showing stock correlations. 
                    Lines connect correlated stocks.
                  </Typography>
                  <Box display="flex" flexWrap="wrap" gap={1}>
                    <Chip label="Red Lines = Strong Positive" color="error" size="small" />
                    <Chip label="Green Lines = Strong Negative" color="success" size="small" />
                  </Box>
                </Box>
              )}
              
              <Box mt={2}>
                <Typography variant="subtitle2" gutterBottom>
                  Controls:
                </Typography>
                <Typography variant="caption" display="block">
                  • Click and drag to rotate
                </Typography>
                <Typography variant="caption" display="block">
                  • Use controls above to reset view
                </Typography>
              </Box>
              
              {riskData.length > 0 && (
                <Box mt={2}>
                  <Typography variant="subtitle2" gutterBottom>
                    Data Points: {riskData.length}
                  </Typography>
                  <Typography variant="caption" color="textSecondary">
                  Showing top stocks with risk metrics
                </Typography>
              </Box>
            )}
          </Paper>
        </Box>
      </CardContent>
    </Card>
  );
};

export default Interactive3DVisualization;
