import React, { useState, useRef, useCallback, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import GlassCard from '../components/Common/GlassCard';

// Types
interface GraphNode {
  id: string;
  label: string;
  type: 'paper' | 'author' | 'concept' | 'institution' | 'keyword';
  x?: number;
  y?: number;
  vx?: number;
  vy?: number;
  fx?: number | null;
  fy?: number | null;
  metadata?: {
    title?: string;
    citations?: number;
    year?: number;
    description?: string;
    publications?: number;
    affiliation?: string;
    [key: string]: any;
  };
}

interface GraphEdge {
  source: string | GraphNode;
  target: string | GraphNode;
  type: 'citation' | 'collaboration' | 'similarity' | 'contains';
  weight: number;
}

interface GraphData {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

interface GraphStats {
  totalNodes: number;
  totalEdges: number;
  density: number;
  avgDegree: number;
}

type LayoutAlgorithm = 'force' | 'circular' | 'hierarchical' | 'radial';

// Node type configurations
const nodeTypeConfig = {
  paper: {
    color: '#3b82f6',
    glowColor: 'rgba(59, 130, 246, 0.6)',
    icon: '📄',
    label: 'Papers',
  },
  author: {
    color: '#a855f7',
    glowColor: 'rgba(168, 85, 247, 0.6)',
    icon: '👤',
    label: 'Authors',
  },
  concept: {
    color: '#10b981',
    glowColor: 'rgba(16, 185, 129, 0.6)',
    icon: '💡',
    label: 'Concepts',
  },
  institution: {
    color: '#f59e0b',
    glowColor: 'rgba(245, 158, 11, 0.6)',
    icon: '🏛️',
    label: 'Institutions',
  },
  keyword: {
    color: '#ec4899',
    glowColor: 'rgba(236, 72, 153, 0.6)',
    icon: '🔑',
    label: 'Keywords',
  },
};

// Edge type configurations
const edgeTypeConfig = {
  citation: { color: '#60a5fa', dashArray: '0', label: 'Citations' },
  collaboration: { color: '#c084fc', dashArray: '5,5', label: 'Collaborations' },
  similarity: { color: '#34d399', dashArray: '2,2', label: 'Similarities' },
  contains: { color: '#fbbf24', dashArray: '0', label: 'Contains' },
};

// Sample data for demonstration
const sampleData: GraphData = {
  nodes: [
    {
      id: '1',
      label: 'Neural Networks Deep Learning',
      type: 'paper',
      metadata: {
        title: 'Neural Networks and Deep Learning',
        citations: 1520,
        year: 2020,
      },
    },
    {
      id: '2',
      label: 'Dr. Sarah Chen',
      type: 'author',
      metadata: {
        publications: 45,
        affiliation: 'MIT',
      },
    },
    {
      id: '3',
      label: 'Machine Learning',
      type: 'concept',
      metadata: {
        description: 'Algorithms that learn from data',
      },
    },
    {
      id: '4',
      label: 'Transformers in NLP',
      type: 'paper',
      metadata: {
        title: 'Attention is All You Need',
        citations: 3200,
        year: 2017,
      },
    },
    {
      id: '5',
      label: 'Dr. John Smith',
      type: 'author',
      metadata: {
        publications: 67,
        affiliation: 'Stanford',
      },
    },
    {
      id: '6',
      label: 'Natural Language Processing',
      type: 'concept',
      metadata: {
        description: 'Understanding and generating human language',
      },
    },
    {
      id: '7',
      label: 'MIT',
      type: 'institution',
      metadata: {
        description: 'Massachusetts Institute of Technology',
      },
    },
    {
      id: '8',
      label: 'Deep Learning',
      type: 'keyword',
      metadata: {},
    },
  ],
  edges: [
    { source: '1', target: '2', type: 'collaboration', weight: 1 },
    { source: '1', target: '3', type: 'contains', weight: 1 },
    { source: '4', target: '5', type: 'collaboration', weight: 1 },
    { source: '4', target: '6', type: 'contains', weight: 1 },
    { source: '1', target: '4', type: 'citation', weight: 0.8 },
    { source: '3', target: '6', type: 'similarity', weight: 0.6 },
    { source: '2', target: '7', type: 'collaboration', weight: 1 },
    { source: '1', target: '8', type: 'contains', weight: 1 },
    { source: '4', target: '8', type: 'contains', weight: 1 },
    { source: '2', target: '5', type: 'similarity', weight: 0.4 },
  ],
};

const Graph: React.FC = () => {
  const [graphData, setGraphData] = useState<GraphData>(sampleData);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [layoutAlgorithm, setLayoutAlgorithm] = useState<LayoutAlgorithm>('force');
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });
  const [draggedNode, setDraggedNode] = useState<GraphNode | null>(null);

  // Node type filters
  const [filters, setFilters] = useState<Record<string, boolean>>({
    paper: true,
    author: true,
    concept: true,
    institution: true,
    keyword: true,
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | undefined>(undefined);
  const containerRef = useRef<HTMLDivElement>(null);

  // Calculate graph statistics
  const calculateStats = useCallback((): GraphStats => {
    const nodes = graphData.nodes.filter((n) => filters[n.type]);
    const edges = graphData.edges.filter((e) => {
      const sourceNode = graphData.nodes.find((n) => n.id === (typeof e.source === 'string' ? e.source : e.source.id));
      const targetNode = graphData.nodes.find((n) => n.id === (typeof e.target === 'string' ? e.target : e.target.id));
      return sourceNode && targetNode && filters[sourceNode.type] && filters[targetNode.type];
    });

    const totalNodes = nodes.length;
    const totalEdges = edges.length;
    const maxEdges = totalNodes * (totalNodes - 1) / 2;
    const density = maxEdges > 0 ? (totalEdges / maxEdges) * 100 : 0;
    const avgDegree = totalNodes > 0 ? (totalEdges * 2) / totalNodes : 0;

    return {
      totalNodes,
      totalEdges,
      density,
      avgDegree,
    };
  }, [graphData, filters]);

  const stats = calculateStats();

  // Apply layout algorithm
  const applyLayout = useCallback((data: GraphData, algorithm: LayoutAlgorithm) => {
    const nodes = [...data.nodes];
    const canvasWidth = canvasRef.current?.width || 800;
    const canvasHeight = canvasRef.current?.height || 600;
    const centerX = canvasWidth / 2;
    const centerY = canvasHeight / 2;

    switch (algorithm) {
      case 'circular':
        nodes.forEach((node, i) => {
          const angle = (i / nodes.length) * 2 * Math.PI;
          const radius = Math.min(canvasWidth, canvasHeight) * 0.35;
          node.x = centerX + radius * Math.cos(angle);
          node.y = centerY + radius * Math.sin(angle);
        });
        break;

      case 'hierarchical':
        const levels: Record<string, number> = {
          institution: 0,
          author: 1,
          paper: 2,
          concept: 3,
          keyword: 4,
        };
        const nodeLevels = nodes.map((n) => levels[n.type] || 2);
        const maxLevel = Math.max(...nodeLevels);

        nodeLevels.forEach((level, i) => {
          const nodesInLevel = nodeLevels.filter((l) => l === level).length;
          const indexInLevel = nodeLevels.slice(0, i).filter((l) => l === level).length;
          nodes[i].y = centerY + ((level / maxLevel) - 0.5) * canvasHeight * 0.8;
          nodes[i].x = centerX + ((indexInLevel / (nodesInLevel || 1)) - 0.5) * canvasWidth * 0.8;
        });
        break;

      case 'radial':
        const typeGroups: Record<string, GraphNode[]> = {};
        nodes.forEach((node) => {
          if (!typeGroups[node.type]) typeGroups[node.type] = [];
          typeGroups[node.type].push(node);
        });

        const types = Object.keys(typeGroups);
        types.forEach((type, typeIndex) => {
          const angle = (typeIndex / types.length) * 2 * Math.PI;
          const groupRadius = Math.min(canvasWidth, canvasHeight) * 0.3;
          const groupCenterX = centerX + groupRadius * Math.cos(angle);
          const groupCenterY = centerY + groupRadius * Math.sin(angle);

          typeGroups[type].forEach((node, nodeIndex) => {
            const nodeAngle = (nodeIndex / typeGroups[type].length) * 2 * Math.PI;
            const nodeRadius = 50;
            node.x = groupCenterX + nodeRadius * Math.cos(nodeAngle);
            node.y = groupCenterY + nodeRadius * Math.sin(nodeAngle);
          });
        });
        break;

      case 'force':
      default:
        // Initialize random positions if not set
        nodes.forEach((node) => {
          if (node.x === undefined) node.x = centerX + (Math.random() - 0.5) * canvasWidth * 0.6;
          if (node.y === undefined) node.y = centerY + (Math.random() - 0.5) * canvasHeight * 0.6;
          if (node.vx === undefined) node.vx = 0;
          if (node.vy === undefined) node.vy = 0;
        });
        break;
    }

    return { ...data, nodes };
  }, []);

  // Force simulation step
  const simulateForces = useCallback((data: GraphData) => {
    const nodes = data.nodes.filter((n) => filters[n.type]);
    const edges = data.edges.filter((e) => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
      const targetId = typeof e.target === 'string' ? e.target : e.target.id;
      const sourceNode = nodes.find((n) => n.id === sourceId);
      const targetNode = nodes.find((n) => n.id === targetId);
      return sourceNode && targetNode;
    });

    const alpha = 0.3;
    const charge = -300;
    const linkDistance = 100;
    const centerX = (canvasRef.current?.width || 800) / 2;
    const centerY = (canvasRef.current?.height || 600) / 2;

    // Reset forces
    nodes.forEach((node) => {
      if (node.fx === null && node.fy === null) {
        node.vx = (node.vx || 0) * 0.9; // damping
        node.vy = (node.vy || 0) * 0.9;
      }
    });

    // Repulsion between nodes
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const nodeA = nodes[i];
        const nodeB = nodes[j];
        const dx = (nodeB.x || 0) - (nodeA.x || 0);
        const dy = (nodeB.y || 0) - (nodeA.y || 0);
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = (charge * alpha) / (distance * distance);

        if (nodeA.fx === null && nodeA.fy === null) {
          nodeA.vx = (nodeA.vx || 0) - (dx / distance) * force;
          nodeA.vy = (nodeA.vy || 0) - (dy / distance) * force;
        }
        if (nodeB.fx === null && nodeB.fy === null) {
          nodeB.vx = (nodeB.vx || 0) + (dx / distance) * force;
          nodeB.vy = (nodeB.vy || 0) + (dy / distance) * force;
        }
      }
    }

    // Spring forces for edges
    edges.forEach((edge) => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
      const source = nodes.find((n) => n.id === sourceId);
      const target = nodes.find((n) => n.id === targetId);

      if (source && target) {
        const dx = (target.x || 0) - (source.x || 0);
        const dy = (target.y || 0) - (source.y || 0);
        const distance = Math.sqrt(dx * dx + dy * dy) || 1;
        const force = ((distance - linkDistance) / distance) * alpha * edge.weight;

        if (source.fx === null && source.fy === null) {
          source.vx = (source.vx || 0) + dx * force;
          source.vy = (source.vy || 0) + dy * force;
        }
        if (target.fx === null && target.fy === null) {
          target.vx = (target.vx || 0) - dx * force;
          target.vy = (target.vy || 0) - dy * force;
        }
      }
    });

    // Center gravity
    nodes.forEach((node) => {
      if (node.fx === null && node.fy === null) {
        const dx = centerX - (node.x || centerX);
        const dy = centerY - (node.y || centerY);
        node.vx = (node.vx || 0) + dx * alpha * 0.01;
        node.vy = (node.vy || 0) + dy * alpha * 0.01;
      }
    });

    // Update positions
    nodes.forEach((node) => {
      if (node.fx !== null && node.fy !== null) {
        node.x = node.fx;
        node.y = node.fy;
      } else {
        node.x = (node.x || 0) + (node.vx || 0);
        node.y = (node.y || 0) + (node.vy || 0);
      }
    });

    return data;
  }, [filters]);

  // Drawing function
  const drawGraph = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas?.getContext('2d');
    if (!canvas || !ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply transformations
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    const visibleNodes = graphData.nodes.filter((n) => filters[n.type]);
    const visibleEdges = graphData.edges.filter((e) => {
      const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
      const targetId = typeof e.target === 'string' ? e.target : e.target.id;
      const sourceNode = graphData.nodes.find((n) => n.id === sourceId);
      const targetNode = graphData.nodes.find((n) => n.id === targetId);
      return sourceNode && targetNode && filters[sourceNode.type] && filters[targetNode.type];
    });

    // Draw edges with glow effect
    visibleEdges.forEach((edge) => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
      const source = visibleNodes.find((n) => n.id === sourceId);
      const target = visibleNodes.find((n) => n.id === targetId);

      if (source && target && source.x !== undefined && target.x !== undefined) {
        const config = edgeTypeConfig[edge.type];

        // Animated glow effect
        const time = Date.now() / 1000;
        const glowIntensity = (Math.sin(time * 2 + edge.weight * 10) + 1) / 2;

        // Draw glow
        ctx.strokeStyle = config.color;
        ctx.lineWidth = 3 * edge.weight * glowIntensity;
        ctx.globalAlpha = 0.3;
        ctx.beginPath();
        ctx.moveTo(source.x, source.y!);
        ctx.lineTo(target.x, target.y!);
        ctx.stroke();

        // Draw main line
        ctx.strokeStyle = config.color;
        ctx.lineWidth = 2 * edge.weight;
        ctx.globalAlpha = 0.7;
        if (config.dashArray) {
          const dash = config.dashArray.split(',').map(Number);
          ctx.setLineDash(dash);
        } else {
          ctx.setLineDash([]);
        }
        ctx.beginPath();
        ctx.moveTo(source.x, source.y!);
        ctx.lineTo(target.x, target.y!);
        ctx.stroke();
        ctx.setLineDash([]);
      }
    });

    // Draw nodes
    visibleNodes.forEach((node) => {
      if (node.x === undefined || node.y === undefined) return;

      const config = nodeTypeConfig[node.type];
      const isSelected = selectedNode?.id === node.id;
      const isHovered = hoveredNode?.id === node.id;
      const radius = isSelected ? 25 : isHovered ? 20 : 15;

      // Draw glow for selected/hovered nodes
      if (isSelected || isHovered) {
        const gradient = ctx.createRadialGradient(node.x, node.y, 0, node.x, node.y, radius * 2);
        gradient.addColorStop(0, config.glowColor);
        gradient.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = gradient;
        ctx.globalAlpha = 0.6;
        ctx.beginPath();
        ctx.arc(node.x, node.y, radius * 2, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw node circle
      ctx.globalAlpha = 1;
      ctx.fillStyle = config.color;
      ctx.beginPath();
      ctx.arc(node.x, node.y, radius, 0, Math.PI * 2);
      ctx.fill();

      // Draw border
      ctx.strokeStyle = isSelected ? '#ffffff' : 'rgba(255, 255, 255, 0.3)';
      ctx.lineWidth = isSelected ? 3 : 1.5;
      ctx.stroke();

      // Draw label
      if (isSelected || isHovered) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = '#ffffff';
        ctx.font = 'bold 12px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        const label = node.label.length > 20 ? node.label.substring(0, 20) + '...' : node.label;

        // Draw text background
        const textMetrics = ctx.measureText(label);
        const padding = 6;
        ctx.fillStyle = 'rgba(0, 0, 0, 0.7)';
        ctx.fillRect(
          node.x - textMetrics.width / 2 - padding,
          node.y + radius + 5,
          textMetrics.width + padding * 2,
          20
        );

        // Draw text
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, node.x, node.y + radius + 15);
      }
    });

    ctx.restore();
  }, [graphData, filters, selectedNode, hoveredNode, zoom, pan]);

  // Animation loop
  useEffect(() => {
    const animate = () => {
      if (layoutAlgorithm === 'force') {
        setGraphData((prev) => simulateForces(prev));
      }
      drawGraph();
      animationRef.current = requestAnimationFrame(animate);
    };

    animate();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [layoutAlgorithm, simulateForces, drawGraph]);

  // Initialize layout on mount or layout change
  useEffect(() => {
    setGraphData((prev) => applyLayout(prev, layoutAlgorithm));
  }, [layoutAlgorithm, applyLayout]);

  // Handle canvas resize
  useEffect(() => {
    const handleResize = () => {
      const canvas = canvasRef.current;
      const container = containerRef.current;
      if (canvas && container) {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
      }
    };

    handleResize();
    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  // Mouse handlers
  const getMousePos = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return { x: 0, y: 0 };

    const rect = canvas.getBoundingClientRect();
    return {
      x: (e.clientX - rect.left - pan.x) / zoom,
      y: (e.clientY - rect.top - pan.y) / zoom,
    };
  };

  const findNodeAtPosition = (x: number, y: number): GraphNode | null => {
    const visibleNodes = graphData.nodes.filter((n) => filters[n.type]);
    return visibleNodes.find((node) => {
      if (node.x === undefined || node.y === undefined) return false;
      const dx = node.x - x;
      const dy = node.y - y;
      const distance = Math.sqrt(dx * dx + dy * dy);
      const radius = selectedNode?.id === node.id ? 25 : 15;
      return distance <= radius;
    }) || null;
  };

  const handleMouseDown = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);
    const node = findNodeAtPosition(pos.x, pos.y);

    if (node) {
      setDraggedNode(node);
      node.fx = node.x || 0;
      node.fy = node.y || 0;
    } else {
      setIsDragging(true);
      setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);

    if (draggedNode) {
      draggedNode.fx = pos.x;
      draggedNode.fy = pos.y;
    } else if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
    } else {
      const node = findNodeAtPosition(pos.x, pos.y);
      setHoveredNode(node);
    }
  };

  const handleMouseUp = () => {
    if (draggedNode) {
      draggedNode.fx = null;
      draggedNode.fy = null;
      setDraggedNode(null);
    }
    setIsDragging(false);
  };

  const handleClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const pos = getMousePos(e);
    const node = findNodeAtPosition(pos.x, pos.y);
    setSelectedNode(node);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    setZoom((prev) => Math.max(0.1, Math.min(3, prev * delta)));
  };

  // Filter toggle
  const toggleFilter = (type: string) => {
    setFilters((prev) => ({ ...prev, [type]: !prev[type] }));
  };

  // Search filter
  const filteredNodes = graphData.nodes.filter((node) =>
    node.label.toLowerCase().includes(searchQuery.toLowerCase()) ||
    node.metadata?.title?.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Get connected nodes
  const getConnectedNodes = (nodeId: string): GraphNode[] => {
    const connectedIds = new Set<string>();
    graphData.edges.forEach((edge) => {
      const sourceId = typeof edge.source === 'string' ? edge.source : edge.source.id;
      const targetId = typeof edge.target === 'string' ? edge.target : edge.target.id;
      if (sourceId === nodeId) connectedIds.add(targetId);
      if (targetId === nodeId) connectedIds.add(sourceId);
    });
    return graphData.nodes.filter((n) => connectedIds.has(n.id));
  };

  // Actions
  const handleAddNode = () => {
    const newNode: GraphNode = {
      id: `node-${Date.now()}`,
      label: 'New Node',
      type: 'concept',
      x: (canvasRef.current?.width || 800) / 2 / zoom - pan.x,
      y: (canvasRef.current?.height || 600) / 2 / zoom - pan.y,
    };
    setGraphData((prev) => ({
      ...prev,
      nodes: [...prev.nodes, newNode],
    }));
  };

  const handleClearGraph = () => {
    if (window.confirm('Are you sure you want to clear the graph?')) {
      setGraphData({ nodes: [], edges: [] });
      setSelectedNode(null);
    }
  };

  const handleDeleteNode = () => {
    if (selectedNode) {
      setGraphData((prev) => ({
        nodes: prev.nodes.filter((n) => n.id !== selectedNode.id),
        edges: prev.edges.filter((e) => {
          const sourceId = typeof e.source === 'string' ? e.source : e.source.id;
          const targetId = typeof e.target === 'string' ? e.target : e.target.id;
          return sourceId !== selectedNode.id && targetId !== selectedNode.id;
        }),
      }));
      setSelectedNode(null);
    }
  };

  const handleExpandNode = () => {
    if (selectedNode) {
      // Simulate expanding node by adding connected nodes
      const newNodes: GraphNode[] = [];
      const newEdges: GraphEdge[] = [];

      for (let i = 0; i < 3; i++) {
        const newNode: GraphNode = {
          id: `expanded-${selectedNode.id}-${i}-${Date.now()}`,
          label: `Related ${i + 1}`,
          type: selectedNode.type,
          x: (selectedNode.x || 0) + (Math.random() - 0.5) * 200,
          y: (selectedNode.y || 0) + (Math.random() - 0.5) * 200,
        };
        newNodes.push(newNode);
        newEdges.push({
          source: selectedNode.id,
          target: newNode.id,
          type: 'similarity',
          weight: 0.5 + Math.random() * 0.5,
        });
      }

      setGraphData((prev) => ({
        nodes: [...prev.nodes, ...newNodes],
        edges: [...prev.edges, ...newEdges],
      }));
    }
  };

  const handleResetView = () => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  };

  const handleLoadSampleData = () => {
    setGraphData(sampleData);
    setSelectedNode(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 p-6">
      <div className="max-w-[1920px] mx-auto">
        {/* Header */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-6"
        >
          <h1 className="text-4xl font-bold text-neon mb-2">
            Knowledge Graph Visualization
          </h1>
          <p className="text-gray-400">
            Explore and analyze relationships between research entities
          </p>
        </motion.div>

        {/* Stats Row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {[
            { label: 'Total Nodes', value: stats.totalNodes, icon: '🔵' },
            { label: 'Total Edges', value: stats.totalEdges, icon: '🔗' },
            { label: 'Graph Density', value: `${stats.density.toFixed(2)}%`, icon: '📊' },
            { label: 'Avg Degree', value: stats.avgDegree.toFixed(2), icon: '📈' },
          ].map((stat, index) => (
            <motion.div
              key={stat.label}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.1 }}
            >
              <GlassCard className="p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <p className="text-sm text-gray-400 mb-1">{stat.label}</p>
                    <p className="text-2xl font-bold text-white">{stat.value}</p>
                  </div>
                  <div className="text-3xl">{stat.icon}</div>
                </div>
              </GlassCard>
            </motion.div>
          ))}
        </div>

        {/* Main Layout */}
        <div className="grid grid-cols-12 gap-6">
          {/* Left Sidebar - Controls */}
          <div className="col-span-12 lg:col-span-3">
            <GlassCard className="p-6 space-y-6">
              <div>
                <h3 className="text-lg font-semibold text-white mb-4">Controls</h3>

                {/* Search */}
                <div className="mb-4">
                  <label className="block text-sm text-gray-400 mb-2">Search Nodes</label>
                  <input
                    type="text"
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    placeholder="Search by name or title..."
                    className="input-glass w-full text-sm"
                  />
                  {searchQuery && (
                    <div className="mt-2 max-h-40 overflow-y-auto">
                      {filteredNodes.map((node) => (
                        <div
                          key={node.id}
                          onClick={() => setSelectedNode(node)}
                          className="p-2 hover:bg-white/10 rounded cursor-pointer text-sm text-gray-300 transition-colors"
                        >
                          {nodeTypeConfig[node.type].icon} {node.label}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Node Type Filters */}
                <div className="mb-4">
                  <label className="block text-sm text-gray-400 mb-2">Node Types</label>
                  <div className="space-y-2">
                    {Object.entries(nodeTypeConfig).map(([type, config]) => (
                      <label
                        key={type}
                        className="flex items-center space-x-3 cursor-pointer group"
                      >
                        <input
                          type="checkbox"
                          checked={filters[type]}
                          onChange={() => toggleFilter(type)}
                          className="w-4 h-4 rounded border-gray-600 bg-white/10 text-cyber-500 focus:ring-cyber-500 focus:ring-offset-slate-950"
                        />
                        <span className="flex items-center space-x-2 text-sm text-gray-300 group-hover:text-white transition-colors">
                          <span>{config.icon}</span>
                          <span>{config.label}</span>
                          <span
                            className="w-3 h-3 rounded-full"
                            style={{ backgroundColor: config.color }}
                          />
                        </span>
                      </label>
                    ))}
                  </div>
                </div>

                {/* Layout Algorithm */}
                <div className="mb-4">
                  <label className="block text-sm text-gray-400 mb-2">Layout Algorithm</label>
                  <select
                    value={layoutAlgorithm}
                    onChange={(e) => setLayoutAlgorithm(e.target.value as LayoutAlgorithm)}
                    className="input-glass w-full text-sm"
                  >
                    <option value="force">Force-Directed</option>
                    <option value="circular">Circular</option>
                    <option value="hierarchical">Hierarchical</option>
                    <option value="radial">Radial</option>
                  </select>
                </div>

                {/* Zoom Controls */}
                <div className="mb-4">
                  <label className="block text-sm text-gray-400 mb-2">
                    Zoom: {(zoom * 100).toFixed(0)}%
                  </label>
                  <input
                    type="range"
                    min="10"
                    max="300"
                    value={zoom * 100}
                    onChange={(e) => setZoom(Number(e.target.value) / 100)}
                    className="w-full"
                  />
                </div>

                {/* Action Buttons */}
                <div className="space-y-2">
                  <button
                    onClick={handleAddNode}
                    className="btn-glass-primary w-full text-sm py-2"
                  >
                    + Add Node
                  </button>
                  <button
                    onClick={handleLoadSampleData}
                    className="btn-glass-secondary w-full text-sm py-2"
                  >
                    Load Sample Data
                  </button>
                  <button
                    onClick={handleResetView}
                    className="btn-glass-secondary w-full text-sm py-2"
                  >
                    Reset View
                  </button>
                  <button
                    onClick={handleClearGraph}
                    className="btn-glass-secondary w-full text-sm py-2 text-red-400 hover:text-red-300"
                  >
                    Clear Graph
                  </button>
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Center - Graph Canvas */}
          <div className="col-span-12 lg:col-span-6">
            <GlassCard className="p-4">
              <div
                ref={containerRef}
                className="relative w-full bg-slate-900/50 rounded-lg overflow-hidden"
                style={{ height: '700px' }}
              >
                <canvas
                  ref={canvasRef}
                  onMouseDown={handleMouseDown}
                  onMouseMove={handleMouseMove}
                  onMouseUp={handleMouseUp}
                  onMouseLeave={handleMouseUp}
                  onClick={handleClick}
                  onWheel={handleWheel}
                  className="cursor-grab active:cursor-grabbing"
                  style={{ display: 'block' }}
                />

                {/* Canvas Instructions */}
                <div className="absolute bottom-4 left-4 right-4 bg-black/70 backdrop-blur-md rounded-lg p-3 text-xs text-gray-300 space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="text-cyber-400">Click:</span>
                    <span>Select node</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-cyber-400">Drag:</span>
                    <span>Pan canvas or move node</span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className="text-cyber-400">Scroll:</span>
                    <span>Zoom in/out</span>
                  </div>
                </div>

                {/* Zoom Indicator */}
                <div className="absolute top-4 right-4 bg-black/70 backdrop-blur-md rounded-lg px-3 py-2 text-xs text-gray-300">
                  Zoom: {(zoom * 100).toFixed(0)}%
                </div>
              </div>
            </GlassCard>
          </div>

          {/* Right Sidebar - Node Details */}
          <div className="col-span-12 lg:col-span-3">
            <GlassCard className="p-6">
              <h3 className="text-lg font-semibold text-white mb-4">Node Details</h3>

              <AnimatePresence mode="wait">
                {selectedNode ? (
                  <motion.div
                    key={selectedNode.id}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    exit={{ opacity: 0, x: -20 }}
                    className="space-y-4"
                  >
                    {/* Node Header */}
                    <div className="flex items-start space-x-3">
                      <div
                        className="w-12 h-12 rounded-full flex items-center justify-center text-2xl"
                        style={{
                          backgroundColor: nodeTypeConfig[selectedNode.type].color,
                        }}
                      >
                        {nodeTypeConfig[selectedNode.type].icon}
                      </div>
                      <div className="flex-1 min-w-0">
                        <h4 className="text-sm font-semibold text-white mb-1 break-words">
                          {selectedNode.label}
                        </h4>
                        <p className="text-xs text-gray-400 capitalize">
                          {selectedNode.type}
                        </p>
                      </div>
                    </div>

                    {/* Node Properties */}
                    <div className="space-y-2">
                      <h5 className="text-sm font-semibold text-gray-300">Properties</h5>
                      <div className="space-y-1 text-xs">
                        <div className="flex justify-between">
                          <span className="text-gray-400">ID:</span>
                          <span className="text-gray-300 font-mono">{selectedNode.id}</span>
                        </div>
                        {selectedNode.metadata &&
                          Object.entries(selectedNode.metadata).map(([key, value]) => (
                            <div key={key} className="flex justify-between">
                              <span className="text-gray-400 capitalize">
                                {key.replace(/_/g, ' ')}:
                              </span>
                              <span className="text-gray-300">{String(value)}</span>
                            </div>
                          ))}
                      </div>
                    </div>

                    {/* Connected Nodes */}
                    <div className="space-y-2">
                      <h5 className="text-sm font-semibold text-gray-300">
                        Connected Nodes ({getConnectedNodes(selectedNode.id).length})
                      </h5>
                      <div className="max-h-40 overflow-y-auto space-y-1">
                        {getConnectedNodes(selectedNode.id).map((node) => (
                          <div
                            key={node.id}
                            onClick={() => setSelectedNode(node)}
                            className="flex items-center space-x-2 p-2 hover:bg-white/10 rounded cursor-pointer text-xs transition-colors"
                          >
                            <span
                              className="w-2 h-2 rounded-full flex-shrink-0"
                              style={{ backgroundColor: nodeTypeConfig[node.type].color }}
                            />
                            <span className="text-gray-300 truncate">{node.label}</span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Actions */}
                    <div className="space-y-2 pt-4 border-t border-white/10">
                      <button
                        onClick={handleExpandNode}
                        className="btn-glass-secondary w-full text-sm py-2"
                      >
                        Expand Node
                      </button>
                      <button
                        onClick={() => {
                          const newLabel = prompt('Enter new label:', selectedNode.label);
                          if (newLabel) {
                            setGraphData((prev) => ({
                              ...prev,
                              nodes: prev.nodes.map((n) =>
                                n.id === selectedNode.id ? { ...n, label: newLabel } : n
                              ),
                            }));
                            setSelectedNode({ ...selectedNode, label: newLabel });
                          }
                        }}
                        className="btn-glass-secondary w-full text-sm py-2"
                      >
                        Edit Label
                      </button>
                      <button
                        onClick={handleDeleteNode}
                        className="btn-glass-secondary w-full text-sm py-2 text-red-400 hover:text-red-300"
                      >
                        Delete Node
                      </button>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="no-selection"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    className="text-center py-12"
                  >
                    <div className="text-4xl mb-4">👆</div>
                    <p className="text-sm text-gray-400">
                      Select a node to view its details
                    </p>
                  </motion.div>
                )}
              </AnimatePresence>
            </GlassCard>

            {/* Legend */}
            <GlassCard className="p-6 mt-6">
              <h3 className="text-lg font-semibold text-white mb-4">Legend</h3>

              <div className="space-y-4">
                {/* Node Types */}
                <div>
                  <h5 className="text-xs font-semibold text-gray-400 mb-2 uppercase">
                    Node Types
                  </h5>
                  <div className="space-y-2">
                    {Object.entries(nodeTypeConfig).map(([type, config]) => (
                      <div key={type} className="flex items-center space-x-3">
                        <div
                          className="w-4 h-4 rounded-full"
                          style={{ backgroundColor: config.color }}
                        />
                        <span className="text-xs text-gray-300">{config.label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Edge Types */}
                <div>
                  <h5 className="text-xs font-semibold text-gray-400 mb-2 uppercase">
                    Edge Types
                  </h5>
                  <div className="space-y-2">
                    {Object.entries(edgeTypeConfig).map(([type, config]) => (
                      <div key={type} className="flex items-center space-x-3">
                        <svg width="24" height="4">
                          <line
                            x1="0"
                            y1="2"
                            x2="24"
                            y2="2"
                            stroke={config.color}
                            strokeWidth="2"
                            strokeDasharray={config.dashArray}
                          />
                        </svg>
                        <span className="text-xs text-gray-300">{config.label}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </GlassCard>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Graph;
