import React, { useRef, useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import GlassCard from '../Common/GlassCard';
import type { GraphData, GraphNode } from '../../types';

interface GraphVisualizationProps {
  data: GraphData;
  onNodeClick?: (node: GraphNode) => void;
}

const GraphVisualization: React.FC<GraphVisualizationProps> = ({ data, onNodeClick }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
  const [hoveredNode, setHoveredNode] = useState<GraphNode | null>(null);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState({ x: 0, y: 0 });

  // Node positions (simplified force-directed layout)
  const [nodePositions, setNodePositions] = useState<Map<string, { x: number; y: number }>>(new Map());

  useEffect(() => {
    if (!canvasRef.current || data.nodes.length === 0) return;

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Initialize node positions if not set
    if (nodePositions.size === 0) {
      const positions = new Map<string, { x: number; y: number }>();
      const centerX = canvas.width / 2;
      const centerY = canvas.height / 2;
      const radius = Math.min(canvas.width, canvas.height) * 0.3;

      data.nodes.forEach((node, index) => {
        const angle = (2 * Math.PI * index) / data.nodes.length;
        positions.set(node.id, {
          x: centerX + radius * Math.cos(angle),
          y: centerY + radius * Math.sin(angle),
        });
      });

      setNodePositions(positions);
    }

    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Apply transformations
    ctx.save();
    ctx.translate(pan.x, pan.y);
    ctx.scale(zoom, zoom);

    // Draw edges
    data.edges.forEach((edge) => {
      const source = nodePositions.get(edge.source);
      const target = nodePositions.get(edge.target);

      if (!source || !target) return;

      ctx.beginPath();
      ctx.moveTo(source.x, source.y);
      ctx.lineTo(target.x, target.y);

      // Edge color based on type
      const edgeColors = {
        citation: 'rgba(96, 165, 250, 0.5)',
        similarity: 'rgba(168, 85, 247, 0.5)',
        'co-authorship': 'rgba(236, 72, 153, 0.5)',
      };
      ctx.strokeStyle = edgeColors[edge.type] || 'rgba(255, 255, 255, 0.2)';
      ctx.lineWidth = edge.weight * 2;
      ctx.stroke();
    });

    // Draw nodes
    data.nodes.forEach((node) => {
      const pos = nodePositions.get(node.id);
      if (!pos) return;

      const isSelected = selectedNode?.id === node.id;
      const isHovered = hoveredNode?.id === node.id;

      // Node size based on weight
      const baseSize = 8;
      const size = baseSize + (node.weight || 0) * 5;

      // Node color based on type
      const nodeColors = {
        paper: '#60a5fa',
        concept: '#a855f7',
        author: '#ec4899',
      };

      // Draw node circle
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, size, 0, 2 * Math.PI);
      ctx.fillStyle = nodeColors[node.type];

      if (isSelected || isHovered) {
        ctx.shadowColor = nodeColors[node.type];
        ctx.shadowBlur = 15;
      }

      ctx.fill();
      ctx.shadowBlur = 0;

      // Draw node border
      if (isSelected || isHovered) {
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Draw label
      if (isSelected || isHovered || zoom > 1.5) {
        ctx.font = '12px sans-serif';
        ctx.fillStyle = 'white';
        ctx.textAlign = 'center';
        ctx.fillText(
          node.label.length > 20 ? node.label.substring(0, 20) + '...' : node.label,
          pos.x,
          pos.y + size + 15
        );
      }
    });

    ctx.restore();
  }, [data, nodePositions, selectedNode, hoveredNode, zoom, pan]);

  const handleCanvasClick = (e: React.MouseEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - pan.x) / zoom;
    const y = (e.clientY - rect.top - pan.y) / zoom;

    // Check if clicked on a node
    let clickedNode: GraphNode | null = null;
    data.nodes.forEach((node) => {
      const pos = nodePositions.get(node.id);
      if (!pos) return;

      const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
      const size = 8 + (node.weight || 0) * 5;

      if (distance <= size) {
        clickedNode = node;
      }
    });

    setSelectedNode(clickedNode);
    if (clickedNode && onNodeClick) {
      onNodeClick(clickedNode);
    }
  };

  const handleMouseMove = (e: React.MouseEvent<HTMLCanvasElement>) => {
    if (isDragging) {
      setPan({
        x: e.clientX - dragStart.x,
        y: e.clientY - dragStart.y,
      });
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left - pan.x) / zoom;
    const y = (e.clientY - rect.top - pan.y) / zoom;

    let hoveredNode: GraphNode | null = null;
    data.nodes.forEach((node) => {
      const pos = nodePositions.get(node.id);
      if (!pos) return;

      const distance = Math.sqrt((x - pos.x) ** 2 + (y - pos.y) ** 2);
      const size = 8 + (node.weight || 0) * 5;

      if (distance <= size) {
        hoveredNode = node;
      }
    });

    setHoveredNode(hoveredNode);
  };

  const handleWheel = (e: React.WheelEvent<HTMLCanvasElement>) => {
    e.preventDefault();
    const newZoom = zoom + (e.deltaY > 0 ? -0.1 : 0.1);
    setZoom(Math.max(0.5, Math.min(3, newZoom)));
  };

  return (
    <GlassCard className="p-6">
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-2xl font-bold text-white mb-1">Knowledge Graph</h2>
            <p className="text-white/60 text-sm">
              {data.nodes.length} nodes, {data.edges.length} edges
            </p>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-2">
            <button
              onClick={() => setZoom(Math.min(3, zoom + 0.2))}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
              aria-label="Zoom in"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM10 7v6m3-3H7" />
              </svg>
            </button>
            <button
              onClick={() => setZoom(Math.max(0.5, zoom - 0.2))}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
              aria-label="Zoom out"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0zM13 10H7" />
              </svg>
            </button>
            <button
              onClick={() => {
                setZoom(1);
                setPan({ x: 0, y: 0 });
              }}
              className="p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-colors"
              aria-label="Reset view"
            >
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
            </button>
          </div>
        </div>

        {/* Canvas */}
        <div className="relative">
          <canvas
            ref={canvasRef}
            width={800}
            height={600}
            className="w-full h-[600px] rounded-xl bg-black/20 cursor-move"
            onClick={handleCanvasClick}
            onMouseMove={handleMouseMove}
            onMouseDown={(e) => {
              setIsDragging(true);
              setDragStart({ x: e.clientX - pan.x, y: e.clientY - pan.y });
            }}
            onMouseUp={() => setIsDragging(false)}
            onMouseLeave={() => setIsDragging(false)}
            onWheel={handleWheel}
          />

          {/* Legend */}
          <motion.div
            className="absolute top-4 right-4 backdrop-blur-xl bg-white/10 border border-white/20 rounded-xl p-4 space-y-2"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
          >
            <p className="text-sm font-semibold text-white mb-2">Node Types</p>
            {[
              { type: 'paper', color: '#60a5fa', label: 'Paper' },
              { type: 'concept', color: '#a855f7', label: 'Concept' },
              { type: 'author', color: '#ec4899', label: 'Author' },
            ].map((item) => (
              <div key={item.type} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: item.color }}
                />
                <span className="text-xs text-white/70">{item.label}</span>
              </div>
            ))}
          </motion.div>
        </div>

        {/* Selected Node Info */}
        {selectedNode && (
          <motion.div
            className="p-4 rounded-xl bg-white/5 border border-white/10"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
          >
            <div className="flex items-start justify-between">
              <div>
                <p className="text-sm text-white/60 mb-1">Selected Node</p>
                <h3 className="text-lg font-semibold text-white">{selectedNode.label}</h3>
                <p className="text-sm text-white/60 mt-1">Type: {selectedNode.type}</p>
              </div>
              <button
                onClick={() => setSelectedNode(null)}
                className="p-1 rounded-lg hover:bg-white/10 transition-colors"
                aria-label="Close"
              >
                <svg className="w-5 h-5 text-white/60" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
          </motion.div>
        )}
      </div>
    </GlassCard>
  );
};

export default GraphVisualization;
