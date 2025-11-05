import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';
import Navigation from './components/Layout/Navigation';
import Sidebar from './components/Layout/Sidebar';

// Import all page components
import Home from './pages/Home';
import Collect from './pages/Collect';
import Ask from './pages/Ask';
import Graph from './pages/Graph';
import Vector from './pages/Vector';
import Upload from './pages/Upload';
import Sessions from './pages/Sessions';

// Animated background orbs component
const AnimatedBackground = () => {
  return (
    <div className="animated-background">
      <motion.div
        className="orb orb-1"
        animate={{
          x: [0, 30, -20, 0],
          y: [0, -50, 20, 0],
          scale: [1, 1.1, 0.9, 1],
        }}
        transition={{
          duration: 7,
          repeat: Infinity,
          ease: "easeInOut",
        }}
      />
      <motion.div
        className="orb orb-2"
        animate={{
          x: [0, -40, 30, 0],
          y: [0, 40, -30, 0],
          scale: [1, 0.9, 1.1, 1],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 1,
        }}
      />
      <motion.div
        className="orb orb-3"
        animate={{
          x: [0, 25, -35, 0],
          y: [0, -30, 40, 0],
          scale: [1, 1.05, 0.95, 1],
        }}
        transition={{
          duration: 9,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 2,
        }}
      />
      <motion.div
        className="orb orb-4"
        animate={{
          x: [0, -30, 20, 0],
          y: [0, 35, -25, 0],
          scale: [1, 0.95, 1.05, 1],
        }}
        transition={{
          duration: 6,
          repeat: Infinity,
          ease: "easeInOut",
          delay: 3,
        }}
      />
    </div>
  );
};

// Layout component with glassmorphism styling
const Layout = ({ children }: { children: React.ReactNode }) => {
  const [isDark, setIsDark] = useState(true);

  const handleThemeToggle = () => {
    setIsDark(!isDark);
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      {/* Animated Background with Floating Orbs */}
      <AnimatedBackground />

      {/* Grid Pattern Overlay */}
      <div className="grid-pattern" />

      {/* Glassmorphic Navigation Bar */}
      <Navigation onThemeToggle={handleThemeToggle} isDark={isDark} />

      {/* Responsive Sidebar */}
      <Sidebar />

      {/* Main Content Area */}
      <motion.main
        className="pt-16 lg:pl-64 pb-16 lg:pb-0"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="relative z-10">
          {children}
        </div>
      </motion.main>
    </div>
  );
};

// Main App Component
function App() {
  return (
    <Router>
      <Layout>
        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/collect" element={<Collect />} />
            <Route path="/ask" element={<Ask />} />
            <Route path="/graph" element={<Graph />} />
            <Route path="/vector" element={<Vector />} />
            <Route path="/upload" element={<Upload />} />
            <Route path="/sessions" element={<Sessions />} />
          </Routes>
        </AnimatePresence>
      </Layout>
    </Router>
  );
}

export default App;
