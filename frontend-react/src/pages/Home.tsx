import { motion } from 'framer-motion';
import {
  Database,
  Network,
  Search,
  Brain,
  Workflow,
  Clock,
  CheckCircle2,
  Target,
  Zap,
  ArrowRight
} from 'lucide-react';

// Floating particle component
const FloatingParticle = ({ delay = 0, duration = 20 }: { delay?: number; duration?: number }) => {
  return (
    <motion.div
      className="absolute w-2 h-2 bg-neon-purple rounded-full"
      initial={{
        x: Math.random() * window.innerWidth,
        y: window.innerHeight + 100,
        opacity: 0
      }}
      animate={{
        y: -100,
        opacity: [0, 1, 1, 0],
        scale: [0, 1.5, 1, 0],
      }}
      transition={{
        duration,
        delay,
        repeat: Infinity,
        ease: "linear",
      }}
    />
  );
};

// Animated border component
const AnimatedBorder = () => {
  return (
    <div className="absolute inset-0 rounded-glass overflow-hidden">
      <motion.div
        className="absolute inset-0 opacity-50"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(168, 85, 247, 0.8), transparent)',
        }}
        animate={{
          x: ['-200%', '200%'],
        }}
        transition={{
          duration: 3,
          repeat: Infinity,
          ease: "linear",
        }}
      />
      <div className="absolute inset-0 rounded-glass border-2 border-transparent bg-gradient-to-r from-cyber-500 via-neon-purple to-neon-pink bg-clip-border"
           style={{
             WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
             WebkitMaskComposite: 'xor',
             maskComposite: 'exclude',
             padding: '2px'
           }}
      />
    </div>
  );
};

const Home = () => {
  const agents = [
    {
      icon: Database,
      title: 'Data Collector',
      description: 'Aggregates information from 7 diverse sources including ArXiv, Wikipedia, and web scraping',
      color: 'from-blue-500 to-cyan-500',
      stats: '7 Sources'
    },
    {
      icon: Network,
      title: 'Knowledge Graph',
      description: 'Neo4j-powered graph database for complex relationship mapping and semantic connections',
      color: 'from-purple-500 to-pink-500',
      stats: 'Neo4j'
    },
    {
      icon: Search,
      title: 'Vector Search',
      description: 'Qdrant vector database enabling lightning-fast semantic search and similarity matching',
      color: 'from-cyan-500 to-blue-500',
      stats: 'Qdrant'
    },
    {
      icon: Brain,
      title: 'Reasoning Agent',
      description: 'Advanced AI reasoning with multi-step logic, inference chains, and contextual analysis',
      color: 'from-pink-500 to-rose-500',
      stats: 'AI-Powered'
    },
    {
      icon: Workflow,
      title: 'Orchestrator',
      description: 'Intelligent workflow coordination managing agent interactions and task distribution',
      color: 'from-violet-500 to-purple-500',
      stats: 'Multi-Agent'
    },
    {
      icon: Clock,
      title: 'Scheduler',
      description: 'Automated task scheduling with priority management and resource optimization',
      color: 'from-indigo-500 to-blue-500',
      stats: 'Automated'
    },
  ];

  const stats = [
    {
      icon: CheckCircle2,
      value: '96.60%',
      label: 'Test Coverage',
      color: 'text-green-400'
    },
    {
      icon: Target,
      value: '291',
      label: 'Tests Passing',
      color: 'text-blue-400'
    },
    {
      icon: Database,
      value: '7',
      label: 'Data Sources',
      color: 'text-purple-400'
    },
    {
      icon: Zap,
      value: '6',
      label: 'Specialized Agents',
      color: 'text-cyan-400'
    },
  ];

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.1,
      },
    },
  };

  const itemVariants = {
    hidden: { y: 20, opacity: 0 },
    visible: {
      y: 0,
      opacity: 1,
      transition: {
        duration: 0.5,
      },
    },
  };

  return (
    <div className="min-h-screen relative overflow-hidden">
      {/* Animated background with floating orbs */}
      <div className="fixed inset-0 -z-10">
        <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-slate-900 to-slate-950" />

        {/* Floating orbs */}
        <motion.div
          className="absolute top-1/4 left-1/4 w-96 h-96 bg-purple-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.2, 1],
            x: [0, 50, 0],
            y: [0, 30, 0],
          }}
          transition={{
            duration: 8,
            repeat: Infinity,
            ease: "easeInOut",
          }}
        />
        <motion.div
          className="absolute top-1/2 right-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.3, 1],
            x: [0, -30, 0],
            y: [0, 50, 0],
          }}
          transition={{
            duration: 10,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 1,
          }}
        />
        <motion.div
          className="absolute bottom-1/4 left-1/2 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl"
          animate={{
            scale: [1, 1.1, 1],
            x: [0, 40, 0],
            y: [0, -40, 0],
          }}
          transition={{
            duration: 12,
            repeat: Infinity,
            ease: "easeInOut",
            delay: 2,
          }}
        />

        {/* Floating particles */}
        {[...Array(20)].map((_, i) => (
          <FloatingParticle key={i} delay={i * 0.5} duration={15 + Math.random() * 10} />
        ))}

        {/* Grid pattern */}
        <div className="absolute inset-0 bg-[url('data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNjAiIGhlaWdodD0iNjAiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+PGRlZnM+PHBhdHRlcm4gaWQ9ImdyaWQiIHdpZHRoPSI2MCIgaGVpZ2h0PSI2MCIgcGF0dGVyblVuaXRzPSJ1c2VyU3BhY2VPblVzZSI+PHBhdGggZD0iTSAxMCAwIEwgMCAwIDAgMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0icmdiYSgyNTUsMjU1LDI1NSwwLjAzKSIgc3Ryb2tlLXdpZHRoPSIxIi8+PC9wYXR0ZXJuPjwvZGVmcz48cmVjdCB3aWR0aD0iMTAwJSIgaGVpZ2h0PSIxMDAlIiBmaWxsPSJ1cmwoI2dyaWQpIi8+PC9zdmc+')]" />
      </div>

      <div className="relative z-10">
        {/* Hero Section */}
        <section className="container mx-auto px-4 pt-20 pb-32">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            className="max-w-5xl mx-auto"
          >
            {/* Hero Card with Animated Border */}
            <div className="relative">
              <AnimatedBorder />

              <motion.div
                className="relative glass-dark p-12 md:p-16 rounded-glass overflow-hidden"
                whileHover={{ scale: 1.02 }}
                transition={{ duration: 0.3 }}
              >
                {/* Background gradient overlay */}
                <div className="absolute inset-0 bg-gradient-to-br from-purple-500/10 via-transparent to-blue-500/10" />

                <div className="relative z-10 text-center">
                  {/* Version badge */}
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
                    className="inline-block mb-6"
                  >
                    <span className="glass px-4 py-2 rounded-full text-sm font-semibold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-400">
                      Latest Release
                    </span>
                  </motion.div>

                  {/* Main title */}
                  <motion.h1
                    className="text-6xl md:text-8xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-white via-purple-200 to-cyan-200"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3, duration: 0.8 }}
                  >
                    ResearcherAI v2.0
                  </motion.h1>

                  {/* Subtitle */}
                  <motion.p
                    className="text-xl md:text-2xl text-gray-300 mb-10 max-w-3xl mx-auto leading-relaxed"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5, duration: 0.8 }}
                  >
                    Multi-Agent RAG System for Advanced Research
                  </motion.p>

                  <motion.p
                    className="text-base md:text-lg text-gray-400 mb-12 max-w-2xl mx-auto"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.6, duration: 0.8 }}
                  >
                    Harness the power of 6 specialized AI agents working in harmony to deliver
                    comprehensive research insights from multiple data sources
                  </motion.p>

                  {/* CTA Buttons */}
                  <motion.div
                    className="flex flex-col sm:flex-row gap-4 justify-center items-center"
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.7, duration: 0.8 }}
                  >
                    <motion.button
                      className="btn-glass-primary group flex items-center gap-2 text-lg px-8 py-4"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      Start Research
                      <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                    </motion.button>

                    <motion.button
                      className="btn-glass-secondary group flex items-center gap-2 text-lg px-8 py-4"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      View Documentation
                    </motion.button>
                  </motion.div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </section>

        {/* Stats Section */}
        <section className="container mx-auto px-4 pb-20">
          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 max-w-6xl mx-auto"
          >
            {stats.map((stat, index) => {
              const Icon = stat.icon;
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="relative group"
                >
                  <div className="glass-card text-center hover:scale-105 transition-transform duration-300">
                    <div className="absolute inset-0 bg-gradient-to-br from-purple-500/5 to-blue-500/5 rounded-glass opacity-0 group-hover:opacity-100 transition-opacity" />
                    <div className="relative">
                      <Icon className={`w-12 h-12 mx-auto mb-4 ${stat.color}`} />
                      <div className={`text-4xl font-bold mb-2 ${stat.color}`}>
                        {stat.value}
                      </div>
                      <div className="text-gray-400 text-sm uppercase tracking-wide">
                        {stat.label}
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </section>

        {/* Features Grid Section */}
        <section className="container mx-auto px-4 pb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-5xl md:text-6xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-400">
              Agent Capabilities
            </h2>
            <p className="text-xl text-gray-400 max-w-3xl mx-auto">
              Six specialized agents working together to deliver comprehensive research results
            </p>
          </motion.div>

          <motion.div
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, margin: "-100px" }}
            className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 max-w-7xl mx-auto"
          >
            {agents.map((agent, index) => {
              const Icon = agent.icon;
              return (
                <motion.div
                  key={index}
                  variants={itemVariants}
                  className="relative group"
                  whileHover={{ y: -10 }}
                  transition={{ duration: 0.3 }}
                >
                  {/* Card */}
                  <div className="glass-card-dark h-full relative overflow-hidden">
                    {/* Gradient overlay on hover */}
                    <div className={`absolute inset-0 bg-gradient-to-br ${agent.color} opacity-0 group-hover:opacity-10 transition-opacity duration-300`} />

                    {/* Content */}
                    <div className="relative">
                      {/* Icon with gradient background */}
                      <div className="mb-4">
                        <div className={`inline-flex p-4 rounded-2xl bg-gradient-to-br ${agent.color} bg-opacity-10`}>
                          <Icon className="w-8 h-8 text-white" />
                        </div>
                      </div>

                      {/* Stats badge */}
                      <div className="mb-3">
                        <span className="text-xs font-semibold px-3 py-1 rounded-full glass text-gray-300">
                          {agent.stats}
                        </span>
                      </div>

                      {/* Title */}
                      <h3 className="text-2xl font-bold mb-3 text-white group-hover:text-transparent group-hover:bg-clip-text group-hover:bg-gradient-to-r group-hover:from-white group-hover:to-gray-300 transition-all">
                        {agent.title}
                      </h3>

                      {/* Description */}
                      <p className="text-gray-400 leading-relaxed">
                        {agent.description}
                      </p>

                      {/* Decorative corner element */}
                      <div className="absolute top-0 right-0 w-20 h-20 opacity-0 group-hover:opacity-100 transition-opacity">
                        <div className={`absolute inset-0 bg-gradient-to-br ${agent.color} blur-2xl`} />
                      </div>
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </motion.div>
        </section>

        {/* Call to Action Section */}
        <section className="container mx-auto px-4 pb-32">
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            whileInView={{ opacity: 1, scale: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 0.6 }}
            className="max-w-4xl mx-auto"
          >
            <div className="relative glass-dark p-12 rounded-glass text-center overflow-hidden">
              {/* Background effects */}
              <div className="absolute inset-0 bg-gradient-to-r from-purple-500/10 via-blue-500/10 to-cyan-500/10" />
              <motion.div
                className="absolute inset-0"
                animate={{
                  background: [
                    'radial-gradient(circle at 0% 0%, rgba(168, 85, 247, 0.1) 0%, transparent 50%)',
                    'radial-gradient(circle at 100% 100%, rgba(59, 130, 246, 0.1) 0%, transparent 50%)',
                    'radial-gradient(circle at 0% 100%, rgba(6, 182, 212, 0.1) 0%, transparent 50%)',
                    'radial-gradient(circle at 0% 0%, rgba(168, 85, 247, 0.1) 0%, transparent 50%)',
                  ],
                }}
                transition={{
                  duration: 10,
                  repeat: Infinity,
                  ease: "linear",
                }}
              />

              <div className="relative z-10">
                <h2 className="text-4xl md:text-5xl font-bold mb-6 text-transparent bg-clip-text bg-gradient-to-r from-white to-gray-300">
                  Ready to Transform Your Research?
                </h2>
                <p className="text-xl text-gray-400 mb-8 max-w-2xl mx-auto">
                  Experience the power of AI-driven research with our multi-agent system
                </p>
                <motion.button
                  className="btn-glass-primary text-lg px-10 py-4 inline-flex items-center gap-2"
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                >
                  Get Started Now
                  <ArrowRight className="w-5 h-5" />
                </motion.button>
              </div>
            </div>
          </motion.div>
        </section>
      </div>
    </div>
  );
};

export default Home;
