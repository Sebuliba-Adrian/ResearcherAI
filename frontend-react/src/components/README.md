# ResearcherAI Frontend Components

This directory contains all React TypeScript components for the ResearcherAI frontend application. All components feature glassmorphism design, responsive layouts, and smooth animations using Framer Motion.

## Component Structure

```
components/
├── Common/              # Reusable UI components
│   ├── GlassCard.tsx           # Glassmorphism card container
│   ├── LoadingSpinner.tsx      # Loading animation
│   └── Toast.tsx               # Toast notification system
│
├── Layout/              # Layout components
│   ├── Navigation.tsx          # Top navigation bar with theme toggle
│   ├── Sidebar.tsx             # Collapsible sidebar navigation
│   └── Footer.tsx              # Footer with links and social media
│
├── Hero/                # Landing page components
│   └── Hero.tsx                # Hero section with animated background
│
├── DataCollection/      # Data collection components
│   ├── CollectForm.tsx         # Form to collect research papers
│   └── SourceSelector.tsx      # Multi-source selector UI
│
├── Query/               # Query components
│   ├── AskQuestion.tsx         # Question input interface
│   └── ResponseDisplay.tsx     # AI response display with markdown
│
├── Graph/               # Knowledge graph components
│   └── GraphVisualization.tsx  # Interactive graph visualization
│
├── Vector/              # Vector search components
│   └── VectorSearch.tsx        # Vector similarity search interface
│
├── Upload/              # File upload components
│   └── FileUpload.tsx          # Drag-and-drop file upload
│
└── Sessions/            # Session management components
    └── SessionManager.tsx      # Session list and management
```

## Common Components

### GlassCard
Reusable glassmorphism card component with backdrop blur and transparent backgrounds.

**Props:**
- `children: React.ReactNode` - Content to display
- `className?: string` - Additional CSS classes
- `hover?: boolean` - Enable hover effects
- `onClick?: () => void` - Click handler

**Usage:**
```tsx
import { GlassCard } from '@/components';

<GlassCard hover onClick={handleClick}>
  <p>Card content</p>
</GlassCard>
```

### LoadingSpinner
Animated loading spinner with optional message.

**Props:**
- `size?: 'sm' | 'md' | 'lg'` - Spinner size (default: 'md')
- `message?: string` - Optional loading message

### Toast
Toast notification system with auto-dismiss and animations.

**Props:**
- `toasts: ToastMessage[]` - Array of toast messages
- `removeToast: (id: string) => void` - Function to remove toast

## Layout Components

### Navigation
Responsive navigation bar with glassmorphism styling.

**Features:**
- Logo and brand name
- Navigation links with active state
- Theme toggle button
- Mobile hamburger menu

### Sidebar
Collapsible sidebar with navigation items.

**Features:**
- Desktop sidebar (collapsible)
- Mobile bottom navigation
- Active route highlighting
- Smooth transitions

### Footer
Application footer with links and information.

**Features:**
- Brand section
- Quick links
- Social media links
- Copyright information

## Feature Components

### DataCollection/CollectForm
Form for collecting research papers from multiple sources.

**Features:**
- Search query input
- Source selection (arXiv, Semantic Scholar, etc.)
- Max results slider
- Advanced date filters
- Example queries

### DataCollection/SourceSelector
Multi-source selection interface.

**Features:**
- Visual source cards
- Multiple selection
- Select all/deselect all
- Source descriptions

### Query/AskQuestion
Question input interface for querying research papers.

**Features:**
- Text area for questions
- Character count
- Example questions
- Submit button with loading state

### Query/ResponseDisplay
Displays AI-generated responses with sources.

**Features:**
- Markdown rendering
- Confidence score
- Source citations
- Expandable sources
- Response history

### Graph/GraphVisualization
Interactive knowledge graph visualization.

**Features:**
- Canvas-based rendering
- Node types (paper, concept, author)
- Edge types (citation, similarity, co-authorship)
- Zoom and pan controls
- Node selection
- Interactive legend

### Vector/VectorSearch
Vector similarity search interface.

**Features:**
- Search input
- Result limit slider
- Similarity scores
- Result cards with paper details
- Semantic search explanation

### Upload/FileUpload
Drag-and-drop file upload interface.

**Features:**
- Drag-and-drop zone
- File browser
- Upload progress tracking
- File type validation
- Multiple file support
- File size display

### Sessions/SessionManager
Session management interface.

**Features:**
- Session grid/list view
- Create new session
- Rename sessions
- Delete sessions with confirmation
- Active session indicator
- Session statistics (paper count, dates)

## TypeScript Types

All components use TypeScript interfaces defined in `/src/types/index.ts`:

- `Paper` - Research paper data
- `DataSource` - Data source type
- `Session` - Research session
- `QueryResponse` - AI query response
- `GraphNode` / `GraphEdge` - Graph data structures
- `VectorSearchResult` - Vector search result
- `ToastMessage` - Toast notification
- `CollectFormData` - Collection form data
- `UploadedFile` - Uploaded file data

## Styling

All components use:
- **Tailwind CSS** for utility classes
- **Glassmorphism** design (backdrop-blur, semi-transparent backgrounds)
- **Framer Motion** for animations
- **Responsive** breakpoints (sm, md, lg, xl)
- **Dark mode** support

## Accessibility

All components include:
- Proper ARIA labels
- Keyboard navigation support
- Focus states
- Screen reader friendly
- Semantic HTML

## Animation Guidelines

Components use Framer Motion with consistent patterns:
- `initial` - Starting state (opacity: 0, y: 20)
- `animate` - End state (opacity: 1, y: 0)
- `exit` - Exit animation
- `whileHover` - Hover effects (scale: 1.02-1.05)
- `whileTap` - Tap effects (scale: 0.95-0.98)

## Import Pattern

Use barrel exports for clean imports:

```tsx
// Import from category
import { GlassCard, LoadingSpinner } from '@/components/Common';

// Import from root
import { GlassCard, Hero, CollectForm } from '@/components';
```

## Component Development Guidelines

1. **TypeScript First** - Always use TypeScript with proper types
2. **Props Interface** - Define clear interface for all props
3. **Accessibility** - Include ARIA labels and keyboard support
4. **Responsive** - Test on mobile, tablet, and desktop
5. **Glassmorphism** - Use consistent glass styling
6. **Animations** - Use Framer Motion for smooth transitions
7. **Error Handling** - Handle loading and error states
8. **Documentation** - Add JSDoc comments for complex logic

## Example Component Template

```tsx
import React from 'react';
import { motion } from 'framer-motion';
import GlassCard from '../Common/GlassCard';

interface MyComponentProps {
  title: string;
  onAction: () => void;
}

const MyComponent: React.FC<MyComponentProps> = ({ title, onAction }) => {
  return (
    <GlassCard className="p-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.3 }}
      >
        <h2 className="text-2xl font-bold text-white mb-4">{title}</h2>
        <motion.button
          onClick={onAction}
          className="px-4 py-2 bg-gradient-to-r from-blue-500 to-purple-600 text-white rounded-xl"
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
        >
          Action Button
        </motion.button>
      </motion.div>
    </GlassCard>
  );
};

export default MyComponent;
```

## Testing

Components should be tested for:
- Rendering with different props
- User interactions (clicks, inputs)
- Accessibility
- Responsive behavior
- Error states

## Contributing

When adding new components:
1. Follow the existing structure
2. Add to appropriate category directory
3. Export from category index.ts
4. Update this README
5. Include TypeScript types
6. Add accessibility features
7. Test responsiveness
