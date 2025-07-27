# 🎭 Multi-Session Guide - Manage Multiple Chats

## ✅ New Feature Added!

**You can now have multiple independent chat sessions and switch between them at will!**

Think of sessions like browser tabs - each one has its own:
- ✅ Conversation history
- ✅ Document source  
- ✅ Knowledge graph
- ✅ Complete context

---

## 🚀 Quick Start

### Create Your First Named Session

```bash
python3 full_memory_rag.py sample_knowledge.txt my_research
```

This creates a session called "my_research" from the document.

### Create a Second Session

```bash
# In the interactive session:
new python_study sample_knowledge.txt
```

### Switch Between Sessions

```bash
# List all sessions
sessions

# Switch to a different session
switch my_research
```

### Resume Last Session

```bash
python3 full_memory_rag.py resume
```

---

## 📊 How It Works

### Session Storage

Sessions are stored in the `sessions/` directory:
```
sessions/
  ├── my_research.pkl        (4.2 KB)
  ├── python_study.pkl        (4.1 KB)
  └── crispr_notes.pkl       (4.3 KB)
```

Each session file contains:
- All document chunks
- Complete knowledge graph
- Full conversation history
- Source document path
- Timestamp

---

## 🎮 Available Commands

| Command | What It Does | Example |
|---------|--------------|---------|
| `sessions` | List all sessions | Shows all available sessions |
| `new <name> [doc]` | Create new session | `new research_paper paper.pdf` |
| `switch <name>` | Switch to session | `switch my_research` |
| `delete <name>` | Delete session | `delete old_session` |
| `stats` | Show current session info | Shows current session details |

---

## 🧪 Complete Example

### Session 1: CRISPR Research

```bash
$ python3 full_memory_rag.py sample_knowledge.txt crispr_research

📌 Current session: 'crispr_research'

👤 You: Who developed CRISPR?
🤖 Agent: Jennifer Doudna and Emmanuelle Charpentier developed the CRISPR-Cas9 system.

👤 You: Where does she work?
🤖 Agent: Jennifer Doudna works at UC Berkeley.

👤 You: save
💾 Session 'crispr_research' saved!
```

### Create Session 2: Python Info

```bash
👤 You: new python_info sample_knowledge.txt

✨ Created new session: 'python_info'
📘 Reading: sample_knowledge.txt
✅ System Ready!

👤 You: Who created Python?
🤖 Agent: Guido van Rossum created Python.

👤 You: What is it used for?
🤖 Agent: Python is used for AI and machine learning applications.
```

### Switch Back to Session 1

```bash
👤 You: switch crispr_research

📂 Session 'crispr_research' loaded!
✅ Switched to session: 'crispr_research'

👤 You: memory

💾 Conversation History (2 turns):
1. Q: Who developed CRISPR?
   A: Jennifer Doudna and Emmanuelle Charpentier...
2. Q: Where does she work?
   A: Jennifer Doudna works at UC Berkeley...

👤 You: What is her full name?
🤖 Agent: Her full name is Jennifer Doudna.
         ☝️ Remembers context from earlier conversation!
```

### List All Sessions

```bash
👤 You: sessions

📂 Available Sessions (2):

→ crispr_research
   Source: sample_knowledge.txt
   Conversations: 3
   Entities: 46
   Last saved: 2025-10-25 19:35:35

  python_info
   Source: sample_knowledge.txt
   Conversations: 2
   Entities: 46
   Last saved: 2025-10-25 19:36:12
```

---

## 🎯 Use Cases

### 1. Multiple Documents

Process different documents in separate sessions:

```bash
# Session for research paper
python3 full_memory_rag.py research_paper.pdf paper_analysis

# Create new session for different paper
new competitor_analysis competitor.pdf

# Switch between them anytime
switch paper_analysis
switch competitor_analysis
```

### 2. Different Topics from Same Document

Explore different aspects separately:

```bash
# Session focused on CRISPR
python3 full_memory_rag.py sample_knowledge.txt crispr_focus

# New session focused on Python
new python_focus sample_knowledge.txt
```

### 3. Experimental vs Production

Keep experimental questions separate:

```bash
# Production analysis
python3 full_memory_rag.py data.txt production_analysis

# Experimental questions
new experimental_tests data.txt
```

### 4. Team Collaboration

Each team member has their own session:

```bash
# Alice's session
python3 full_memory_rag.py project.pdf alice_notes

# Bob's session  
new bob_notes project.pdf

# Share sessions by sharing .pkl files
cp sessions/alice_notes.pkl /shared/folder/
```

---

## 📁 Session Management

### Viewing Sessions

```bash
# From command line
ls -lh sessions/

# From interactive mode
sessions
```

### Backing Up Sessions

```bash
# Backup all sessions
cp -r sessions/ sessions_backup/

# Backup specific session
cp sessions/important_research.pkl backups/
```

### Sharing Sessions

```bash
# Send session to colleague
scp sessions/my_research.pkl colleague@server:/path/

# They can then:
cp my_research.pkl sessions/
python3 full_memory_rag.py resume
switch my_research
```

### Deleting Sessions

```bash
# From interactive mode (with confirmation)
delete old_session
⚠️  Delete session 'old_session'? (yes/no): yes
🗑️  Deleted session: 'old_session'

# Or manually
rm sessions/old_session.pkl
```

---

## 🔍 Command Reference

### Starting Sessions

```bash
# Create new named session
python3 full_memory_rag.py <document> <session_name>

# Resume default session
python3 full_memory_rag.py resume

# Start without arguments (shows available sessions)
python3 full_memory_rag.py
```

### Interactive Commands

```bash
sessions              # List all sessions
new <name> [doc]      # Create new session
switch <name>         # Switch to session
delete <name>         # Delete session (with confirmation)
save                  # Save current session
stats                 # Show current session info
memory                # Show conversation history
exit                  # Exit (auto-saves)
```

---

## 🎨 Session Lifecycle

```
1. CREATE
   python3 full_memory_rag.py doc.pdf my_session
   ↓
2. USE
   Ask questions, build knowledge
   ↓
3. AUTO-SAVE
   exit → saves to sessions/my_session.pkl
   ↓
4. RESUME
   python3 full_memory_rag.py resume
   switch my_session
   ↓
5. CONTINUE
   Pick up exactly where you left off!
```

---

## 🏆 Benefits

### 1. Organization
- Keep different research topics separate
- No mixing of context between projects
- Clean conversation histories

### 2. Flexibility
- Work on multiple documents simultaneously
- Switch between topics instantly
- Maintain perfect context for each

### 3. Persistence
- All sessions auto-save
- Resume any session anytime
- Never lose progress

### 4. Collaboration
- Share sessions with team members
- Each person has their own workspace
- Portable session files

---

## 📊 Comparison

### Before Multi-Session:
```
Single conversation → Exit → ALL LOST
Restart → Start over → No history
```

### After Multi-Session:
```
Session A: CRISPR research (10 conversations)
Session B: Python study (5 conversations)
Session C: Einstein notes (15 conversations)
↓
Switch between them anytime
All conversations preserved
Perfect context maintained
```

---

## ⚠️ Important Notes

### Auto-Save Behavior

- ✅ Auto-saves on `exit`
- ✅ Auto-saves when switching sessions
- ✅ Auto-saves when creating new session
- ⚠️ Use `save` for manual backup

### Session Names

- Must be valid filenames (no special characters like / \ : * ? " < > |)
- Case-sensitive: `Research` ≠ `research`
- Spaces allowed: `my research session` works fine

### Current Session Marker

```bash
sessions

→ my_research        # ← Current session (arrow marker)
  python_study
  notes_2024
```

---

## 🔧 Troubleshooting

### "No saved session found"

```bash
# Check available sessions
ls sessions/

# Create new session
python3 full_memory_rag.py sample_knowledge.txt my_session
```

### "Session not found"

```bash
# List all sessions first
sessions

# Use exact name (case-sensitive)
switch research_session    # Correct
switch Research_Session    # Wrong if not exact match
```

### Want to start fresh

```bash
# Option 1: Create new session
new fresh_start sample_knowledge.txt

# Option 2: Delete sessions directory
rm -r sessions/
```

---

## 📝 Summary

**Multiple Sessions = Multiple Independent Conversations**

- **Create**: `new <name> [doc]`
- **Switch**: `switch <name>`
- **List**: `sessions`
- **Delete**: `delete <name>`

Each session remembers:
- ✅ Its own conversation history
- ✅ Its own document source
- ✅ Its own knowledge graph
- ✅ Complete context

**Result:** Work on multiple topics/documents with perfect organization! 🎯

---

*Generated: 2025-10-25*
*Multi-session feature tested and verified ✅*
*Manage unlimited chat sessions with ease!*
