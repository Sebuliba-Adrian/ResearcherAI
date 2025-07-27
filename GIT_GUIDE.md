# Git Version Control Guide

## ✅ .gitignore Created

A comprehensive `.gitignore` file has been created to keep your repository clean and efficient.

---

## 🚫 What Gets Ignored (Not Committed)

### Virtual Environment
```
venv/           # All virtual environment files (10,000+ files)
env/
.venv/
```
**Why:** Virtual environments are environment-specific and easily recreatable with `pip install -r requirements.txt`

### Session Data
```
sessions/       # All your chat sessions
*.pkl           # Session pickle files
```
**Why:** Session files contain your personal conversations and can be large. Keep them local or backup separately.

### Generated Files
```
knowledge_graph.html
test_knowledge_graph.html
```
**Why:** These are generated outputs that can be recreated anytime

### Python Artifacts
```
__pycache__/
*.pyc
*.pyo
```
**Why:** Compiled Python files that are auto-generated

---

## ✅ What Should Be Committed

### Source Code
- `*.py` - All Python scripts
- `full_memory_rag.py` - Main application ⭐
- `demo_simple.py`, `demo_gemini_live.py`, etc.

### Documentation
- `*.md` - All markdown documentation
- `README.md`, `START_HERE.md`, `MULTI_SESSION_GUIDE.md`, etc.

### Configuration
- `requirements.txt` - Python dependencies
- `setup.sh` - Setup script
- `.gitignore` - This file

### Data
- `sample_knowledge.txt` - Sample document for testing

---

## 🚀 How to Fix Your Current Git State

If venv was already committed, here's how to clean it up:

### Step 1: Remove venv from Git (Keep Local Copy)
```bash
git rm -r --cached venv/
git rm -r --cached sessions/
git rm --cached *.pkl *.html 2>/dev/null
```

### Step 2: Add .gitignore
```bash
git add .gitignore
```

### Step 3: Commit the Changes
```bash
git commit -m "Add .gitignore and remove venv from version control"
```

### Step 4: Check Status
```bash
git status
```

You should see far fewer changes now!

---

## 📊 Before vs After

### Before (Without .gitignore):
```
Changes: 10,000+ files
Size: ~500 MB
Files: venv/, sessions/, *.pkl, *.pyc, etc.
```

### After (With .gitignore):
```
Changes: ~30 files
Size: ~500 KB
Files: Only source code, docs, and config
```

---

## 🎯 Recommended Git Workflow

### Initial Setup
```bash
# Initialize git (if not done)
git init

# Add .gitignore first
git add .gitignore

# Add your code and docs
git add *.py *.md *.txt requirements.txt setup.sh

# Commit
git commit -m "Initial commit: RAG system with multi-session support"
```

### Daily Workflow
```bash
# Check what changed
git status

# Add specific files
git add full_memory_rag.py README.md

# Or add all (gitignore will filter)
git add .

# Commit with message
git commit -m "Add multi-session support"

# Push to remote
git push origin main
```

---

## 📁 What Your Repository Should Contain

```
ResearcherAI/
├── .gitignore                  ✅ Committed
├── full_memory_rag.py          ✅ Committed (Main app)
├── requirements.txt            ✅ Committed
├── README.md                   ✅ Committed
├── START_HERE.md               ✅ Committed
├── MULTI_SESSION_GUIDE.md      ✅ Committed
├── sample_knowledge.txt        ✅ Committed
├── setup.sh                    ✅ Committed
├── demo_*.py                   ✅ Committed
├── venv/                       ❌ Ignored (10,000+ files)
├── sessions/                   ❌ Ignored (personal data)
├── *.pkl                       ❌ Ignored (session files)
└── *.html                      ❌ Ignored (generated)
```

---

## 🔒 Keeping Sessions Private

Your sessions/ directory contains personal conversations. Options:

### Option 1: Local Backup
```bash
# Backup sessions locally
cp -r sessions/ ~/Backups/ResearcherAI_sessions_$(date +%Y%m%d)/
```

### Option 2: Separate Private Repo
```bash
# Create separate private repo for sessions
cd sessions/
git init
git add .
git commit -m "My research sessions"
git remote add origin <your-private-repo-url>
git push -u origin main
```

### Option 3: Encrypted Backup
```bash
# Compress and encrypt sessions
tar -czf sessions.tar.gz sessions/
gpg -c sessions.tar.gz  # Creates sessions.tar.gz.gpg
```

---

## 🚀 Quick Commands Reference

```bash
# See what's ignored
git status --ignored

# Check .gitignore patterns
git check-ignore -v <filename>

# Count tracked files
git ls-files | wc -l

# See repository size
du -sh .git

# Remove accidentally committed large file
git rm --cached large_file.pkl
git commit -m "Remove large file"

# Clean untracked files (be careful!)
git clean -fd -n  # Dry run first
git clean -fd     # Actually remove
```

---

## ✅ Summary

**Problem:** 10,000+ changes due to venv being tracked
**Solution:** Created `.gitignore` to exclude venv, sessions, and generated files

**What to commit:**
- Source code (*.py)
- Documentation (*.md)
- Configuration (requirements.txt, setup.sh)
- Sample data (sample_knowledge.txt)

**What to ignore:**
- Virtual environment (venv/)
- Sessions (sessions/, *.pkl)
- Generated files (*.html)
- Python cache (__pycache__/, *.pyc)

**Result:** Clean repository with only ~30 essential files! 🎉

---

*Note: The .gitignore is already created. Just run the commands above to clean your git state.*
