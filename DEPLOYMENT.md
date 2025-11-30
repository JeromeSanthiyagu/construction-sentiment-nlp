# Construction Sentiment NLP - Deployment Guide

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended - FREE)

**Pros:**
- Free hosting
- Easy deployment from GitHub
- Automatic updates when you push to GitHub
- Built-in support for Streamlit apps
- HTTPS included

**Steps:**

1. **Create a GitHub Repository**
   ```bash
   cd C:\Users\jerom\.gemini\antigravity\scratch\construction-sentiment-nlp
   git init
   git add .
   git commit -m "Initial commit - Construction Sentiment NLP"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/construction-sentiment-nlp.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository: `construction-sentiment-nlp`
   - Main file path: `src/app.py`
   - Click "Deploy"

3. **Wait for Deployment**
   - Streamlit will install dependencies automatically
   - Your app will be live at: `https://YOUR_USERNAME-construction-sentiment-nlp.streamlit.app`

---

### Option 2: Hugging Face Spaces (Alternative - FREE)

**Pros:**
- Free hosting
- Good for ML/AI apps
- Easy to share

**Steps:**

1. Create a new Space on [huggingface.co/spaces](https://huggingface.co/spaces)
2. Choose "Streamlit" as the SDK
3. Upload your files or connect GitHub
4. Add `app.py` to the root (move from `src/`)

---

### Option 3: Railway.app (Paid after trial)

**Pros:**
- More control
- Can run background tasks
- Database support

**Cost:** $5/month after trial

---

## Required Files for Deployment

All files are already created and ready:

✅ `requirements.txt` - Dependencies  
✅ `src/app.py` - Main application  
✅ `src/model.py` - Model training  
✅ `src/generate_data.py` - Data generation  
✅ `.gitignore` - Git ignore file  
✅ `README.md` - Project documentation  

---

## Post-Deployment Checklist

- [ ] Test all pages on deployed app
- [ ] Verify file upload works
- [ ] Check analytics dashboard loads
- [ ] Test daily summary generation
- [ ] Share link with stakeholders

---

## Troubleshooting

### Issue: "Module not found"
**Solution:** Check `requirements.txt` has all dependencies

### Issue: "File not found"
**Solution:** Ensure all file paths are relative, not absolute

### Issue: "Memory limit exceeded"
**Solution:** Reduce dataset size or upgrade to paid tier

---

## Next Steps After Deployment

1. **Custom Domain** (Optional)
   - Streamlit Cloud allows custom domains on paid plans
   
2. **Analytics**
   - Add Google Analytics to track usage
   
3. **Authentication** (Optional)
   - Add login for sensitive data
   
4. **Database Integration** (Optional)
   - Connect to PostgreSQL for persistent storage

---

## Recommended: Streamlit Community Cloud

For this project, I recommend **Streamlit Community Cloud** because:
- It's completely free
- Perfect for Streamlit apps
- Easy deployment process
- Automatic HTTPS
- No server management needed

Your app will be publicly accessible and you can share the link with anyone!
