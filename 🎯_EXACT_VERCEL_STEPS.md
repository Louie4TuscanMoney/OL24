# 🎯 EXACT VERCEL STEPS - FOLLOW THIS EXACTLY

**Current Error:** Vercel looking in `/vercel/path0/` instead of `live-system/dashboard_pro/`

---

## **📸 STEP-BY-STEP WITH SCREENSHOTS:**

### **Step 1: Go to Vercel**
1. Open browser
2. Go to: https://vercel.com/dashboard
3. You should see your project list

### **Step 2: Click Your Project**
- Look for project named something like:
  - `OL24` OR
  - `ontologicxyz` OR
  - The one pointing to `ontologicxyz.com`
- **Click on the project name**

### **Step 3: Go to Settings**
- At the top of the page, you'll see tabs:
  ```
  [Overview] [Deployments] [Analytics] [Logs] [Settings]
  ```
- **Click "Settings"**

### **Step 4: Find Root Directory**
- You should be on the "General" settings page by default
- **Scroll down** until you see a section that says:
  ```
  Build & Development Settings
  ```
- Look for **"Root Directory"**
  - It might say: `./` or be empty
  - There should be an **[Edit]** button next to it

### **Step 5: Edit Root Directory**
1. Click the **[Edit]** button next to "Root Directory"
2. A text box will appear
3. Type **EXACTLY**: `live-system/dashboard_pro`
   - No spaces
   - No leading `/`
   - No trailing `/`
   - Exactly as written above
4. Click **[Save]** or **[Confirm]**

### **Step 6: Verify It Saved**
- After saving, you should see:
  ```
  Root Directory: live-system/dashboard_pro
  ```
- If you still see `./` or empty, **it didn't save** - try again!

### **Step 7: Trigger New Deployment**
1. Click **"Deployments"** tab at the top
2. You'll see a list of deployments
3. Find the **most recent one** (top of the list)
4. On the right side, click the **three dots (...)** menu
5. Click **"Redeploy"**
6. A popup will appear asking to confirm
7. Click **"Redeploy"** button in the popup

### **Step 8: Watch the Build**
- The new deployment will start building
- Click on the new deployment to see logs
- **Look for these lines in the logs:**

**❌ BAD (If you still see this, Root Directory didn't save):**
```
npm error path /vercel/path0/package.json
npm error enoent Could not read package.json
```

**✅ GOOD (What you WANT to see):**
```
Running "install" command: `npm install`...
added 150 packages
Running "build" command: `npm run build`...
vite v5.0.7 building for production...
✓ built in 3.45s
Build Completed in /vercel/output [15s]
```

---

## **🔍 IF ROOT DIRECTORY OPTION IS MISSING:**

### **Alternative Location #1: Framework Preset**
Some Vercel projects show Root Directory in a different place:
1. In Settings → General
2. Look for **"Framework Preset"** dropdown
3. Select: **Vite**
4. Then look for Root Directory field below it

### **Alternative Location #2: Build Settings**
1. In Settings → General
2. Scroll to **"Build & Output Settings"**
3. Click **"Override"** next to Build Command
4. Look for Root Directory field

### **Alternative: Use Vercel CLI**
If you can't find the setting in the dashboard, we can use CLI:

```bash
# Install Vercel CLI
npm install -g vercel

# Login
vercel login

# Link to project
cd "/Users/test/Desktop/Tuscan Money/Ontologic XYZ/ML Research"
vercel link

# Set root directory
vercel --cwd live-system/dashboard_pro
```

---

## **📋 VERIFICATION CHECKLIST:**

After setting Root Directory and redeploying:

### **Check Build Logs:**
- [ ] No more `ENOENT` errors
- [ ] Says `package.json` found
- [ ] Says `vite build` completed
- [ ] Shows `dist` folder created

### **Check Deployed Site:**
- [ ] Visit https://ontologicxyz.com
- [ ] See login screen (not blank page)
- [ ] Login works with: `rwwc2018`
- [ ] Shows "🟢 ONLINE" status
- [ ] Displays NBA games

---

## **🆘 EMERGENCY ALTERNATIVE: MOVE FILES TO ROOT**

If you absolutely **cannot** set Root Directory in Vercel (some accounts have restrictions), we can move the frontend files to the repo root instead:

```bash
# Move frontend to root
mv live-system/dashboard_pro/* .

# Update paths
# (I can help with this if needed)

# Push to GitHub
git add .
git commit -m "Move frontend to root for Vercel"
git push
```

**But try the Root Directory setting first!** It's much cleaner.

---

## **❓ COMMON QUESTIONS:**

### **"Where exactly is Root Directory setting?"**
Path in Vercel Dashboard:
```
Your Project
  └─ Settings
      └─ General (left sidebar or top)
          └─ Build & Development Settings (section)
              └─ Root Directory (field)
```

### **"What if I have multiple projects?"**
Make sure you're in the **correct project**:
- Check the domain shows: `ontologicxyz.com`
- Check the GitHub repo shows: `Louie4TuscanMoney/OL24`

### **"Do I need to set it for all environments?"**
No, Root Directory applies to all environments (Production, Preview, Development).

---

## **📞 WHAT TO TELL ME IF STILL STUCK:**

1. **"I found Root Directory and set it to `live-system/dashboard_pro`"**
   → Great! Wait 3 min after redeploy and test site

2. **"I can't find Root Directory setting anywhere"**
   → Tell me your Vercel plan (Hobby/Pro) and I'll help with CLI method

3. **"I set it but still getting same error"**
   → Send screenshot of the Settings page showing Root Directory value

4. **"Build succeeded but site still broken"**
   → Different issue - we'll troubleshoot the app itself

---

**Most likely: You just need to find and set Root Directory in Vercel Settings!** 🎯

