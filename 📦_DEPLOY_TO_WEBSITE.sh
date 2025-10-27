#!/bin/bash

# DEPLOY DASHBOARD TO ONTOLOGICXYZ.COM

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📦 DEPLOY DASHBOARD TO ONTOLOGICXYZ.COM"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

SOURCE_DIR="5. Live System/dashboard_pro"
TARGET_DIR="/Users/test/Desktop/Tuscan Money/Websites/OntologicXYZ.com/NBADashboard"

echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"
echo ""

# Check if source exists
if [ ! -d "$SOURCE_DIR" ]; then
    echo "❌ Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Create target directory if needed
echo "📁 Preparing target directory..."
mkdir -p "$TARGET_DIR"

# Copy dashboard files
echo "📦 Copying dashboard files..."
echo ""

# Copy main files
cp -r "$SOURCE_DIR/src" "$TARGET_DIR/"
cp -r "$SOURCE_DIR/public" "$TARGET_DIR/" 2>/dev/null || true
cp "$SOURCE_DIR/package.json" "$TARGET_DIR/"
cp "$SOURCE_DIR/vite.config.ts" "$TARGET_DIR/"
cp "$SOURCE_DIR/tsconfig.json" "$TARGET_DIR/"
cp "$SOURCE_DIR/vercel.json" "$TARGET_DIR/"
cp "$SOURCE_DIR/tailwind.config.js" "$TARGET_DIR/"
cp "$SOURCE_DIR/postcss.config.js" "$TARGET_DIR/"
cp "$SOURCE_DIR/index.html" "$TARGET_DIR/"

# Create .env for production
echo "🔧 Creating production .env..."
cat > "$TARGET_DIR/.env" << 'EOF'
# Production API URL
# Update this to your deployed backend API
VITE_API_URL=https://your-api-backend.herokuapp.com

# Or for local testing:
# VITE_API_URL=http://localhost:8001
EOF

# Create README
echo "📄 Creating deployment README..."
cat > "$TARGET_DIR/README.md" << 'EOF'
# NBA Trading Dashboard

Professional NBA betting dashboard for OntologicXYZ.com

## Deploy to Vercel

```bash
npm install -g vercel
vercel login
vercel deploy --prod
```

## Local Development

```bash
npm install
npm run dev
```

## Environment Variables

Set in Vercel dashboard or `.env`:
- `VITE_API_URL` - Backend API URL

## Built with

- SolidJS
- Vite
- TailwindCSS
- OntoRisk
EOF

echo "✅ Files copied successfully"
echo ""

# Summary
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ DASHBOARD READY FOR DEPLOYMENT"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "📁 Location: $TARGET_DIR"
echo ""
echo "🚀 To deploy to Vercel:"
echo "   cd \"$TARGET_DIR\""
echo "   npm install"
echo "   vercel login"
echo "   vercel deploy --prod"
echo ""
echo "🌐 Will be live at: https://ontologicxyz.com/NBADashboard"
echo ""
echo "⚙️ Don't forget to:"
echo "   1. Update VITE_API_URL in Vercel environment variables"
echo "   2. Deploy backend API to Heroku/Railway/Render"
echo "   3. Update vercel.json with correct backend URL"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

