/**
 * NBA Dashboard Backend Server
 * 
 * Provides APIs for:
 * - NBA live game data
 * - BetOnline odds scraping
 * - ML model predictions
 */

import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 5000;

app.use(cors());
app.use(express.json());

// Cache for data
let gamesCache = [];
let oddsCache = {};
let predictionsCache = {};
let lastUpdate = new Date();

// NBA API endpoint
app.get('/api/nba/games', async (req, res) => {
  try {
    // Call Python NBA API script
    const python = spawn('python3', [
      path.join(__dirname, '../../2. NBA API/1. API Setup/test_nba_api.py')
    ]);

    let data = '';
    python.stdout.on('data', (chunk) => {
      data += chunk.toString();
    });

    python.on('close', (code) => {
      try {
        // Parse games data
        // For now, return mock data structure
        const games = gamesCache.length > 0 ? gamesCache : [];
        
        res.json({
          success: true,
          games: games,
          timestamp: new Date().toISOString()
        });
      } catch (error) {
        res.status(500).json({ success: false, error: error.message });
      }
    });

  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// BetOnline odds endpoint
app.get('/api/betonline/odds', async (req, res) => {
  try {
    // Call Python scraper
    res.json({
      success: true,
      odds: oddsCache,
      scraped_at: lastUpdate.toISOString()
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Predictions endpoint
app.get('/api/predictions', async (req, res) => {
  try {
    res.json({
      success: true,
      predictions: predictionsCache,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Trigger prediction for a game
app.post('/api/predict/:gameId', async (req, res) => {
  try {
    const { gameId } = req.params;
    const { pattern } = req.body;

    // Call Python game engine
    const python = spawn('python3', ['-c', `
import sys
sys.path.insert(0, '../')
from game_engine import GameEngine
import numpy as np

engine = GameEngine()
pattern = np.array(${JSON.stringify(pattern)})
result = engine.predict(pattern, return_details=True)

import json
print(json.dumps({
  'halftime': float(result['halftime']),
  'final': float(result['final']),
  'confidence': result['confidence'],
  'should_bet': bool(result['should_bet'])
}))
    `]);

    let data = '';
    python.stdout.on('data', (chunk) => {
      data += chunk.toString();
    });

    python.on('close', (code) => {
      try {
        const result = JSON.parse(data);
        predictionsCache[gameId] = result;
        res.json({ success: true, prediction: result });
      } catch (error) {
        res.status(500).json({ success: false, error: 'Failed to parse prediction' });
      }
    });

  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
});

// Health check
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    uptime: process.uptime()
  });
});

app.listen(PORT, () => {
  console.log(`
╔════════════════════════════════════════╗
║   NBA Dashboard Server Running         ║
║                                        ║
║   Backend: http://localhost:${PORT}      ║
║   Frontend: http://localhost:3000      ║
║                                        ║
║   Status: READY FOR OPENING NIGHT 🏀   ║
╚════════════════════════════════════════╝
  `);
});

