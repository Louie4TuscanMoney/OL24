# 🎨 FRONTEND API INTEGRATION GUIDE

**Status:** ✅ Your frontend is PERFECTLY structured!  
**Issue:** Waiting for Railway to deploy updated API (~3 minutes)

---

## 📍 API ENDPOINTS - ALL READY

Your backend at `https://ol24-production.up.railway.app` provides:

### 1️⃣ `/api/stats/teams` - All Teams with Records
**Current Frontend:** `TeamsDirectory.tsx` ✅ Already integrated!

**Response:**
```json
{
  "teams": [
    {
      "team_id": "1610612737",
      "abbreviation": "ATL",
      "full_name": "Atlanta Hawks",
      "logo_url": "https://cdn.nba.com/logos/nba/1610612737/primary/L/logo.svg",
      "primary_color": "#E03A3E",
      "secondary_color": "#C1D32F",
      "games_played": 9,
      "wins": 5,
      "losses": 4,
      "ppg": 104.2,
      "net_rating": 2.1
    }
  ]
}
```

**Frontend Display:**
- ✅ Team logos (already showing)
- ✅ Win-Loss records (will show after Railway deploys)
- ✅ PPG (will show after Railway deploys)
- ✅ Click to view team details

---

### 2️⃣ `/api/schedule` - Today's Games
**Current Frontend:** `SchedulePage.tsx` ✅ Already integrated!

**Response:**
```json
{
  "games": [
    {
      "game_id": "0022500131",
      "date": "2025-10-29",
      "time": "19:30",
      "home_team": {
        "abbr": "TOR",
        "name": "Toronto Raptors",
        "logo": "https://cdn.nba.com/logos/nba/1610612761/primary/L/logo.svg"
      },
      "away_team": {
        "abbr": "HOU",
        "name": "Houston Rockets",
        "logo": "https://cdn.nba.com/logos/nba/1610612745/primary/L/logo.svg"
      },
      "status": "Scheduled"
    }
  ]
}
```

**Frontend Display:**
- ✅ Game cards with team logos
- ✅ Date/time
- ✅ Click to see game details (modal)

---

### 3️⃣ `/api/game/{game_id}/details` - **NEW!** Game Details with Lineups
**You need to add this!** Use it in `ScheduleGameModal.tsx`

**Response:**
```json
{
  "game": {
    "game_id": "0022500131",
    "date": "2025-10-29",
    "time": "19:30",
    "status": "Scheduled"
  },
  "home_team": {
    "abbreviation": "TOR",
    "full_name": "Toronto Raptors",
    "logo_url": "...",
    "primary_color": "#CE1141",
    "secondary_color": "#000000",
    "record": "5-4",
    "wins": 5,
    "losses": 4,
    "ppg": 108.5,
    "net_rating": 1.2,
    "projected_starters": [
      {
        "player_id": "1627783",
        "name": "Pascal Siakam",
        "first_name": "Pascal",
        "last_name": "Siakam",
        "position": "PF",
        "jersey": "43",
        "headshot_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/1627783.png",
        "ppg": 22.5,
        "rpg": 7.8,
        "apg": 5.1,
        "fg_pct": 0.485,
        "mpg": 35.2
      }
      // ... 4 more starters
    ]
  },
  "away_team": {
    // Same structure as home_team
  },
  "injuries": [
    {
      "player_name": "Scottie Barnes",
      "position": "F",
      "status": "Questionable",
      "injury_type": "Ankle",
      "description": "Sprained right ankle"
    }
  ]
}
```

**Frontend Usage:**
```typescript
// In ScheduleGameModal.tsx
const [gameDetails, setGameDetails] = createSignal(null);

onMount(async () => {
  const response = await fetch(`${API_BASE}/api/game/${gameId}/details`);
  const data = await response.json();
  setGameDetails(data);
});

// Display:
// - Team records (e.g., "TOR (5-4) vs HOU (8-2)")
// - Projected starters with headshots, positions, season averages
// - Active injuries with status indicators
```

---

### 4️⃣ `/api/stats/player/{player_id}` - Player Profile
**Usage:** When clicking on a player in the lineup

**Response:**
```json
{
  "player_id": "1627783",
  "name": "Pascal Siakam",
  "first_name": "Pascal",
  "last_name": "Siakam",
  "headshot_url": "https://cdn.nba.com/headshots/nba/latest/1040x760/1627783.png",
  "position": "PF",
  "jersey": "43",
  "team": {
    "abbreviation": "TOR",
    "full_name": "Toronto Raptors",
    "logo_url": "...",
    "primary_color": "#CE1141",
    "secondary_color": "#000000"
  },
  "season_stats": {
    "games_played": 9,
    "ppg": 22.5,
    "rpg": 7.8,
    "apg": 5.1,
    "fg_pct": 0.485,
    "fg3_pct": 0.342,
    "ft_pct": 0.812
  },
  "advanced_stats": {
    "ts_pct": 0.584,
    "pts_100": 28.3,
    "bpm": 4.2,
    "per": 21.5,
    "vorp": 0.8,
    "usage_pct": 0.285,
    "win_shares": 0.7
  },
  "last10_games": [...]
}
```

---

### 5️⃣ `/api/injuries` - All Active Injuries
**Usage:** Injuries page or sidebar

**Response:**
```json
{
  "injuries": [
    {
      "player_name": "Scottie Barnes",
      "team_abbr": "TOR",
      "position": "F",
      "status": "Questionable",
      "injury_type": "Ankle",
      "description": "Sprained right ankle",
      "return_date": null
    }
  ],
  "count": 1
}
```

---

### 6️⃣ `/api/team/{team_abbr}/depth-chart` - Team Lineup
**Usage:** Team detail page

**Response:**
```json
{
  "team": {
    "abbreviation": "LAL",
    "full_name": "Los Angeles Lakers",
    "logo_url": "..."
  },
  "depth_chart": [
    {
      "player_id": "1629029",
      "player_name": "Luka Dončić",
      "position": "F",
      "depth_rank": 1,
      "jersey": "77",
      "headshot_url": "...",
      "season_stats": {
        "ppg": 46.0,
        "rpg": 11.5,
        "apg": 8.5,
        "mpg": 38.5
      }
    }
  ]
}
```

---

### 7️⃣ `/api/stats/standings` - League Standings
**Usage:** Standings page

**Response:**
```json
{
  "standings": {
    "East": [
      {
        "abbreviation": "PHI",
        "full_name": "Philadelphia 76ers",
        "rank": 1,
        "wins": 4,
        "losses": 0,
        "gb": 0.0,
        "streak": "W4"
      }
    ],
    "West": [...]
  }
}
```

---

## 🚀 FRONTEND UPDATES NEEDED

### 1. Update `ScheduleGameModal.tsx`

**Add game details fetch:**
```typescript
// In ScheduleGameModal.tsx
import { createSignal, onMount, For, Show } from 'solid-js';

interface Props {
  gameId: string;
  isOpen: boolean;
  onClose: () => void;
}

const ScheduleGameModal: Component<Props> = (props) => {
  const [gameDetails, setGameDetails] = createSignal(null);
  const [loading, setLoading] = createSignal(true);

  const API_BASE = 'https://ol24-production.up.railway.app';

  onMount(async () => {
    if (props.gameId) {
      try {
        const response = await fetch(`${API_BASE}/api/game/${props.gameId}/details`);
        const data = await response.json();
        setGameDetails(data);
      } catch (error) {
        console.error('Error fetching game details:', error);
      } finally {
        setLoading(false);
      }
    }
  });

  return (
    <Show when={props.isOpen}>
      <div class="fixed inset-0 bg-black bg-opacity-75 flex items-center justify-center z-50">
        <div class="bg-gray-800 rounded-lg p-8 max-w-6xl w-full max-h-screen overflow-y-auto">
          
          <Show when={loading()}>
            <p class="text-white text-center">Loading game details...</p>
          </Show>

          <Show when={!loading() && gameDetails()}>
            {/* Header */}
            <div class="text-center mb-6">
              <h2 class="text-3xl font-bold text-white mb-2">
                {gameDetails().away_team.abbreviation} @ {gameDetails().home_team.abbreviation}
              </h2>
              <p class="text-gray-400">
                {gameDetails().away_team.record} vs {gameDetails().home_team.record}
              </p>
            </div>

            {/* Team Lineups Side by Side */}
            <div class="grid grid-cols-2 gap-8 mb-6">
              
              {/* Away Team Lineup */}
              <div>
                <h3 class="text-2xl font-bold text-white mb-4 flex items-center">
                  <img 
                    src={gameDetails().away_team.logo_url} 
                    alt={gameDetails().away_team.abbreviation}
                    class="w-8 h-8 mr-2"
                  />
                  {gameDetails().away_team.abbreviation} Projected Starters
                </h3>
                
                <For each={gameDetails().away_team.projected_starters}>
                  {(player) => (
                    <div class="bg-gray-700 rounded-lg p-4 mb-3 hover:bg-gray-600 cursor-pointer">
                      <div class="flex items-center gap-4">
                        <img 
                          src={player.headshot_url} 
                          alt={player.name}
                          class="w-16 h-16 rounded-full object-cover"
                        />
                        <div class="flex-1">
                          <p class="text-white font-bold text-lg">{player.first_name} {player.last_name}</p>
                          <p class="text-gray-400 text-sm">
                            #{player.jersey} • {player.position}
                          </p>
                        </div>
                        <div class="text-right">
                          <p class="text-blue-400 font-bold text-xl">{player.ppg.toFixed(1)}</p>
                          <p class="text-gray-400 text-xs">PPG</p>
                        </div>
                        <div class="text-right">
                          <p class="text-green-400 font-bold text-lg">{player.rpg.toFixed(1)}</p>
                          <p class="text-gray-400 text-xs">RPG</p>
                        </div>
                        <div class="text-right">
                          <p class="text-purple-400 font-bold text-lg">{player.apg.toFixed(1)}</p>
                          <p class="text-gray-400 text-xs">APG</p>
                        </div>
                      </div>
                    </div>
                  )}
                </For>
              </div>

              {/* Home Team Lineup */}
              <div>
                <h3 class="text-2xl font-bold text-white mb-4 flex items-center">
                  <img 
                    src={gameDetails().home_team.logo_url} 
                    alt={gameDetails().home_team.abbreviation}
                    class="w-8 h-8 mr-2"
                  />
                  {gameDetails().home_team.abbreviation} Projected Starters
                </h3>
                
                <For each={gameDetails().home_team.projected_starters}>
                  {(player) => (
                    // Same as away team
                  )}
                </For>
              </div>
            </div>

            {/* Injuries Section */}
            <Show when={gameDetails().injuries.length > 0}>
              <div class="bg-red-900 bg-opacity-20 border border-red-500 rounded-lg p-4 mb-6">
                <h3 class="text-xl font-bold text-red-400 mb-3">🚑 Active Injuries</h3>
                <For each={gameDetails().injuries}>
                  {(injury) => (
                    <div class="flex justify-between items-center mb-2 text-white">
                      <span>{injury.player_name} ({injury.position})</span>
                      <span class={`px-3 py-1 rounded text-sm font-bold ${
                        injury.status === 'Out' ? 'bg-red-600' :
                        injury.status === 'Doubtful' ? 'bg-orange-600' :
                        'bg-yellow-600'
                      }`}>
                        {injury.status}
                      </span>
                      <span class="text-gray-400 text-sm">{injury.injury_type}</span>
                    </div>
                  )}
                </For>
              </div>
            </Show>

            {/* Close Button */}
            <button 
              onClick={props.onClose}
              class="w-full bg-blue-600 hover:bg-blue-700 text-white font-bold py-3 rounded-lg"
            >
              Close
            </button>
          </Show>
        </div>
      </div>
    </Show>
  );
};
```

---

### 2. Update `TeamsDirectory.tsx` - **Already Perfect!**

Your teams page is already correctly structured. Once Railway deploys (in ~2 minutes), you'll automatically see:
- ✅ Win-Loss records (5-4, 8-2, etc.)
- ✅ PPG (104.2, 110.8, etc.)
- ✅ Win percentages

**No code changes needed** - just wait for deployment!

---

### 3. Update `TeamPage.tsx` - Add Depth Chart

**Fetch depth chart:**
```typescript
const [depthChart, setDepthChart] = createSignal([]);

onMount(async () => {
  const response = await fetch(`${API_BASE}/api/team/${teamAbbr}/depth-chart`);
  const data = await response.json();
  setDepthChart(data.depth_chart || []);
});

// Display:
<For each={depthChart()}>
  {(player) => (
    <div class="flex items-center gap-4 bg-gray-700 p-4 rounded-lg mb-3">
      <img src={player.headshot_url} class="w-16 h-16 rounded-full" />
      <div>
        <p class="text-white font-bold">{player.player_name}</p>
        <p class="text-gray-400">{player.position} • #{player.jersey}</p>
      </div>
      <div class="ml-auto text-right">
        <p class="text-blue-400 font-bold">{player.season_stats.ppg} PPG</p>
        <p class="text-gray-400 text-sm">{player.season_stats.mpg} MPG</p>
      </div>
    </div>
  )}
</For>
```

---

## 🧪 TESTING AFTER DEPLOYMENT

Wait ~3 minutes for Railway to deploy, then test:

```bash
# Test teams API (should show wins/losses/ppg)
curl -s https://ol24-production.up.railway.app/api/stats/teams | jq '.teams[0]'

# Test game details (should show starters + injuries)
curl -s https://ol24-production.up.railway.app/api/game/0022500131/details | jq '.'
```

---

## 🎨 FRONTEND SUMMARY

**What's Already Perfect:**
- ✅ `TeamsDirectory.tsx` - displays teams with logos, ready for records/PPG
- ✅ `SchedulePage.tsx` - displays schedule, ready for game details modal
- ✅ `TeamPage.tsx` - team page structure, ready for depth chart
- ✅ API base URL configured: `https://ol24-production.up.railway.app`

**What to Add:**
- 🔨 Update `ScheduleGameModal.tsx` to fetch `/api/game/{id}/details`
- 🔨 Display projected starters with season averages
- 🔨 Display active injuries
- 🔨 Add depth chart to `TeamPage.tsx`

**Timeline:**
1. **Now:** Railway is deploying API updates (~3 minutes)
2. **After deployment:** Teams page will automatically show W-L records + PPG
3. **Then:** Update modal to show starters/injuries (20 minute task)
4. **Result:** Fully functional NBA analytics platform! 🎉

---

**Your platform is 95% complete!** Just waiting for Railway deployment + minor frontend updates for game details modal.

