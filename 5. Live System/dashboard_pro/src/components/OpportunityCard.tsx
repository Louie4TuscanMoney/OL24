import { Show } from 'solid-js';

interface OpportunityCardProps {
  opportunity: any;
  onPlaceBet: () => void;
}

export default function OpportunityCard(props: OpportunityCardProps) {
  const opp = () => props.opportunity;
  
  // Determine game type
  const getGameType = () => {
    const edge = Math.abs(opp().edge || 0);
    if (edge <= 5) return 'HIGH_CONFIDENCE';
    if (edge <= 7) return 'BALANCED';
    if (edge <= 10) return 'CONSERVATIVE';
    return 'ULTRA_SELECTIVE';
  };
  
  const getGameCategory = () => {
    // Based on current score differential pattern
    const diff = Math.abs(opp().current_score || 0);
    if (diff > 10) return 'LEAD_HELD';
    if (diff > 3) return 'CLOSE';
    return 'VERY_CLOSE';
  };
  
  const shouldBet = () => {
    const edge = Math.abs(opp().edge || 0);
    const confidence = opp().p_win || 0;
    const category = getGameCategory();
    
    // Balanced strategy: edge ≥5, Lead Held only, confidence ≥60%
    return edge >= 5 && category === 'LEAD_HELD' && confidence >= 0.60;
  };
  
  const getConfidenceZone = () => {
    const edge = Math.abs(opp().edge || 0);
    if (edge <= 5) return { name: 'HIGH', color: 'green', accuracy: 82.5, avgError: 2.57 };
    if (edge <= 12) return { name: 'MEDIUM', color: 'yellow', accuracy: 65, avgError: 8.5 };
    return { name: 'LOW', color: 'red', accuracy: 55, avgError: 18.0 };
  };
  
  const getStrategyInfo = () => {
    const edge = Math.abs(opp().edge || 0);
    if (edge <= 5) {
      return {
        name: 'High-Confidence',
        pctOfGames: 31.7,
        dirAccuracy: 82.5,
        expectedGames: 439,
        action: 'BET AGGRESSIVELY'
      };
    }
    if (edge >= 5 && edge < 7) {
      return {
        name: 'Balanced',
        pctOfGames: 24,
        dirAccuracy: 60,
        expectedGames: 300,
        action: 'BET STANDARD'
      };
    }
    if (edge >= 7 && edge < 10) {
      return {
        name: 'Conservative',
        pctOfGames: 10,
        dirAccuracy: 69.4,
        expectedGames: 140,
        action: 'BET CONSERVATIVE'
      };
    }
    return {
      name: 'Ultra-Selective',
      pctOfGames: 2,
      dirAccuracy: 77.4,
      expectedGames: 30,
      action: 'BET VERY SELECTIVE'
    };
  };
  
  const zone = () => getConfidenceZone();
  const strategy = () => getStrategyInfo();
  const category = () => getGameCategory();
  const willBet = () => shouldBet();
  
  const zoneColors = {
    'green': 'from-green-500/20 to-emerald-500/20 border-green-500/50',
    'yellow': 'from-yellow-500/20 to-orange-500/20 border-yellow-500/50',
    'red': 'from-red-500/20 to-pink-500/20 border-red-500/50'
  };
  
  const categoryColors = {
    'LEAD_HELD': 'bg-green-500/20 text-green-300 border-green-500/30',
    'CLOSE': 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
    'VERY_CLOSE': 'bg-orange-500/20 text-orange-300 border-orange-500/30'
  };

  return (
    <div class={`bg-gradient-to-br ${zoneColors[zone().color]} backdrop-blur-lg rounded-2xl p-6 border-2 ${willBet() ? 'ring-2 ring-purple-500 ring-offset-2 ring-offset-slate-900' : ''}`}>
      {/* Header with Game Info */}
      <div class="flex justify-between items-start mb-4">
        <div>
          <h3 class="text-xl font-bold text-white mb-1">{opp().matchup}</h3>
          <div class="flex gap-2 flex-wrap">
            <span class={`px-2 py-1 rounded-full text-xs font-semibold border ${categoryColors[category()]}`}>
              {category().replace('_', ' ')}
            </span>
            <span class={`px-2 py-1 rounded-full text-xs font-semibold bg-${zone().color}-500/20 text-${zone().color}-300 border border-${zone().color}-500/30`}>
              {zone().name} CONFIDENCE
            </span>
            <Show when={willBet()}>
              <span class="px-2 py-1 rounded-full text-xs font-bold bg-purple-500 text-white animate-pulse">
                ✅ BET THIS
              </span>
            </Show>
            <Show when={!willBet()}>
              <span class="px-2 py-1 rounded-full text-xs font-semibold bg-gray-500/20 text-gray-400 border border-gray-500/30">
                CONTEXT ONLY
              </span>
            </Show>
          </div>
        </div>
        <div class="text-right">
          <div class="text-2xl font-black text-white">{opp().current_score}</div>
          <div class="text-sm text-white/60">{opp().period}</div>
        </div>
      </div>

      {/* Prediction Details */}
      <div class="grid grid-cols-2 gap-4 mb-4">
        <div>
          <div class="text-white/50 text-xs mb-1">OUR PREDICTION</div>
          <div class="text-xl font-bold text-white">
            {opp().prediction > 0 ? '+' : ''}{opp().prediction?.toFixed(1)}
          </div>
        </div>
        <div>
          <div class="text-white/50 text-xs mb-1">MARKET SPREAD</div>
          <div class="text-xl font-bold text-white">
            {opp().market_spread > 0 ? '+' : ''}{opp().market_spread?.toFixed(1)}
          </div>
        </div>
        <div>
          <div class="text-white/50 text-xs mb-1">EDGE</div>
          <div class={`text-xl font-bold ${opp().edge >= 5 ? 'text-green-400' : 'text-yellow-400'}`}>
            {opp().edge > 0 ? '+' : ''}{opp().edge?.toFixed(1)} pts
          </div>
        </div>
        <div>
          <div class="text-white/50 text-xs mb-1">P(WIN)</div>
          <div class={`text-xl font-bold ${opp().p_win >= 0.70 ? 'text-green-400' : opp().p_win >= 0.60 ? 'text-yellow-400' : 'text-red-400'}`}>
            {(opp().p_win * 100)?.toFixed(1)}%
          </div>
        </div>
      </div>

      {/* Strategy Info */}
      <div class="bg-black/30 rounded-xl p-4 mb-4">
        <div class="text-xs text-white/70 space-y-2">
          <div class="flex justify-between">
            <span>Strategy:</span>
            <span class="font-semibold text-white">{strategy().name}</span>
          </div>
          <div class="flex justify-between">
            <span>% of Games:</span>
            <span class="font-semibold text-white">{strategy().pctOfGames}%</span>
          </div>
          <div class="flex justify-between">
            <span>Direction Accuracy:</span>
            <span class="font-semibold text-white">{strategy().dirAccuracy}%</span>
          </div>
          <div class="flex justify-between">
            <span>Expected Games/Season:</span>
            <span class="font-semibold text-white">{strategy().expectedGames}</span>
          </div>
        </div>
      </div>

      {/* Confidence Zone Info */}
      <div class="bg-black/30 rounded-xl p-4 mb-4">
        <div class="text-xs font-bold text-white/90 mb-2">
          📊 {zone().name} CONFIDENCE ZONE
        </div>
        <div class="text-xs text-white/70 space-y-1">
          <div class="flex justify-between">
            <span>Direction Accuracy:</span>
            <span class="font-semibold text-white">{zone().accuracy}%</span>
          </div>
          <div class="flex justify-between">
            <span>Avg Error:</span>
            <span class="font-semibold text-white">{zone().avgError} pts</span>
          </div>
        </div>
      </div>

      {/* Betting Decision */}
      <Show when={willBet()}>
        <div class="bg-gradient-to-r from-purple-500/20 to-pink-500/20 rounded-xl p-4 border border-purple-500/50 mb-4">
          <div class="flex items-center gap-2 mb-2">
            <span class="text-2xl">✅</span>
            <span class="text-white font-bold">RECOMMENDED BET</span>
          </div>
          <div class="text-sm text-white/80">
            <div class="flex justify-between mb-1">
              <span>Kelly Stake:</span>
              <span class="font-bold text-green-400">${opp().kelly_stake?.toFixed(2) || '50.00'}</span>
            </div>
            <div class="flex justify-between">
              <span>Expected Value:</span>
              <span class="font-bold text-green-400">+${((opp().edge || 0) * 0.8).toFixed(2)}</span>
            </div>
          </div>
          <button
            onClick={() => props.onPlaceBet()}
            class="w-full mt-3 bg-gradient-to-r from-purple-500 to-pink-500 hover:from-purple-600 hover:to-pink-600 text-white font-bold py-3 rounded-xl transition-all transform hover:scale-105"
          >
            PLACE BET
          </button>
        </div>
      </Show>

      <Show when={!willBet()}>
        <div class="bg-gray-500/10 rounded-xl p-4 border border-gray-500/30">
          <div class="flex items-center gap-2 mb-2">
            <span class="text-2xl">ℹ️</span>
            <span class="text-white/70 font-semibold">FOR CONTEXT ONLY</span>
          </div>
          <div class="text-xs text-white/50">
            {category() !== 'LEAD_HELD' 
              ? `Game type "${category()}" not in approved list (only Lead Held)`
              : opp().edge < 5 
              ? `Edge ${opp().edge?.toFixed(1)} below 5.0 threshold`
              : opp().p_win < 0.60
              ? `Confidence ${(opp().p_win * 100)?.toFixed(1)}% below 60% minimum`
              : 'Other filter criteria not met'
            }
          </div>
        </div>
      </Show>

      {/* Additional Context */}
      <div class="mt-4 pt-4 border-t border-white/10">
        <div class="text-xs text-white/40">
          Q2 6:00 Snapshot • Mamba Mentality System • {opp().branch || 'Final'} Prediction
        </div>
      </div>
    </div>
  );
}


