import { createSignal, Show } from 'solid-js';
import axios from 'axios';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  opportunity?: any;
  apiUrl: string;
}

export default function BetTrackingModal(props: Props) {
  const [matchup, setMatchup] = createSignal(props.opportunity?.matchup || '');
  const [betLine, setBetLine] = createSignal(props.opportunity?.bet_line || '');
  const [stake, setStake] = createSignal(props.opportunity?.recommended_stake || 0);
  const [book, setBook] = createSignal('DraftKings');
  const [notes, setNotes] = createSignal('');
  const [submitting, setSubmitting] = createSignal(false);

  const handleSubmit = async (e: Event) => {
    e.preventDefault();
    setSubmitting(true);

    try {
      await axios.post(`${props.apiUrl}/api/bets/add`, {
        matchup: matchup(),
        bet_type: 'SPREAD',
        bet_line: betLine(),
        stake: stake(),
        book: book(),
        prediction: props.opportunity?.prediction,
        market_spread: props.opportunity?.market_spread,
        edge: props.opportunity?.edge,
        p_win: props.opportunity?.p_win,
        notes: notes()
      });

      alert('✅ Bet logged successfully!');
      props.onClose();
    } catch (error) {
      alert('❌ Error logging bet');
      console.error(error);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Show when={props.isOpen}>
      <div 
        class="fixed inset-0 bg-black/80 backdrop-blur-sm flex items-center justify-center z-50 p-4"
        onClick={props.onClose}
      >
        <div 
          class="bg-gradient-to-br from-slate-800 to-slate-900 rounded-3xl p-8 max-w-2xl w-full border-2 border-white/20 shadow-2xl"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div class="flex justify-between items-center mb-6">
            <h2 class="text-3xl font-black text-white">📝 Log Bet</h2>
            <button 
              onClick={props.onClose}
              class="text-white/60 hover:text-white text-3xl leading-none"
            >
              ×
            </button>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit}>
            <div class="space-y-4">
              {/* Matchup */}
              <div>
                <label class="text-white/70 text-sm font-semibold block mb-2">Matchup</label>
                <input 
                  type="text"
                  value={matchup()}
                  onInput={(e) => setMatchup(e.currentTarget.value)}
                  placeholder="e.g., BOS @ LAL"
                  class="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:border-white/40"
                  required
                />
              </div>

              {/* Bet Line */}
              <div>
                <label class="text-white/70 text-sm font-semibold block mb-2">Bet Line</label>
                <input 
                  type="text"
                  value={betLine()}
                  onInput={(e) => setBetLine(e.currentTarget.value)}
                  placeholder="e.g., LAL -3.5"
                  class="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:border-white/40"
                  required
                />
              </div>

              {/* Stake and Book */}
              <div class="grid grid-cols-2 gap-4">
                <div>
                  <label class="text-white/70 text-sm font-semibold block mb-2">Stake ($)</label>
                  <input 
                    type="number"
                    value={stake()}
                    onInput={(e) => setStake(parseFloat(e.currentTarget.value))}
                    placeholder="0"
                    step="10"
                    class="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:border-white/40"
                    required
                  />
                </div>
                <div>
                  <label class="text-white/70 text-sm font-semibold block mb-2">Sportsbook</label>
                  <select 
                    value={book()}
                    onChange={(e) => setBook(e.currentTarget.value)}
                    class="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white focus:outline-none focus:border-white/40"
                  >
                    <option value="DraftKings">DraftKings</option>
                    <option value="FanDuel">FanDuel</option>
                    <option value="BetMGM">BetMGM</option>
                    <option value="Caesars">Caesars</option>
                    <option value="BetOnline">BetOnline</option>
                    <option value="Other">Other</option>
                  </select>
                </div>
              </div>

              {/* Notes */}
              <div>
                <label class="text-white/70 text-sm font-semibold block mb-2">Notes (Optional)</label>
                <textarea 
                  value={notes()}
                  onInput={(e) => setNotes(e.currentTarget.value)}
                  placeholder="Any notes about this bet..."
                  rows={3}
                  class="w-full bg-white/10 border border-white/20 rounded-xl px-4 py-3 text-white placeholder-white/40 focus:outline-none focus:border-white/40 resize-none"
                />
              </div>

              {/* Prediction Info (if from opportunity) */}
              <Show when={props.opportunity}>
                <div class="bg-blue-500/20 border border-blue-500/50 rounded-xl p-4">
                  <div class="text-blue-300 font-semibold mb-2">📊 ML Prediction Data:</div>
                  <div class="grid grid-cols-2 gap-2 text-sm text-white/80">
                    <div>Edge: {props.opportunity?.edge.toFixed(1)} pts</div>
                    <div>P(Win): {(props.opportunity?.p_win * 100).toFixed(1)}%</div>
                    <div>Our Pred: {props.opportunity?.prediction > 0 ? '+' : ''}{props.opportunity?.prediction.toFixed(1)}</div>
                    <div>Market: {props.opportunity?.market_spread > 0 ? '+' : ''}{props.opportunity?.market_spread.toFixed(1)}</div>
                  </div>
                </div>
              </Show>

              {/* Submit */}
              <div class="flex gap-3 pt-4">
                <button 
                  type="button"
                  onClick={props.onClose}
                  class="flex-1 bg-white/10 hover:bg-white/20 text-white font-bold px-6 py-3 rounded-xl transition-all"
                >
                  Cancel
                </button>
                <button 
                  type="submit"
                  disabled={submitting()}
                  class="flex-1 bg-gradient-to-r from-green-500 to-emerald-600 hover:from-green-600 hover:to-emerald-700 disabled:opacity-50 text-white font-bold px-6 py-3 rounded-xl transition-all"
                >
                  {submitting() ? 'Logging...' : 'Log Bet 💰'}
                </button>
              </div>
            </div>
          </form>
        </div>
      </div>
    </Show>
  );
}

