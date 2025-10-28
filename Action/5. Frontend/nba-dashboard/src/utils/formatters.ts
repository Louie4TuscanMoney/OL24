/**
 * Utility formatters for NBA data
 */

/**
 * Format NBA clock from PT format to readable time
 * Examples:
 *   PT06M15.00S → 6:15
 *   PT00M42.00S → 0:42
 *   PT11M59.00S → 11:59
 */
export function formatClock(clock: string): string {
  if (!clock || clock === '0.0' || clock === '0:00') {
    return '0:00';
  }
  
  // Handle PT format (e.g., PT06M15.00S)
  if (clock.includes('PT') && clock.includes('M')) {
    try {
      const minutes = parseInt(clock.split('M')[0].replace('PT', ''));
      const secondsPart = clock.split('M')[1].replace('S', '').split('.')[0];
      const seconds = parseInt(secondsPart) || 0;
      
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    } catch (e) {
      console.warn('Failed to parse PT clock format:', clock);
      return clock;
    }
  }
  
  // Handle MM:SS format (already good)
  if (clock.includes(':')) {
    return clock;
  }
  
  // Handle decimal format (e.g., 6.25 = 6:15)
  try {
    const total = parseFloat(clock);
    const minutes = Math.floor(total);
    const seconds = Math.round((total - minutes) * 60);
    return `${minutes}:${seconds.toString().padStart(2, '0')}`;
  } catch (e) {
    return clock;
  }
}

/**
 * Format score differential with + sign
 */
export function formatDifferential(diff: number): string {
  return diff > 0 ? `+${diff}` : `${diff}`;
}

/**
 * Format money
 */
export function formatMoney(amount: number): string {
  return `$${amount.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
}

/**
 * Format percentage
 */
export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

/**
 * Format decimal number
 */
export function formatDecimal(value: number, decimals: number = 2): string {
  return value.toFixed(decimals);
}

