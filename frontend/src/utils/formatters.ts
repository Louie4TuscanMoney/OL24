/**
 * Utility formatters for NBA data
 */

/**
 * Format NBA clock from PT format to readable time
 * BULLETPROOF CONVERTER - Handles ALL formats
 * Examples:
 *   PT06M15.00S → 6:15
 *   PT00M42.00S → 0:42
 *   PT11M59.00S → 11:59
 *   6:15 → 6:15 (already formatted)
 *   6.25 → 6:15 (decimal)
 */
export function formatClock(clock: string | undefined | null): string {
  // Handle null/undefined/empty
  if (!clock) {
    return '0:00';
  }
  
  // Convert to string if needed
  const clockStr = String(clock).trim();
  
  if (!clockStr || clockStr === '0.0' || clockStr === '0:00' || clockStr === 'N/A') {
    return '0:00';
  }
  
  // ==========================================
  // METHOD 1: Handle PT format (ISO 8601 Duration)
  // PT06M15.00S → 6:15
  // ==========================================
  if (clockStr.includes('PT') && clockStr.includes('M')) {
    try {
      // Extract minutes: PT06M15.00S → 06
      const minutesMatch = clockStr.match(/PT(\d+)M/);
      const minutes = minutesMatch ? parseInt(minutesMatch[1], 10) : 0;
      
      // Extract seconds: PT06M15.00S → 15
      const secondsMatch = clockStr.match(/M(\d+(?:\.\d+)?)S/);
      const seconds = secondsMatch ? parseInt(secondsMatch[1], 10) : 0;
      
      const formatted = `${minutes}:${seconds.toString().padStart(2, '0')}`;
      console.log(`✅ Clock converted: ${clockStr} → ${formatted}`);
      return formatted;
    } catch (e) {
      console.error('❌ Failed to parse PT format:', clockStr, e);
    }
  }
  
  // ==========================================
  // METHOD 2: Already in MM:SS format
  // 6:15 → 6:15
  // ==========================================
  if (clockStr.includes(':')) {
    return clockStr;
  }
  
  // ==========================================
  // METHOD 3: Decimal format
  // 6.25 → 6:15
  // ==========================================
  try {
    const total = parseFloat(clockStr);
    if (!isNaN(total)) {
      const minutes = Math.floor(total);
      const seconds = Math.round((total - minutes) * 60);
      return `${minutes}:${seconds.toString().padStart(2, '0')}`;
    }
  } catch (e) {
    // Ignore
  }
  
  // ==========================================
  // FALLBACK: Return original if all else fails
  // ==========================================
  console.warn('⚠️ Could not parse clock format:', clockStr);
  return clockStr;
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

