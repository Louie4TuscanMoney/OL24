import { type Component } from 'solid-js';

interface Props {
  message?: string;
}

const LoadingScreen: Component<Props> = (props) => {
  return (
    <div class="fixed inset-0 bg-gray-900 flex items-center justify-center z-50">
      <div class="text-center">
        {/* Animated Logo/Spinner */}
        <div class="mb-6">
          <div class="inline-block">
            <svg class="animate-spin h-16 w-16 text-blue-500" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
              <circle class="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path class="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
          </div>
        </div>

        {/* Ontologic Logo */}
        <div class="mb-4">
          <h1 class="text-4xl font-bold text-white mb-2">
            <span class="text-5xl">🎯</span> Ontologic XYZ
          </h1>
          <p class="text-gray-400 text-lg">NBA Analytics Platform</p>
        </div>

        {/* Loading Message */}
        <div class="text-blue-400 font-semibold mb-2">
          {props.message || 'Connecting to NBA API...'}
        </div>

        {/* Loading Dots */}
        <div class="flex justify-center gap-2">
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 0ms"></div>
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 150ms"></div>
          <div class="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style="animation-delay: 300ms"></div>
        </div>

        {/* System Status */}
        <div class="mt-8 text-sm text-gray-500 space-y-1">
          <div>✓ Checking NBA API connection...</div>
          <div>✓ Loading ML models...</div>
          <div>✓ Fetching live games...</div>
        </div>
      </div>
    </div>
  );
};

export default LoadingScreen;

