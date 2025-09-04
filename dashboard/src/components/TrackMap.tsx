import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { TrackMapResponse, CircuitInfo } from '../types';

interface TrackMapProps {
  season: number;
  event_key: string;
}

const TrackMap: React.FC<TrackMapProps> = ({ season, event_key }) => {
  const [trackData, setTrackData] = useState<TrackMapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadTrackMap();
  }, [season, event_key]);

  const loadTrackMap = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const data = await apiClient.getTrackMap(season, event_key);
      setTrackData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load track map');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="w-full h-96 bg-f1-panel rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-f1-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-f1-muted">Loading track map...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="w-full h-96 bg-f1-panel rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🏁</div>
          <p className="text-f1-bad mb-2">Track map unavailable</p>
          <p className="text-f1-muted text-sm">{error}</p>
          <button 
            onClick={loadTrackMap}
            className="mt-4 px-4 py-2 bg-f1-accent text-f1-bg rounded-lg text-sm hover:bg-opacity-90 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!trackData || trackData.placeholder) {
    return (
      <div className="w-full h-96 bg-f1-panel rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🏁</div>
          <p className="text-f1-muted mb-2">Track map not available yet</p>
          <p className="text-f1-muted text-sm">
            {trackData?.message || 'No circuit mapping found for this event'}
          </p>
          <div className="mt-4 p-3 bg-f1-border rounded-lg">
            <p className="text-xs text-f1-muted">
              Add SVG files to <code>data/assets/tracks/</code> and update <code>config/circuits_map.yaml</code>
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (!trackData.circuit) {
    return (
      <div className="w-full h-96 bg-f1-panel rounded-lg flex items-center justify-center">
        <div className="text-center">
          <div className="text-4xl mb-4">🏁</div>
          <p className="text-f1-muted">No circuit data available</p>
        </div>
      </div>
    );
  }

  const { circuit } = trackData;

  return (
    <div className="w-full">
      {/* Track Map SVG */}
      <div className="w-full bg-f1-bg rounded-lg overflow-hidden border border-f1-border">
        <svg
          viewBox="0 0 1000 1000"
          className="w-full h-96"
          style={{ background: 'linear-gradient(135deg, #0b0e14 0%, #121722 100%)' }}
        >
          {/* Render each track path with neon effects */}
          {circuit.paths.map((pathData, index) => (
            <g key={index}>
              {/* Outer glow path */}
              <path
                d={pathData}
                stroke="rgba(255,255,255,0.35)"
                strokeWidth="14"
                fill="none"
                strokeLinecap="round"
                opacity="0.25"
                className="neon-outline"
              />
              {/* Main neon path */}
              <path
                d={pathData}
                stroke="#cfe8ff"
                strokeWidth="10"
                fill="none"
                strokeLinecap="round"
                className="neon-path"
                filter="drop-shadow(0 0 6px #a8d1ff) drop-shadow(0 0 12px #7fb8ff)"
              />
            </g>
          ))}
        </svg>
      </div>

      {/* Circuit Info */}
      <div className="mt-4 text-center">
        <h3 className="text-lg font-semibold text-f1-accent mb-2">{circuit.name}</h3>
        <p className="text-f1-muted text-sm">
          {event_key.replace('_', ' Round ')}, {season}
        </p>
        <p className="text-f1-muted text-xs mt-1">
          {circuit.paths.length} path{circuit.paths.length !== 1 ? 's' : ''} • Static SVG
        </p>
      </div>
    </div>
  );
};

export default TrackMap;
