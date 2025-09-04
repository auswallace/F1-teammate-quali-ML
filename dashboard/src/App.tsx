import React, { useState, useEffect } from 'react';
import SeasonEventPicker from './components/SeasonEventPicker';
import TrackMap from './components/TrackMap';
import PredictionsTable from './components/PredictionsTable';
import StatBadge from './components/StatBadge';
import Loading from './components/Loading';
import Empty from './components/Empty';
import { apiClient } from './api/client';
import { 
  APIStatus, 
  EventsResponse, 
  PredictionsResponse, 
  TrackMapResponse 
} from './types';

function App() {
  const [status, setStatus] = useState<APIStatus | null>(null);
  const [events, setEvents] = useState<EventsResponse | null>(null);
  const [selectedSeason, setSelectedSeason] = useState<number | undefined>();
  const [selectedEvent, setSelectedEvent] = useState<string | undefined>();
  const [predictions, setPredictions] = useState<PredictionsResponse | null>(null);
  const [trackMap, setTrackMap] = useState<TrackMapResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Load initial data
  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      setError(null);

      // Load status and events in parallel
      const [statusData, eventsData] = await Promise.all([
        apiClient.getStatus(),
        apiClient.getEvents()
      ]);

      setStatus(statusData);
      setEvents(eventsData);

      // Set default selection to most recent season/event
      if (eventsData.seasons.length > 0) {
        const latestSeason = eventsData.seasons[0];
        const latestEvents = eventsData.events_by_season[latestSeason];
        if (latestEvents.length > 0) {
          setSelectedSeason(latestSeason);
          setSelectedEvent(latestEvents[latestEvents.length - 1]);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load initial data');
    } finally {
      setLoading(false);
    }
  };

  const handleSeasonEventSelect = async (selection: { season: number; event_key: string }) => {
    try {
      setLoading(true);
      setError(null);
      setSelectedSeason(selection.season);
      setSelectedEvent(selection.event_key);

      // Load predictions and track map in parallel
      const [predictionsData, trackMapData] = await Promise.all([
        apiClient.getPredictions(selection.season, selection.event_key),
        apiClient.getTrackMap(selection.season, selection.event_key)
      ]);

      setPredictions(predictionsData);
      setTrackMap(trackMapData);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load event data');
    } finally {
      setLoading(false);
    }
  };

  if (loading && !status) {
    return <Loading message="Loading F1 Dashboard..." />;
  }

  if (error) {
    return (
      <div className="min-h-screen bg-f1-bg text-f1-text p-8">
        <div className="max-w-6xl mx-auto">
          <div className="f1-card p-8 text-center">
            <h1 className="text-2xl font-bold text-f1-bad mb-4">🚨 Error Loading Dashboard</h1>
            <p className="text-f1-muted mb-6">{error}</p>
            <button 
              onClick={loadInitialData}
              className="px-6 py-3 bg-f1-accent text-f1-bg rounded-xl font-semibold hover:bg-opacity-90 transition-colors"
            >
              Retry
            </button>
          </div>
        </div>
      </div>
    );
  }

  if (!status || !events) {
    return <Empty message="No data available" />;
  }

  return (
    <div className="min-h-screen bg-f1-bg text-f1-text font-inter">
      {/* Header */}
      <header className="f1-card m-4 p-6">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          <div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-f1-accent to-f1-accent-2 bg-clip-text text-transparent">
              🏁 F1 Teammate Qualifying Dashboard
            </h1>
            <p className="text-f1-muted mt-2">
              Machine Learning predictions with neon track maps
            </p>
          </div>
          
          <SeasonEventPicker
            onSelect={handleSeasonEventSelect}
            selectedSeason={selectedSeason}
            selectedEvent={selectedEvent}
          />
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto p-4">
        {loading ? (
          <Loading message="Loading event data..." />
        ) : selectedSeason && selectedEvent && predictions && trackMap ? (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Left Column - Track Map */}
            <div className="f1-card p-6">
              <h2 className="text-xl font-semibold mb-4">🗺️ Track Map</h2>
              <TrackMap 
                season={selectedSeason} 
                event_key={selectedEvent}
              />
            </div>

            {/* Right Column - Predictions */}
            <div className="f1-card p-6">
              <h2 className="text-xl font-semibold mb-4">🏆 Predictions</h2>
              
              {/* Summary Stats */}
              <div className="grid grid-cols-2 gap-4 mb-6">
                <StatBadge 
                  label="Teams" 
                  value={predictions.summary.total_teams} 
                  variant="default"
                />
                <StatBadge 
                  label="Model Accuracy" 
                  value={`${(predictions.summary.model_accuracy * 100).toFixed(1)}%`}
                  variant={predictions.summary.model_accuracy > 0.6 ? 'success' : 'default'}
                />
                <StatBadge 
                  label="Correct" 
                  value={predictions.summary.model_correct}
                  variant="success"
                />
                <StatBadge 
                  label="Incorrect" 
                  value={predictions.summary.model_incorrect}
                  variant="error"
                />
              </div>

              <PredictionsTable 
                predictions={predictions.predictions}
                summary={predictions.summary}
              />
            </div>
          </div>
        ) : (
          <div className="f1-card p-8 text-center">
            <h2 className="text-xl font-semibold mb-4">Select an Event</h2>
            <p className="text-f1-muted">
              Choose a season and event from the dropdown above to view predictions and track maps.
            </p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="f1-card m-4 p-4 text-center">
        <p className="text-f1-muted text-sm">
          Powered by FastF1 telemetry and machine learning • Built with React + FastAPI
        </p>
      </footer>
    </div>
  );
}

export default App;
