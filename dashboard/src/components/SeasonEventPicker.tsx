import React, { useState, useEffect } from 'react';
import { apiClient } from '../api/client';
import { EventsResponse } from '../types';

interface SeasonEventPickerProps {
  onSelect: (selection: { season: number; event_key: string }) => void;
  selectedSeason?: number;
  selectedEvent?: string;
}

const SeasonEventPicker: React.FC<SeasonEventPickerProps> = ({
  onSelect,
  selectedSeason,
  selectedEvent
}) => {
  const [events, setEvents] = useState<EventsResponse | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    loadEvents();
  }, []);

  const loadEvents = async () => {
    try {
      setLoading(true);
      const eventsData = await apiClient.getEvents();
      setEvents(eventsData);
    } catch (error) {
      console.error('Failed to load events:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSeasonChange = (season: number) => {
    const seasonEvents = events?.events_by_season[season] || [];
    const defaultEvent = seasonEvents[seasonEvents.length - 1] || '';
    
    onSelect({ season, event_key: defaultEvent });
  };

  const handleEventChange = (event_key: string) => {
    if (selectedSeason) {
      onSelect({ season: selectedSeason, event_key });
    }
  };

  if (loading || !events) {
    return (
      <div className="flex items-center gap-4">
        <div className="w-32 h-10 bg-f1-panel rounded-lg animate-pulse"></div>
        <div className="w-48 h-10 bg-f1-panel rounded-lg animate-pulse"></div>
      </div>
    );
  }

  return (
    <div className="flex flex-col sm:flex-row gap-3">
      {/* Season Select */}
      <div className="flex flex-col">
        <label htmlFor="season" className="text-sm font-medium text-f1-muted mb-1">
          Season
        </label>
        <select
          id="season"
          value={selectedSeason || ''}
          onChange={(e) => handleSeasonChange(Number(e.target.value))}
          className="px-4 py-2 bg-f1-panel border border-f1-border rounded-lg text-f1-text focus:outline-none focus:ring-2 focus:ring-f1-accent focus:border-transparent"
        >
          <option value="">Select Season</option>
          {events.seasons.map((season) => (
            <option key={season} value={season}>
              {season}
            </option>
          ))}
        </select>
      </div>

      {/* Event Select */}
      <div className="flex flex-col">
        <label htmlFor="event" className="text-sm font-medium text-f1-muted mb-1">
          Event
        </label>
        <select
          id="event"
          value={selectedEvent || ''}
          onChange={(e) => handleEventChange(e.target.value)}
          disabled={!selectedSeason}
          className="px-4 py-2 bg-f1-panel border border-f1-border rounded-lg text-f1-text focus:outline-none focus:ring-2 focus:ring-f1-accent focus:border-transparent disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <option value="">Select Event</option>
          {selectedSeason && events.events_by_season[selectedSeason]?.map((event_key) => {
            const details = events.event_details[event_key];
            const displayName = details ? `${details.track_name}, ${details.location}` : event_key;
            return (
              <option key={event_key} value={event_key}>
                {displayName}
              </option>
            );
          })}
        </select>
      </div>
    </div>
  );
};

export default SeasonEventPicker;
