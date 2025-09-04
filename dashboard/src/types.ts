// API Response Types
export interface APIStatus {
  status: string;
  models: {
    xgboost: boolean;
    logistic_regression: boolean;
    calibrator: boolean;
  };
  data: {
    processed: boolean;
    labeled: boolean;
    input_linked: boolean;
  };
  api_version: string;
}

export interface EventDetails {
  season: number;
  event_key: string;
  track_name: string;
  location: string;
  round: number | null;
  event_date: string | null;
}

export interface EventsResponse {
  seasons: number[];
  events_by_season: Record<number, string[]>;
  event_details: Record<string, EventDetails>;
}

export interface DriverData {
  id: string;
  name: string;
  model_pick: number;
  model_confidence: number;
  actual_beats_teammate: number | null;
  teammate_gap_ms: number | null;
}

export interface PredictionRow {
  team: string;
  driver_a: DriverData;
  driver_b: DriverData;
  model_pick: string;
  model_confidence: number;
  model_correct: boolean | null;
}

export interface PredictionsResponse {
  season: number;
  event_key: string;
  predictions: PredictionRow[];
  summary: {
    total_teams: number;
    model_accuracy: number;
    model_correct: number;
    model_incorrect: number;
  };
}

export interface TrackPath {
  label: string;
  color: string;
  d: string;
}

export interface CircuitInfo {
  id: string;
  name: string;
  svg_url: string;
  paths: string[];
  viewbox?: {
    x: number;
    y: number;
    width: number;
    height: number;
  };
  country_code: string;
}

export interface TrackMapResponse {
  ok: boolean;
  season: number;
  event_key: string;
  placeholder?: boolean;
  message?: string;
  circuit?: CircuitInfo;
}

export interface DriverInfo {
  code: string;
  color: string;
  name: string;
}

export interface DriversResponse {
  season: number;
  event_key: string;
  drivers: Record<string, DriverInfo>;
}

// Component Props Types
export interface SeasonEventPickerProps {
  onSelect: (selection: { season: number; event_key: string }) => void;
  selectedSeason?: number;
  selectedEvent?: string;
}

export interface TrackMapProps {
  season: number;
  event_key: string;
  drivers?: string[];
}

export interface PredictionsTableProps {
  predictions: PredictionRow[];
  summary: PredictionsResponse['summary'];
}

export interface StatBadgeProps {
  label: string;
  value: string | number;
  variant?: 'default' | 'success' | 'error' | 'info';
}
