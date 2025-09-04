// API client for F1 Teammate Qualifying backend

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

class APIClient {
  private baseURL: string;

  constructor(baseURL: string = API_BASE) {
    this.baseURL = baseURL;
  }

  private async request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    });

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`API request failed: ${response.status} ${response.statusText} - ${errorText}`);
    }

    return response.json();
  }

  // Status endpoints
  async getStatus(): Promise<any> {
    return this.request('/api/status');
  }

  // Events endpoints
  async getEvents(): Promise<any> {
    return this.request('/api/events');
  }

  async getSeasonEvents(season: number): Promise<any> {
    return this.request(`/api/events/${season}`);
  }

  // Predictions endpoints
  async getPredictions(season: number, event_key: string): Promise<any> {
    return this.request(`/api/predictions/${season}/${event_key}`);
  }

  async getPredictionsSummary(season: number, event_key: string): Promise<any> {
    return this.request(`/api/predictions/${season}/${event_key}/summary`);
  }

  // Track map endpoints
  async getTrackMap(season: number, event_key: string, drivers?: string): Promise<any> {
    const params = drivers ? `?drivers=${drivers}` : '';
    return this.request(`/api/trackmap/${season}/${event_key}${params}`);
  }

  async getAvailableDrivers(season: number, event_key: string): Promise<any> {
    return this.request(`/api/trackmap/${season}/${event_key}/drivers`);
  }

  // Utility method to get asset URL
  getAssetURL(path: string): string {
    return `${this.baseURL}/static/assets/${path}`;
  }
}

// Export singleton instance
export const apiClient = new APIClient();

// Export types for convenience
export type { APIClient };
