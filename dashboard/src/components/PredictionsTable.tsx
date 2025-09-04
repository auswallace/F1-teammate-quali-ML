import React from 'react';
import { PredictionRow } from '../types';
import { apiClient } from '../api/client';

interface PredictionsTableProps {
  predictions: PredictionRow[];
  summary: {
    total_teams: number;
    model_accuracy: number;
    model_correct: number;
    model_incorrect: number;
  };
}

const PredictionsTable: React.FC<PredictionsTableProps> = ({ predictions, summary }) => {
  const getDriverImageUrl = (driverId: string) => {
    // Try to get driver image from assets
    return apiClient.getAssetURL(`headshots/${driverId.toLowerCase()}.png`);
  };

  const getTeamImageUrl = (teamName: string) => {
    // Try to get team logo from assets
    const teamKey = teamName.toLowerCase().replace(/\s+/g, '_');
    return apiClient.getAssetURL(`logos/${teamKey}.png`);
  };

  const getModelPickDisplay = (prediction: PredictionRow) => {
    const isDriverA = prediction.model_pick === prediction.driver_a.name;
    const driver = isDriverA ? prediction.driver_a : prediction.driver_b;
    
    return {
      name: driver.name,
      confidence: driver.model_confidence,
      isDriverA
    };
  };

  const getRowStatus = (prediction: PredictionRow) => {
    if (prediction.model_correct === null) return '';
    return prediction.model_correct ? 'ok' : 'bad';
  };

  return (
    <div className="w-full">
      {/* Table Header */}
      <div className="f1-row font-semibold text-f1-accent bg-f1-panel rounded-t-lg">
        <div className="flex-1">Team</div>
        <div className="flex-1">Driver A</div>
        <div className="flex-1">Driver B</div>
        <div className="flex-1">Model Pick</div>
        <div className="w-24 text-center">Result</div>
      </div>

      {/* Table Rows */}
      {predictions.map((prediction, index) => {
        const modelPick = getModelPickDisplay(prediction);
        const rowStatus = getRowStatus(prediction);
        
        return (
          <div key={index} className={`f1-row ${rowStatus}`}>
            {/* Team */}
            <div className="flex-1 flex items-center gap-3">
              <img
                src={getTeamImageUrl(prediction.team)}
                alt={prediction.team}
                className="circle-40"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                }}
              />
              <span className="font-medium">{prediction.team}</span>
            </div>

            {/* Driver A */}
            <div className="flex-1 flex items-center gap-3">
              <img
                src={getDriverImageUrl(prediction.driver_a.id)}
                alt={prediction.driver_a.name}
                className="circle-48"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                }}
              />
              <div>
                <div className="font-medium">{prediction.driver_a.name}</div>
                <div className="text-sm text-f1-muted">{prediction.driver_a.id}</div>
              </div>
            </div>

            {/* Driver B */}
            <div className="flex-1 flex items-center gap-3">
              <img
                src={getDriverImageUrl(prediction.driver_b.id)}
                alt={prediction.driver_b.name}
                className="circle-48"
                onError={(e) => {
                  const target = e.target as HTMLImageElement;
                  target.style.display = 'none';
                }}
              />
              <div>
                <div className="font-medium">{prediction.driver_b.name}</div>
                <div className="text-sm text-f1-muted">{prediction.driver_b.id}</div>
              </div>
            </div>

            {/* Model Pick */}
            <div className="flex-1">
              <div className="font-medium text-f1-accent">{modelPick.name}</div>
              <div className="text-sm text-f1-muted">
                {(modelPick.confidence * 100).toFixed(1)}% confidence
              </div>
              {modelPick.isDriverA ? (
                <span className="pill-accent text-xs">Driver A</span>
              ) : (
                <span className="pill-alt text-xs">Driver B</span>
              )}
            </div>

            {/* Result */}
            <div className="w-24 text-center">
              {prediction.model_correct !== null ? (
                prediction.model_correct ? (
                  <div className="text-f1-good font-semibold">✅</div>
                ) : (
                  <div className="text-f1-bad font-semibold">❌</div>
                )
              ) : (
                <div className="text-f1-muted text-sm">—</div>
              )}
            </div>
          </div>
        );
      })}

      {/* Table Footer */}
      <div className="f1-row bg-f1-panel rounded-b-lg border-t border-f1-border">
        <div className="flex-1 text-sm text-f1-muted">
          {summary.total_teams} teams
        </div>
        <div className="flex-1"></div>
        <div className="flex-1"></div>
        <div className="flex-1 text-sm text-f1-muted">
          Model accuracy: {(summary.model_accuracy * 100).toFixed(1)}%
        </div>
        <div className="w-24 text-center text-sm text-f1-muted">
          {summary.model_correct}✅ / {summary.model_incorrect}❌
        </div>
      </div>
    </div>
  );
};

export default PredictionsTable;
