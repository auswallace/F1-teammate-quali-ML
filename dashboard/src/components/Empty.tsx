import React from 'react';

interface EmptyProps {
  message?: string;
}

const Empty: React.FC<EmptyProps> = ({ message = "No data available" }) => {
  return (
    <div className="min-h-screen bg-f1-bg text-f1-text flex items-center justify-center">
      <div className="text-center">
        <div className="text-6xl mb-4">🏁</div>
        <h2 className="text-2xl font-bold text-f1-muted mb-2">Dashboard Empty</h2>
        <p className="text-f1-muted">{message}</p>
      </div>
    </div>
  );
};

export default Empty;
