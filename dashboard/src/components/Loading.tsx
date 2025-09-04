import React from 'react';

interface LoadingProps {
  message?: string;
}

const Loading: React.FC<LoadingProps> = ({ message = "Loading..." }) => {
  return (
    <div className="min-h-screen bg-f1-bg text-f1-text flex items-center justify-center">
      <div className="text-center">
        <div className="w-16 h-16 border-4 border-f1-accent border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
        <p className="text-f1-muted text-lg">{message}</p>
      </div>
    </div>
  );
};

export default Loading;
