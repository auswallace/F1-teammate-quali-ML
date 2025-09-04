# F1 Teammate Qualifying Dashboard

A modern React dashboard for F1 teammate qualifying predictions with neon track maps.

## 🚀 Features

- **Neon Track Maps**: Real-time track visualization using FastF1 telemetry
- **ML Predictions**: View machine learning predictions for teammate battles
- **F1-Style Design**: Dark theme with F1 branding and colors
- **Responsive Layout**: Works on desktop and mobile devices
- **Real-time Data**: Live updates from the FastAPI backend

## 🛠️ Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS + Custom F1 design tokens
- **Backend**: FastAPI (Python)
- **Data**: FastF1 telemetry + ML predictions
- **Build Tool**: Vite

## 📦 Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Start development server:**
   ```bash
   npm run dev
   ```

3. **Build for production:**
   ```bash
   npm run build
   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the dashboard directory:

```env
# API base URL (default: http://localhost:8000)
VITE_API_BASE=http://localhost:8000
```

### Backend Requirements

The dashboard requires the FastAPI backend to be running. See the main project README for backend setup instructions.

## 🎨 Design System

The dashboard uses a shared design system defined in `../ui/tokens.css`:

- **Colors**: F1-inspired palette with deep navy, cyan accents
- **Typography**: Inter font family for modern readability
- **Components**: Reusable card, row, and button styles
- **Responsive**: Mobile-first design with Tailwind utilities

## 🏁 Track Maps

Track maps are generated from FastF1 telemetry data:

1. **Center Line**: Main track outline with outer glow effect
2. **Driver Paths**: Individual driver lap comparisons with neon glow
3. **Fallbacks**: Static SVG fallbacks when telemetry unavailable
4. **Caching**: Results cached for performance

## 📱 Components

- **SeasonEventPicker**: Dropdown selectors for season/event
- **TrackMap**: SVG-based track visualization with neon effects
- **PredictionsTable**: Team-by-team prediction results
- **StatBadge**: Metric display with variant styling

## 🚀 Development

### Project Structure

```
src/
├── components/          # React components
├── api/                # API client and types
├── types.ts            # TypeScript interfaces
├── App.tsx             # Main application
└── main.tsx            # Entry point
```

### Adding New Features

1. **New API endpoints**: Add to `src/api/client.ts`
2. **New components**: Create in `src/components/`
3. **New types**: Add to `src/types.ts`
4. **Styling**: Use Tailwind classes + F1 design tokens

### Styling Guidelines

- Use Tailwind for layout and spacing
- Use F1 design tokens for colors and components
- Follow mobile-first responsive design
- Maintain consistent spacing and typography

## 🔍 Troubleshooting

### Common Issues

1. **Backend not running**: Ensure FastAPI server is started on port 8000
2. **CORS errors**: Check backend CORS configuration
3. **Missing assets**: Verify backend static file serving
4. **Build errors**: Check TypeScript compilation

### Debug Mode

Enable debug logging in the browser console:

```typescript
// In src/api/client.ts
console.log('API Request:', endpoint, options);
```

## 📚 API Reference

The dashboard communicates with these backend endpoints:

- `GET /api/status` - System health and status
- `GET /api/events` - Available seasons and events
- `GET /api/predictions/{season}/{event_key}` - ML predictions
- `GET /api/trackmap/{season}/{event_key}` - Track map data

## 🎯 Performance

- **Lazy loading**: Components load data on demand
- **Caching**: Track maps cached for repeated requests
- **Optimization**: SVG paths optimized for rendering
- **Responsive**: Efficient mobile and desktop layouts

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add TypeScript types for new features
3. Test on both desktop and mobile devices
4. Update this README for significant changes

## 📄 License

Part of the F1 Teammate Qualifying project. See main project README for details.
