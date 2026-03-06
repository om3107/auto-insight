/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
  LineChart, Line, ScatterChart, Scatter, Cell, PieChart, Pie
} from 'recharts';

interface ChartProps {
  data: any[];
  title?: string;
}

export const CorrelationHeatmap: React.FC<ChartProps> = ({ data }) => {
  // Recharts doesn't have a native heatmap, but we can simulate it with a grid of colored cells
  // or just use a simple bar chart for correlations if needed.
  // For now, let's show a simple bar chart of top correlations.
  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data.slice(0, 10)} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" domain={[-1, 1]} />
          <YAxis dataKey="name" type="category" width={100} />
          <Tooltip />
          <Bar dataKey="value" fill="#8884d8" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const DistributionChart: React.FC<{ data: number[], col: string }> = ({ data, col }) => {
  // Create histogram bins
  const minVal: number = data.length > 0 ? Math.min(...data) : 0;
  const maxVal: number = data.length > 0 ? Math.max(...data) : 0;
  const binCount = 10;
  const binWidth: number = data.length > 0 ? (maxVal - minVal) / binCount : 0;
  
  const bins = [];
  for (let index = 0; index < binCount; index++) {
    const startValue = (minVal as any) + (index * (binWidth as any));
    const endValue = (minVal as any) + ((index + 1) * (binWidth as any));
    bins.push({
      range: `${Number(startValue).toFixed(2)} - ${Number(endValue).toFixed(2)}`,
      count: data.filter(v => v >= startValue && v < endValue).length
    });
  }

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={bins}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="range" />
          <YAxis />
          <Tooltip />
          <Bar dataKey="count" fill="#82ca9d" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const FeatureImportanceChart: React.FC<{ importance: Record<string, number> }> = ({ importance }) => {
  const data = Object.entries(importance)
    .map(([name, value]) => ({ name, value }))
    .sort((a: any, b: any) => Number(b.value) - Number(a.value))
    .slice(0, 10);

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} layout="vertical">
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis type="number" />
          <YAxis dataKey="name" type="category" width={100} />
          <Tooltip />
          <Bar dataKey="value" fill="#ff7300" />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ActualVsPredictedChart: React.FC<{ actuals: number[], predictions: number[] }> = ({ actuals, predictions }) => {
  const data = actuals.map((a, i) => ({ actual: Number(a), predicted: Number(predictions[i]) }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid />
          <XAxis type="number" dataKey="actual" name="Actual" />
          <YAxis type="number" dataKey="predicted" name="Predicted" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter name="Predictions" data={data} fill="#8884d8" />
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ForecastChart: React.FC<{ historical: number[], forecast: number[], historicalDates: number[], futureDates: number[] }> = ({ historical, forecast, historicalDates, futureDates }) => {
  const data = [
    ...historical.map((v, i) => ({ date: new Date(historicalDates[i]).toLocaleDateString(), value: v, type: 'historical' })),
    ...forecast.map((v, i) => ({ date: new Date(futureDates[i]).toLocaleDateString(), value: v, type: 'forecast' }))
  ];

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Legend />
          <Line type="monotone" dataKey="value" stroke="#8884d8" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export const AnomalyChart: React.FC<{ X: number[][], mask: boolean[] }> = ({ X, mask }) => {
  // Use first two dimensions for 2D scatter
  const data = X.map((row, i) => ({
    x: row[0],
    y: row[1],
    isAnomaly: mask[i]
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name="Feature 1" />
          <YAxis type="number" dataKey="y" name="Feature 2" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter name="Data Points" data={data}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.isAnomaly ? '#ff0000' : '#8884d8'} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

export const ClusterChart: React.FC<{ X: number[][], labels: number[] }> = ({ X, labels }) => {
  const colors = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#0088fe', '#00c49f'];
  const data = X.map((row, i) => ({
    x: row[0],
    y: row[1],
    cluster: labels[i]
  }));

  return (
    <div className="h-64 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <ScatterChart>
          <CartesianGrid />
          <XAxis type="number" dataKey="x" name="Feature 1" />
          <YAxis type="number" dataKey="y" name="Feature 2" />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Scatter name="Clusters" data={data}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={colors[entry.cluster % colors.length]} />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};
