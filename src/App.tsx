/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback, useMemo } from 'react';
import { 
  Upload, FileText, BarChart3, Settings, AlertCircle, CheckCircle2, 
  Download, Play, Database, Layers, PieChart, Activity, TrendingUp, Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { DataFrame } from 'danfojs';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { ingestFile, IngestionResult } from './modules/ingestion';
import { fullPreprocessPipeline, AuditReport } from './modules/preprocessing';
import { 
  detectMLTask, runRegression, runClassification, runClustering, 
  runAnomalyDetection, runForecasting, MLResult 
} from './modules/analysis';
import { 
  ActualVsPredictedChart, AnomalyChart, ClusterChart, 
  DistributionChart, FeatureImportanceChart, ForecastChart 
} from './modules/visualization';
import { generateReport } from './modules/reporting';
import { validateTarget, removeConstantColumns } from './modules/validation';
import { getLogger } from './logger';

const logger = getLogger('App');

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [ingestionResult, setIngestionResult] = useState<IngestionResult | null>(null);
  const [targetCol, setTargetCol] = useState<string>('');
  const [runAnomaly, setRunAnomaly] = useState(true);
  const [runClusteringTask, setRunClusteringTask] = useState(true);
  const [isProcessing, setIsProcessing] = useState(false);
  const [audit, setAudit] = useState<AuditReport | null>(null);
  const [mlResults, setMlResults] = useState<MLResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'data' | 'preprocess' | 'analysis' | 'report'>('data');

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const uploadedFile = e.target.files?.[0];
    if (!uploadedFile) return;

    setFile(uploadedFile);
    setIsProcessing(true);
    setError(null);
    setMlResults([]);
    setAudit(null);

    try {
      const result = await ingestFile(uploadedFile);
      setIngestionResult(result);
      if (!result.df) {
        setError(result.status);
      } else {
        // Default target to last column
        setTargetCol(result.df.columns[result.df.columns.length - 1]);
      }
    } catch (err: any) {
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  const runAnalysis = async () => {
    if (!ingestionResult?.df) return;
    
    setIsProcessing(true);
    setError(null);
    
    try {
      let df = ingestionResult.df;
      
      // 1. Validation
      const targetValidation = validateTarget(df, targetCol);
      if (!targetValidation.isValid) {
        throw new Error(targetValidation.message);
      }

      // 2. Preprocessing
      const { df: preprocessedDf, audit: preprocessAudit } = await fullPreprocessPipeline(df, targetCol);
      setAudit(preprocessAudit);

      // 3. Analysis
      const results: MLResult[] = [];
      const task = detectMLTask(preprocessedDf, targetCol);
      
      if (task === 'regression') {
        results.push(runRegression(preprocessedDf, targetCol));
      } else {
        results.push(runClassification(preprocessedDf, targetCol));
      }

      // Optional tasks
      if (runClusteringTask) {
        const clusteringDf = preprocessedDf.columns.includes(targetCol) 
          ? preprocessedDf.drop({ columns: [targetCol] }) 
          : preprocessedDf;
        results.push(runClustering(clusteringDf));
      }
      
      if (runAnomaly) {
        const anomalyDf = preprocessedDf.columns.includes(targetCol) 
          ? preprocessedDf.drop({ columns: [targetCol] }) 
          : preprocessedDf;
        results.push(runAnomalyDetection(anomalyDf));
      }

      // Forecasting Detection
      const dateCols = Object.entries(preprocessAudit.inferredTypes)
        .filter(([col, type]) => type === 'datetime')
        .map(([col]) => col);
      
      if (dateCols.length > 0 && task === 'regression') {
        results.push(runForecasting(preprocessedDf, dateCols[0], targetCol));
      }

      setMlResults(results);
      setActiveTab('analysis');
    } catch (err: any) {
      setError(err.message);
      logger.error('Analysis failed', err);
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadReport = () => {
    if (!ingestionResult || !audit || mlResults.length === 0) return;
    const reportUrl = generateReport(
      ingestionResult.metadata.filename,
      {
        rowCount: ingestionResult.metadata.rowCount,
        colCount: ingestionResult.metadata.colCount,
        filename: ingestionResult.metadata.filename
      },
      audit,
      mlResults
    );
    const link = document.createElement('a');
    link.href = reportUrl;
    link.download = `Auto-Insight-Report-${ingestionResult.metadata.filename.split('.')[0]}.pdf`;
    link.click();
  };

  return (
    <div className="min-h-screen bg-[#F8F9FA] flex font-sans text-slate-900">
      {/* Sidebar */}
      <aside className="w-80 bg-white border-r border-slate-200 flex flex-col sticky top-0 h-screen">
        <div className="p-6 border-bottom border-slate-100">
          <div className="flex items-center gap-3 mb-2">
            <div className="bg-indigo-600 p-2 rounded-lg">
              <Activity className="text-white w-6 h-6" />
            </div>
            <h1 className="text-xl font-bold tracking-tight text-slate-900">Auto-Insight</h1>
          </div>
          <p className="text-xs text-slate-500 font-medium uppercase tracking-wider">Analytics Engine v1.0</p>
        </div>

        <div className="flex-1 overflow-y-auto p-6 space-y-8">
          {/* File Upload Section */}
          <section className="space-y-4">
            <label className="block text-sm font-semibold text-slate-700">Data Source</label>
            <div className="relative group">
              <input
                type="file"
                onChange={handleFileUpload}
                accept=".csv,.xlsx,.xls,.json"
                className="absolute inset-0 w-full h-full opacity-0 cursor-pointer z-10"
              />
              <div className={cn(
                "border-2 border-dashed rounded-xl p-6 transition-all duration-200 flex flex-col items-center text-center gap-2",
                file ? "border-indigo-200 bg-indigo-50/30" : "border-slate-200 group-hover:border-indigo-300 group-hover:bg-slate-50"
              )}>
                <Upload className={cn("w-8 h-8", file ? "text-indigo-600" : "text-slate-400")} />
                <span className="text-sm font-medium text-slate-600">
                  {file ? file.name : "Upload CSV, Excel, or JSON"}
                </span>
                <span className="text-[10px] text-slate-400 uppercase tracking-tight">Max 200MB</span>
              </div>
            </div>
          </section>

          {/* Configuration Section */}
          <AnimatePresence>
            {ingestionResult?.df && (
              <motion.section
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: 10 }}
                className="space-y-4"
              >
                <label className="block text-sm font-semibold text-slate-700">Configuration</label>
                <div className="space-y-3">
                  <div>
                    <span className="text-xs text-slate-500 mb-1 block">Target Column</span>
                    <select
                      value={targetCol}
                      onChange={(e) => setTargetCol(e.target.value)}
                      className="w-full bg-white border border-slate-200 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
                    >
                      {ingestionResult.df.columns.map(col => (
                        <option key={col} value={col}>{col}</option>
                      ))}
                    </select>
                  </div>

                  <div className="space-y-2">
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={runAnomaly}
                        onChange={(e) => setRunAnomaly(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-xs font-medium text-slate-600 group-hover:text-slate-900 transition-colors">Run Anomaly Detection</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input
                        type="checkbox"
                        checked={runClusteringTask}
                        onChange={(e) => setRunClusteringTask(e.target.checked)}
                        className="w-4 h-4 rounded border-slate-300 text-indigo-600 focus:ring-indigo-500"
                      />
                      <span className="text-xs font-medium text-slate-600 group-hover:text-slate-900 transition-colors">Run Clustering</span>
                    </label>
                  </div>
                  
                  <button
                    onClick={runAnalysis}
                    disabled={isProcessing}
                    className="w-full bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 text-white font-semibold py-2.5 rounded-lg flex items-center justify-center gap-2 transition-all shadow-sm shadow-indigo-200"
                  >
                    {isProcessing ? (
                      <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                    ) : (
                      <Play className="w-4 h-4 fill-current" />
                    )}
                    Run Engine
                  </button>
                </div>
              </motion.section>
            )}
          </AnimatePresence>
        </div>

        {/* Footer Info */}
        <div className="p-6 bg-slate-50 border-t border-slate-200">
          <div className="flex items-center gap-2 text-slate-500 mb-1">
            <Info className="w-4 h-4" />
            <span className="text-xs font-medium">System Status</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
            <span className="text-[10px] font-bold text-slate-700 uppercase tracking-wider">Operational</span>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Header Nav */}
        <header className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-0 z-20">
          <nav className="flex gap-8">
            {[
              { id: 'data', label: 'Data Explorer', icon: Database },
              { id: 'preprocess', label: 'Preprocessing', icon: Layers },
              { id: 'analysis', label: 'ML Insights', icon: BarChart3 },
              { id: 'report', label: 'Reporting', icon: FileText },
            ].map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id as any)}
                className={cn(
                  "flex items-center gap-2 py-2 text-sm font-semibold transition-all relative",
                  activeTab === tab.id ? "text-indigo-600" : "text-slate-500 hover:text-slate-700"
                )}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
                {activeTab === tab.id && (
                  <motion.div layoutId="activeTab" className="absolute -bottom-[17px] left-0 right-0 h-0.5 bg-indigo-600" />
                )}
              </button>
            ))}
          </nav>

          <button
            onClick={downloadReport}
            disabled={mlResults.length === 0}
            className="flex items-center gap-2 bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 text-white px-4 py-2 rounded-lg text-sm font-semibold transition-all"
          >
            <Download className="w-4 h-4" />
            Export PDF
          </button>
        </header>

        {/* Content Area */}
        <div className="flex-1 p-8 overflow-y-auto">
          <AnimatePresence mode="wait">
            {error && (
              <motion.div
                initial={{ opacity: 0, y: -10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -10 }}
                className="bg-red-50 border border-red-200 rounded-xl p-4 flex items-start gap-3 mb-8"
              >
                <AlertCircle className="text-red-600 w-5 h-5 mt-0.5" />
                <div>
                  <h3 className="text-sm font-bold text-red-900">Engine Error</h3>
                  <p className="text-sm text-red-700">{error}</p>
                </div>
              </motion.div>
            )}

            {!file && (
              <motion.div
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto"
              >
                <div className="bg-indigo-50 p-6 rounded-full mb-6">
                  <Database className="w-12 h-12 text-indigo-600" />
                </div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">Welcome to Auto-Insight</h2>
                <p className="text-slate-500 mb-8">
                  Upload your dataset to start the automated analysis engine. We'll handle preprocessing, 
                  ML task detection, and generate a comprehensive insight report.
                </p>
                <div className="grid grid-cols-2 gap-4 w-full">
                  <div className="p-4 bg-white border border-slate-200 rounded-xl text-left">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500 mb-2" />
                    <h4 className="text-sm font-bold text-slate-900">Auto-ML</h4>
                    <p className="text-xs text-slate-500">Regression & Classification</p>
                  </div>
                  <div className="p-4 bg-white border border-slate-200 rounded-xl text-left">
                    <CheckCircle2 className="w-5 h-5 text-emerald-500 mb-2" />
                    <h4 className="text-sm font-bold text-slate-900">Smart Cleaning</h4>
                    <p className="text-xs text-slate-500">Outliers & Missing Values</p>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'data' && ingestionResult?.df && (
              <motion.div
                key="data-tab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-8"
              >
                <div className="grid grid-cols-4 gap-6">
                  {[
                    { label: 'Total Rows', value: ingestionResult.metadata.rowCount, icon: Database },
                    { label: 'Total Columns', value: ingestionResult.metadata.colCount, icon: Layers },
                    { label: 'File Type', value: ingestionResult.metadata.extension.toUpperCase(), icon: FileText },
                    { label: 'Sampling', value: ingestionResult.metadata.sampled ? 'Enabled' : 'Disabled', icon: Activity },
                  ].map((stat) => (
                    <div key={stat.label} className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-xs font-bold text-slate-500 uppercase tracking-wider">{stat.label}</span>
                        <stat.icon className="w-4 h-4 text-indigo-600" />
                      </div>
                      <div className="text-2xl font-bold text-slate-900">{stat.value}</div>
                    </div>
                  ))}
                </div>

                <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
                  <div className="px-6 py-4 border-b border-slate-100 bg-slate-50/50 flex items-center justify-between">
                    <h3 className="font-bold text-slate-900">Dataset Preview (First 10 Rows)</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="bg-slate-50 text-slate-500 font-bold uppercase text-[10px] tracking-wider">
                        <tr>
                          {ingestionResult.df.columns.map(col => (
                            <th key={col} className="px-6 py-3 border-b border-slate-100">{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-slate-100">
                        {ingestionResult.df.head(10).values.map((row: any, i: number) => (
                          <tr key={i} className="hover:bg-slate-50 transition-colors">
                            {row.map((cell: any, j: number) => (
                              <td key={j} className="px-6 py-4 text-slate-600 whitespace-nowrap">
                                {typeof cell === 'number' ? cell.toFixed(2) : String(cell)}
                              </td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'preprocess' && audit && (
              <motion.div
                key="preprocess-tab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-8"
              >
                <div className="grid grid-cols-3 gap-6">
                  <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                    <h4 className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                      <AlertCircle className="w-4 h-4 text-amber-500" />
                      Missing Values
                    </h4>
                    <div className="space-y-3">
                      {Object.entries(audit.missingValues).length > 0 ? (
                        Object.entries(audit.missingValues).map(([col, count]) => (
                          <div key={col} className="flex justify-between text-sm">
                            <span className="text-slate-600">{col}</span>
                            <span className="font-bold text-slate-900">{count}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-slate-500 italic">No missing values detected.</p>
                      )}
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                    <h4 className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                      <Activity className="w-4 h-4 text-indigo-500" />
                      Outliers Capped
                    </h4>
                    <div className="space-y-3">
                      {Object.entries(audit.outliersDetected).length > 0 ? (
                        Object.entries(audit.outliersDetected).map(([col, count]) => (
                          <div key={col} className="flex justify-between text-sm">
                            <span className="text-slate-600">{col}</span>
                            <span className="font-bold text-slate-900">{count}</span>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-slate-500 italic">No outliers detected.</p>
                      )}
                    </div>
                  </div>

                  <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm">
                    <h4 className="text-sm font-bold text-slate-900 mb-4 flex items-center gap-2">
                      <Layers className="w-4 h-4 text-emerald-500" />
                      Feature Engineering
                    </h4>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-600">Encoded Columns</span>
                        <span className="font-bold text-slate-900">{audit.encodedColumns.length}</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-slate-600">Dropped Columns</span>
                        <span className="font-bold text-slate-900">{audit.droppedColumns.length}</span>
                      </div>
                    </div>
                  </div>
                </div>

                <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
                  <h3 className="font-bold text-slate-900 mb-6">Inferred Column Types</h3>
                  <div className="grid grid-cols-4 gap-4">
                    {Object.entries(audit.inferredTypes).map(([col, type]) => (
                      <div key={col} className="p-3 bg-slate-50 rounded-xl border border-slate-100">
                        <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-1">{type}</div>
                        <div className="text-sm font-bold text-slate-700 truncate">{col}</div>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.div>
            )}

            {activeTab === 'analysis' && mlResults.length > 0 && (
              <motion.div
                key="analysis-tab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="space-y-8 pb-12"
              >
                {mlResults.map((result, i) => (
                  <div key={i} className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm">
                    <div className="flex items-center justify-between mb-8">
                      <div>
                        <h3 className="text-xl font-bold text-slate-900 capitalize">{result.task} Analysis</h3>
                        <p className="text-sm text-slate-500">Automated model performance and insights</p>
                      </div>
                      <div className="flex gap-4">
                        {Object.entries(result.metrics).map(([key, val]) => (
                          <div key={key} className="text-right">
                            <div className="text-[10px] font-bold text-slate-400 uppercase tracking-wider">{key}</div>
                            <div className="text-lg font-bold text-indigo-600">
                              {typeof val === 'number' ? val.toFixed(4) : val}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="grid grid-cols-2 gap-8">
                      {result.task === 'regression' && result.actuals && result.predictions && (
                        <>
                          <div className="space-y-4">
                            <h4 className="text-sm font-bold text-slate-700">Actual vs Predicted</h4>
                            <ActualVsPredictedChart actuals={result.actuals} predictions={result.predictions} />
                          </div>
                          <div className="space-y-4">
                            <h4 className="text-sm font-bold text-slate-700">Feature Importance</h4>
                            <FeatureImportanceChart importance={result.featureImportance || {}} />
                          </div>
                        </>
                      )}

                      {result.task === 'classification' && result.actuals && result.predictions && (
                        <>
                          <div className="space-y-4">
                            <h4 className="text-sm font-bold text-slate-700">Prediction Distribution</h4>
                            <DistributionChart data={result.predictions.map(p => typeof p === 'number' ? p : 0)} col="Predictions" />
                          </div>
                          <div className="space-y-4">
                            <h4 className="text-sm font-bold text-slate-700">Feature Importance</h4>
                            <FeatureImportanceChart importance={result.featureImportance || {}} />
                          </div>
                        </>
                      )}

                      {result.task === 'clustering' && result.extra?.X && result.predictions && (
                        <div className="col-span-2 space-y-4">
                          <h4 className="text-sm font-bold text-slate-700">Cluster Visualization (2D Projection)</h4>
                          <ClusterChart X={result.extra.X} labels={result.predictions} />
                        </div>
                      )}

                      {result.task === 'anomaly' && result.extra?.X && result.predictions && (
                        <div className="col-span-2 space-y-4">
                          <h4 className="text-sm font-bold text-slate-700">Anomaly Detection Map</h4>
                          <AnomalyChart X={result.extra.X} mask={result.predictions} />
                        </div>
                      )}

                      {result.task === 'forecast' && result.extra?.historicalDates && result.predictions && result.actuals && (
                        <div className="col-span-2 space-y-4">
                          <h4 className="text-sm font-bold text-slate-700">Time Series Forecast</h4>
                          <ForecastChart 
                            historical={result.actuals} 
                            forecast={result.predictions} 
                            historicalDates={result.extra.historicalDates} 
                            futureDates={result.extra.futureDates} 
                          />
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </motion.div>
            )}

            {activeTab === 'report' && (
              <motion.div
                key="report-tab"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto"
              >
                <div className="bg-slate-900 p-6 rounded-full mb-6">
                  <FileText className="w-12 h-12 text-white" />
                </div>
                <h2 className="text-2xl font-bold text-slate-900 mb-2">Ready for Export</h2>
                <p className="text-slate-500 mb-8">
                  Your comprehensive analysis report is ready. It includes dataset metadata, 
                  preprocessing audit logs, and detailed ML performance metrics.
                </p>
                <button
                  onClick={downloadReport}
                  disabled={mlResults.length === 0}
                  className="w-full bg-slate-900 hover:bg-slate-800 disabled:bg-slate-300 text-white font-bold py-4 rounded-xl flex items-center justify-center gap-3 transition-all shadow-lg shadow-slate-200"
                >
                  <Download className="w-5 h-5" />
                  Download Insight Report
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}
