/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { jsPDF } from 'jspdf';
import { AuditReport } from './preprocessing';
import { MLResult } from './analysis';

/**
 * Generates a PDF report of the analysis.
 */
export const generateReport = (
  filename: string,
  datasetInfo: { rowCount: number; colCount: number; filename: string },
  audit: AuditReport,
  mlResults: MLResult[]
): string => {
  const doc = new jsPDF();
  let y = 20;

  // Header
  doc.setFontSize(22);
  doc.text('Auto-Insight Analysis Report', 20, y);
  y += 10;
  doc.setFontSize(12);
  doc.text(`Generated on: ${new Date().toLocaleString()}`, 20, y);
  y += 15;

  // Dataset Info
  doc.setFontSize(16);
  doc.setFillColor(240, 240, 240);
  doc.rect(20, y - 5, 170, 10, 'F');
  doc.text('Dataset Information', 25, y);
  y += 10;
  doc.setFontSize(12);
  doc.text(`Filename: ${datasetInfo.filename}`, 25, y);
  y += 7;
  doc.text(`Rows: ${datasetInfo.rowCount}`, 25, y);
  y += 7;
  doc.text(`Columns: ${datasetInfo.colCount}`, 25, y);
  y += 15;

  // Preprocessing Audit
  doc.setFontSize(16);
  doc.setFillColor(240, 240, 240);
  doc.rect(20, y - 5, 170, 10, 'F');
  doc.text('Preprocessing Audit', 25, y);
  y += 10;
  doc.setFontSize(12);
  doc.text(`Dropped Columns: ${audit.droppedColumns.length > 0 ? audit.droppedColumns.join(', ') : 'None'}`, 25, y);
  y += 7;
  doc.text(`Missing Values Handled: ${Object.keys(audit.missingValues).length}`, 25, y);
  y += 7;
  doc.text(`Outliers Detected: ${Object.values(audit.outliersDetected).reduce((a, b) => a + b, 0)}`, 25, y);
  y += 15;

  // ML Results
  mlResults.forEach((result, index) => {
    if (y > 250) {
      doc.addPage();
      y = 20;
    }
    doc.setFontSize(16);
    doc.setFillColor(240, 240, 240);
    doc.rect(20, y - 5, 170, 10, 'F');
    doc.text(`Task ${index + 1}: ${result.task.toUpperCase()}`, 25, y);
    y += 10;
    doc.setFontSize(12);
    Object.entries(result.metrics).forEach(([key, val]) => {
      doc.text(`${key}: ${typeof val === 'number' ? val.toFixed(4) : val}`, 25, y);
      y += 7;
    });
    y += 10;
  });

  // Save as blob URL
  const blob = doc.output('blob');
  return URL.createObjectURL(blob);
};
