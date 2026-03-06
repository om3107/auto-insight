/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { DataFrame, Series } from 'danfojs';
import { getLogger } from '../logger';
import { CONFIG } from '../config';

const logger = getLogger('preprocessing');

export type ColumnType = 'numeric' | 'categorical' | 'datetime' | 'text';

export interface AuditReport {
  missingValues: Record<string, number>;
  droppedColumns: string[];
  outliersDetected: Record<string, number>;
  encodedColumns: string[];
  inferredTypes: Record<string, ColumnType>;
}

/**
 * Infers the type of each column in the DataFrame.
 */
export const inferColumnTypes = (df: DataFrame): Record<string, ColumnType> => {
  const types: Record<string, ColumnType> = {};
  const rowCount = df.shape[0];

  df.columns.forEach(col => {
    const series = df.column(col);
    if (!series || !series.values || series.values.length === 0) {
      types[col] = 'text';
      return;
    }
    const dtype = series.dtype;
    const uniqueCount = new Set(series.values as any[]).size;
    const cardinalityRatio = uniqueCount / rowCount;

    if (dtype === 'float32' || dtype === 'int32') {
      types[col] = 'numeric';
    } else if (cardinalityRatio < CONFIG.CARDINALITY_RATIO_THRESHOLD) {
      types[col] = 'categorical';
    } else if (isDateString(series.values[0])) {
      types[col] = 'datetime';
    } else {
      types[col] = 'text';
    }
  });

  return types;
};

const isDateString = (val: any): boolean => {
  if (typeof val !== 'string') return false;
  const date = new Date(val);
  return !isNaN(date.getTime()) && val.includes('-'); // Simple heuristic
};

/**
 * Coerces columns to their inferred types.
 */
export const fixDtypes = (df: DataFrame, colTypes: Record<string, ColumnType>): DataFrame => {
  let fixedDf = df;
  Object.entries(colTypes).forEach(([col, type]) => {
    try {
      if (type === 'numeric') {
        const colIndex = fixedDf.columns.indexOf(col);
        if (colIndex !== -1) {
          const series = fixedDf.column(col);
          // Convert to numeric, replacing non-numeric with NaN
          const values = series.values.map(v => {
            const n = Number(v);
            return isNaN(n) ? null : n;
          });
          fixedDf = fixedDf.drop({ columns: [col] });
          fixedDf = fixedDf.addColumn(col, new Series(values), { atIndex: colIndex }) as DataFrame;
        }
      } else if (type === 'datetime') {
        // Custom datetime conversion
      }
    } catch (err) {
      logger.warn(`Failed to fix dtype for column ${col}: ${err}`);
    }
  });
  return fixedDf;
};

/**
 * Handles missing values based on column type.
 */
export const handleMissing = (df: DataFrame, colTypes: Record<string, ColumnType>, targetCol?: string): { df: DataFrame; audit: Partial<AuditReport> } => {
  const missingValues: Record<string, number> = {};
  const droppedColumns: string[] = [];
  const rowCount = df.shape[0];
  let cleanedDf = df;

  df.columns.forEach(col => {
    const series = df.column(col);
    const nullCount = (series.values as any[]).filter(v => v === null || v === undefined || (typeof v === 'number' && isNaN(v))).length;
    
    if (nullCount > 0) {
      missingValues[col] = nullCount;
      const missingRatio = nullCount / rowCount;

      if (missingRatio > CONFIG.MISSING_THRESHOLD && col !== targetCol) {
        droppedColumns.push(col);
      } else {
        const type = colTypes[col];
        const colIndex = cleanedDf.columns.indexOf(col);
        if (colIndex !== -1) {
          if (type === 'numeric') {
            try {
              const median = series.median();
              cleanedDf = cleanedDf.drop({ columns: [col] });
              cleanedDf = cleanedDf.addColumn(col, series.fillNa(median), { atIndex: colIndex }) as DataFrame;
            } catch (e) {
              logger.warn(`Could not calculate median for ${col}, skipping imputation.`);
            }
          } else if (type === 'categorical') {
            try {
              const modeArray = series.mode();
              const mode = modeArray.length > 0 ? modeArray[0] : 'Unknown';
              cleanedDf = cleanedDf.drop({ columns: [col] });
              cleanedDf = cleanedDf.addColumn(col, series.fillNa(String(mode)), { atIndex: colIndex }) as DataFrame;
            } catch (e) {
              logger.warn(`Could not calculate mode for ${col}, using 'Unknown' as fallback.`);
              cleanedDf = cleanedDf.drop({ columns: [col] });
              cleanedDf = cleanedDf.addColumn(col, series.fillNa('Unknown'), { atIndex: colIndex }) as DataFrame;
            }
          }
        }
      }
    }
  });

  if (droppedColumns.length > 0) {
    cleanedDf = cleanedDf.drop({ columns: droppedColumns });
  }

  // Final pass to drop any remaining NaT in datetime columns
  // (Simplified for this implementation)

  return { df: cleanedDf, audit: { missingValues, droppedColumns } };
};

/**
 * Handles outliers using Z-score and Winsorization.
 */
export const handleOutliers = (df: DataFrame, colTypes: Record<string, ColumnType>, targetCol?: string): { df: DataFrame; audit: Partial<AuditReport> } => {
  const outliersDetected: Record<string, number> = {};
  let cleanedDf = df;

  df.columns.forEach(col => {
    if (colTypes[col] === 'numeric') {
      const series = df.column(col);
      try {
        const mean = series.mean();
        const std = series.std();
        
        if (typeof mean !== 'number' || typeof std !== 'number' || isNaN(mean) || isNaN(std)) {
          return;
        }

        let count = 0;
        const values = (series.values as number[]).map(v => {
          const z = Math.abs((v - mean) / std);
          if (z > CONFIG.OUTLIER_Z_THRESHOLD) {
            count++;
            // Winsorize: Cap at 1st/99th percentile
            // Simplified: cap at mean +/- 3*std
            return v > mean ? mean + 3 * std : mean - 3 * std;
          }
          return v;
        });

        if (count > 0) {
          outliersDetected[col] = count;
          const colIndex = cleanedDf.columns.indexOf(col);
          if (colIndex !== -1) {
            cleanedDf = cleanedDf.drop({ columns: [col] });
            cleanedDf = cleanedDf.addColumn(col, new Series(values), { atIndex: colIndex }) as DataFrame;
          }
        }
      } catch (e) {
        logger.warn(`Could not handle outliers for ${col}: ${e}`);
      }
    }
  });

  return { df: cleanedDf, audit: { outliersDetected } };
};

/**
 * Encodes categorical variables.
 */
export const encodeCategoricals = (df: DataFrame, colTypes: Record<string, ColumnType>, targetCol?: string): { df: DataFrame; encodedCols: string[] } => {
  const encodedCols: string[] = [];
  let processedDf = df;

  df.columns.forEach(col => {
    if (colTypes[col] === 'categorical' && col !== targetCol) {
      const series = df.column(col);
      const uniqueValues = Array.from(new Set(series.values as any[]));
      
      if (uniqueValues.length <= 2) {
        // Label encoding (factorize)
        const map: Record<string, number> = {};
        uniqueValues.forEach((v, i) => map[String(v)] = i);
        const colIndex = processedDf.columns.indexOf(col);
        if (colIndex !== -1) {
          processedDf = processedDf.drop({ columns: [col] });
          processedDf = processedDf.addColumn(col, new Series((series.values as any[]).map(v => map[String(v)])), { atIndex: colIndex }) as DataFrame;
          encodedCols.push(col);
        }
      } else if (uniqueValues.length <= CONFIG.CATEGORICAL_UNIQUE_LIMIT) {
        // One-hot encoding (get_dummies)
        // Simplified: factorize for now to avoid column explosion in this basic implementation
        const map: Record<string, number> = {};
        uniqueValues.forEach((v, i) => map[String(v)] = i);
        const colIndex = processedDf.columns.indexOf(col);
        if (colIndex !== -1) {
          processedDf = processedDf.drop({ columns: [col] });
          processedDf = processedDf.addColumn(col, new Series((series.values as any[]).map(v => map[String(v)])), { atIndex: colIndex }) as DataFrame;
          encodedCols.push(col);
        }
      } else {
        // Label encoding for high cardinality
        const map: Record<string, number> = {};
        uniqueValues.forEach((v, i) => map[String(v)] = i);
        const colIndex = processedDf.columns.indexOf(col);
        if (colIndex !== -1) {
          processedDf = processedDf.drop({ columns: [col] });
          processedDf = processedDf.addColumn(col, new Series((series.values as any[]).map(v => map[String(v)])), { atIndex: colIndex }) as DataFrame;
          encodedCols.push(col);
        }
      }
    }
  });

  return { df: processedDf, encodedCols };
};

/**
 * Full preprocessing pipeline.
 */
export const fullPreprocessPipeline = async (df: DataFrame, targetCol: string): Promise<{ df: DataFrame; audit: AuditReport }> => {
  logger.info('Starting full preprocessing pipeline');
  
  const inferredTypes = inferColumnTypes(df);
  let processedDf = fixDtypes(df, inferredTypes);
  
  const { df: dfMissing, audit: auditMissing } = handleMissing(processedDf, inferredTypes, targetCol);
  const { df: dfOutliers, audit: auditOutliers } = handleOutliers(dfMissing, inferredTypes, targetCol);
  const { df: dfEncoded, encodedCols } = encodeCategoricals(dfOutliers, inferredTypes, targetCol);

  // Final step: Drop any columns that are not numeric and not the target column
  let finalDf = dfEncoded;
  const columnsToDrop: string[] = [];
  finalDf.columns.forEach(col => {
    if (col !== targetCol) {
      const series = finalDf.column(col);
      const dtype = series.dtype;
      const firstVal = series.values[0];
      const isNumeric = (dtype.includes('float') || dtype.includes('int')) && typeof firstVal === 'number';
      const isDatetime = inferredTypes[col] === 'datetime';
      
      if (!isNumeric && !isDatetime) {
        columnsToDrop.push(col);
      }
    }
  });

  if (columnsToDrop.length > 0) {
    logger.info(`Dropping non-numeric columns before ML: ${columnsToDrop.join(', ')}`);
    finalDf = finalDf.drop({ columns: columnsToDrop });
  }

  const fullAudit: AuditReport = {
    missingValues: auditMissing.missingValues || {},
    droppedColumns: [...(auditMissing.droppedColumns || []), ...columnsToDrop],
    outliersDetected: auditOutliers.outliersDetected || {},
    encodedColumns: encodedCols,
    inferredTypes
  };

  logger.info('Preprocessing pipeline complete');
  return { df: finalDf, audit: fullAudit };
};
