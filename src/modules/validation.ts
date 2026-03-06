/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { getLogger } from '../logger';
import { DataFrame } from 'danfojs';

const logger = getLogger('validation');

/**
 * Validates the target column exists and is suitable for ML.
 * @param df The input DataFrame
 * @param targetCol The name of the target column
 * @returns { isValid: boolean, message: string }
 */
export const validateTarget = (df: DataFrame, targetCol: string): { isValid: boolean; message: string } => {
  if (!targetCol) {
    return { isValid: false, message: 'Target column name is required.' };
  }

  if (!df.columns.includes(targetCol)) {
    return { isValid: false, message: `Target column "${targetCol}" not found in dataset.` };
  }

  const targetData = df[targetCol];
  const uniqueCount = new Set(targetData.values).size;

  if (uniqueCount < 2) {
    return { isValid: false, message: 'Target column must have at least 2 unique values for ML tasks.' };
  }

  return { isValid: true, message: 'Target column is valid.' };
};

/**
 * Removes constant columns (columns with only one unique value).
 * @param df The input DataFrame
 * @returns The cleaned DataFrame and a list of removed columns
 */
export const removeConstantColumns = (df: DataFrame): { df: DataFrame; removedCols: string[] } => {
  const removedCols: string[] = [];
  const columns = df.columns;

  columns.forEach(col => {
    const uniqueCount = new Set(df[col].values).size;
    if (uniqueCount <= 1) {
      removedCols.push(col);
    }
  });

  if (removedCols.length > 0) {
    logger.info(`Removing constant columns: ${removedCols.join(', ')}`);
    return { df: df.drop({ columns: removedCols }), removedCols };
  }

  return { df, removedCols };
};
