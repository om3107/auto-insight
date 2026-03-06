/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { getLogger } from '../logger';
import { CONFIG } from '../config';
import * as Papa from 'papaparse';
import * as XLSX from 'xlsx';
import { DataFrame } from 'danfojs';

const logger = getLogger('ingestion');

export interface IngestionResult {
  df: DataFrame | null;
  status: string;
  metadata: {
    filename: string;
    rowCount: number;
    colCount: number;
    sampled: boolean;
    extension: string;
  };
}

/**
 * Ingests data from a File object.
 * @param file The uploaded file
 * @returns IngestionResult
 */
export const ingestFile = async (file: File): Promise<IngestionResult> => {
  const extension = '.' + (file.name.split('.').pop()?.toLowerCase() || '');
  const fileSizeMB = file.size / (1024 * 1024);
  const SUPPORTED_FORMATS = ['.csv', '.xlsx', '.xls', '.json'];

  logger.info(`Starting ingestion for ${file.name} (${fileSizeMB.toFixed(2)} MB)`);

  if (!SUPPORTED_FORMATS.includes(extension)) {
    return {
      df: null,
      status: `File format not supported! Please upload ${SUPPORTED_FORMATS.join(', ')}`,
      metadata: { filename: file.name, rowCount: 0, colCount: 0, sampled: false, extension }
    };
  }

  if (fileSizeMB > CONFIG.MAX_FILE_SIZE_MB) {
    return {
      df: null,
      status: `File too large (${fileSizeMB.toFixed(2)} MB). Max allowed is ${CONFIG.MAX_FILE_SIZE_MB} MB.`,
      metadata: { filename: file.name, rowCount: 0, colCount: 0, sampled: false, extension }
    };
  }

  try {
    let data: any[] = [];

    if (extension === '.csv') {
      data = await parseCSV(file);
    } else if (extension === '.xlsx' || extension === '.xls') {
      data = await parseExcel(file);
    } else if (extension === '.json') {
      data = await parseJSON(file);
    }

    if (data.length === 0) {
      return {
        df: null,
        status: 'The uploaded file is empty.',
        metadata: { filename: file.name, rowCount: 0, colCount: 0, sampled: false, extension }
      };
    }

    let rowCount = data.length;
    let sampled = false;

    if (rowCount > CONFIG.MAX_ROWS_FULL_LOAD) {
      logger.info(`Dataset exceeds ${CONFIG.MAX_ROWS_FULL_LOAD} rows. Sampling ${CONFIG.SAMPLE_ROWS} rows.`);
      data = sampleData(data, CONFIG.SAMPLE_ROWS);
      rowCount = data.length;
      sampled = true;
    }

    const df = new DataFrame(data);
    
    if (!df || df.columns.length === 0) {
      throw new Error("Failed to create DataFrame. The file might have an invalid format.");
    }

    logger.info(`Successfully loaded ${file.name}. Rows: ${rowCount}, Cols: ${df.columns.length}, Sampled: ${sampled}`);

    return {
      df,
      status: 'Success',
      metadata: {
        filename: file.name,
        rowCount,
        colCount: df.columns.length,
        sampled,
        extension
      }
    };

  } catch (error: any) {
    logger.error(`Error ingesting file ${file.name}: ${error.message}`);
    return {
      df: null,
      status: `Error: ${error.message}`,
      metadata: { filename: file.name, rowCount: 0, colCount: 0, sampled: false, extension }
    };
  }
};

const parseCSV = (file: File): Promise<any[]> => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      skipEmptyLines: true,
      complete: (results) => resolve(results.data),
      error: (error) => reject(error)
    });
  });
};

const parseExcel = (file: File): Promise<any[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const data = new Uint8Array(e.target?.result as ArrayBuffer);
        const workbook = XLSX.read(data, { type: 'array' });
        const firstSheetName = workbook.SheetNames[0];
        const worksheet = workbook.Sheets[firstSheetName];
        const json = XLSX.utils.sheet_to_json(worksheet);
        resolve(json);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = (err) => reject(err);
    reader.readAsArrayBuffer(file);
  });
};

const parseJSON = (file: File): Promise<any[]> => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      try {
        const json = JSON.parse(e.target?.result as string);
        resolve(Array.isArray(json) ? json : [json]);
      } catch (err) {
        reject(err);
      }
    };
    reader.onerror = (err) => reject(err);
    reader.readAsText(file);
  });
};

const sampleData = (data: any[], size: number): any[] => {
  const shuffled = [...data].sort(() => 0.5 - Math.random());
  return shuffled.slice(0, size);
};
