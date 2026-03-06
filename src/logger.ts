/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

type LogLevel = 'info' | 'warn' | 'error' | 'debug';

class Logger {
  private name: string;

  constructor(name: string) {
    this.name = name;
  }

  private log(level: LogLevel, message: string, ...args: any[]) {
    const timestamp = new Date().toISOString();
    const formattedMessage = `[${timestamp}] [${this.name}] [${level.toUpperCase()}]: ${message}`;
    
    switch (level) {
      case 'info':
        console.info(formattedMessage, ...args);
        break;
      case 'warn':
        console.warn(formattedMessage, ...args);
        break;
      case 'error':
        console.error(formattedMessage, ...args);
        break;
      case 'debug':
        console.debug(formattedMessage, ...args);
        break;
    }
  }

  info(message: string, ...args: any[]) { this.log('info', message, ...args); }
  warn(message: string, ...args: any[]) { this.log('warn', message, ...args); }
  error(message: string, ...args: any[]) { this.log('error', message, ...args); }
  debug(message: string, ...args: any[]) { this.log('debug', message, ...args); }
}

export const getLogger = (name: string) => new Logger(name);
