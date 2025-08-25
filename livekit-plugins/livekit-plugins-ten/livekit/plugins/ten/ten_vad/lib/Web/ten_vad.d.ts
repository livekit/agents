/**
 * This file is part of TEN Framework, an open source project.
 * Licensed under the Apache License, Version 2.0.
 * See the LICENSE file for more information.
 * 
 * TEN VAD (Voice Activity Detection) WebAssembly Module
 * TypeScript type definitions
 */

export interface TenVADModule {
  /**
   * Create and initialize a VAD instance
   * @param handlePtr Pointer to store the VAD handle
   * @param hopSize Number of samples between consecutive analysis frames (e.g., 256)
   * @param threshold VAD detection threshold [0.0, 1.0]
   * @returns 0 on success, -1 on error
   */
  _ten_vad_create(handlePtr: number, hopSize: number, threshold: number): number;

  /**
   * Process audio frame for voice activity detection
   * @param handle Valid VAD handle from ten_vad_create
   * @param audioDataPtr Pointer to int16 audio samples array
   * @param audioDataLength Length of audio data (should equal hopSize)
   * @param outProbabilityPtr Pointer to output probability [0.0, 1.0]
   * @param outFlagPtr Pointer to output flag (0: no voice, 1: voice detected)
   * @returns 0 on success, -1 on error
   */
  _ten_vad_process(
    handle: number,
    audioDataPtr: number,
    audioDataLength: number,
    outProbabilityPtr: number,
    outFlagPtr: number
  ): number;

  /**
   * Destroy VAD instance and release resources
   * @param handlePtr Pointer to the VAD handle
   * @returns 0 on success, -1 on error
   */
  _ten_vad_destroy(handlePtr: number): number;

  /**
   * Get library version string
   * @returns Version string pointer
   */
  _ten_vad_get_version(): number;

  // WebAssembly Memory Management
  _malloc(size: number): number;
  _free(ptr: number): void;

  // Memory access helpers
  HEAP16: Int16Array;
  HEAPF32: Float32Array;
  HEAP32: Int32Array;
  HEAPU8: Uint8Array;

  // Value access methods
  getValue(ptr: number, type: 'i8' | 'i16' | 'i32' | 'float' | 'double'): number;
  setValue(ptr: number, value: number, type: 'i8' | 'i16' | 'i32' | 'float' | 'double'): void;

  // String utilities
  UTF8ToString(ptr: number): string;
  lengthBytesUTF8(str: string): number;
  stringToUTF8(str: string, outPtr: number, maxBytesToWrite: number): void;
}

/**
 * High-level TypeScript wrapper for TEN VAD
 */
export class TenVAD {
  private module: TenVADModule;
  private handle: number | null;
  private hopSize: number;

  constructor(module: TenVADModule, hopSize: number, threshold: number);

  /**
   * Process audio samples for voice activity detection
   * @param audioData Int16Array of audio samples (length must equal hopSize)
   * @returns Object with probability and voice detection flag
   */
  process(audioData: Int16Array): {
    probability: number;
    isVoice: boolean;
  } | null;

  /**
   * Get library version
   */
  getVersion(): string;

  /**
   * Destroy VAD instance
   */
  destroy(): void;

  /**
   * Check if VAD instance is valid
   */
  isValid(): boolean;
}

/**
 * Create TEN VAD WebAssembly module
 */
declare function createVADModule(): Promise<TenVADModule>;

export default createVADModule; 