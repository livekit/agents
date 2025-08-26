/*
 * @file ten_vad.h
 * @brief Ten Voice Activity Detection (ten_vad) C API
 * Version: 1.0.0
 *
 * Provides functions to create, process, and destroy a VAD instance.
 */
#ifndef TEN_VAD_H
#define TEN_VAD_H

#if defined(__APPLE__) || defined(__ANDROID__) || defined(__linux__)
#define TENVAD_API __attribute__((visibility("default")))
#elif defined(_WIN32) || defined(__CYGWIN__)
/**
 * @def TENVAD_API
 * @brief Export/import macro for ten_vad shared library symbols.
 */
#ifdef TENVAD_EXPORTS
#define TENVAD_API __declspec(dllexport)
#else
#define TENVAD_API __declspec(dllimport)
#endif
#else
#define TENVAD_API
#endif

#include <stddef.h> /* size_t */
#include <stdint.h> /* int16_t */

#ifdef __cplusplus
extern "C"
{
#endif

  /**
   * @typedef ten_vad_handle
   * @brief Opaque handle for ten_vad instance.
   */
  typedef void *ten_vad_handle_t;

  /**
   * @brief Create and initialize a ten_vad instance.
   *
   * @param[out] handle       Pointer to receive the vad handle.
   * @param[in]  hop_size     The number of samples between the start points of
   * two consecutive analysis frames. (e.g., 256).
   * @param[in]  threshold    VAD detection threshold ranging from [0.0, 1.0]
   * (default: 0.5).
   * @return 0 on success, or -1 error occurs.
   */
  TENVAD_API int ten_vad_create(ten_vad_handle_t *handle, size_t hop_size,
                                float threshold);

  /**
   * @brief Process one audio frame for voice activity detection.
   * Must call ten_vad_init() before calling this, and ten_vad_destroy() when done.
   *
   * @param[in]  handle           Valid VAD handle returned by ten_vad_create().
   * @param[in]  audio_data       Pointer to an array of int16_t samples,
   * buffer length must equal the hop size specified at ten_vad_create.
   * @param[in]  audio_data_length  size of audio_data buffer, here should be equal to hop_size.
   * @param[out] out_probability  Pointer to a float (size 1) that receives the
   * voice activity probability in the range [0.0, 1.0].
   * @param[out] out_flag         Pointer to an int (size 1) that receives the
   * detection result: 0 = no voice, 1 = voice detected.
   * @return 0 on success, or -1 error occurs.
   */
  TENVAD_API int ten_vad_process(ten_vad_handle_t handle, const int16_t *audio_data, size_t audio_data_length,
                                 float *out_probability, int *out_flag);

  /**
   * @brief Destroy a ten_vad instance and release its resources.
   *
   * @param[in,out] handle Pointer to the ten_vad handle; set to NULL on return.
   * @return 0 on success, or -1 error occurs.
   */
  TENVAD_API int ten_vad_destroy(ten_vad_handle_t *handle);

  /**
   * @brief Get the ten_vad library version string.
   *
   * @return The version string (e.g., "1.0.0").
   */
  TENVAD_API const char *ten_vad_get_version(void);

#ifdef __cplusplus
}
#endif

#endif /* TEN_VAD_H */