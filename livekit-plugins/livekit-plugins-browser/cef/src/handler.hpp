#ifndef LKCEF_HANDLER_HPP
#define LKCEF_HANDLER_HPP

#include "include/cef_client.h"

#include <list>

class AgentHandler : public CefClient,
                     public CefDisplayHandler,
                     public CefRenderHandler,
                     public CefAudioHandler,
                     public CefLifeSpanHandler,
                     public CefLoadHandler {

public:
  AgentHandler();
  ~AgentHandler() override;

  static AgentHandler *GetInstance();

  CefRefPtr<CefDisplayHandler> GetDisplayHandler() override { return this; }
  CefRefPtr<CefAudioHandler> GetAudioHandler() override { return this; }
  CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override { return this; }
  CefRefPtr<CefLoadHandler> GetLoadHandler() override { return this; }

  // CefDisplayHandler methods
  void OnTitleChange(CefRefPtr<CefBrowser> browser,
                     const CefString &title) override;

  // CefRenderHandler methods
  void OnPaint(CefRefPtr<CefBrowser> browser, PaintElementType type,
               const RectList &dirtyRects, const void *buffer, int width,
               int height) override;

  void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect &rect) override;

  // CefAudioHandler methods
  void OnAudioStreamPacket(CefRefPtr<CefBrowser> browser, const float **data,
                           int frames, int64_t pts) override;

  void OnAudioStreamStarted(CefRefPtr<CefBrowser> browser,
                            const CefAudioParameters &params,
                            int channels) override;

  void OnAudioStreamStopped(CefRefPtr<CefBrowser> browser) override;

  void OnAudioStreamError(CefRefPtr<CefBrowser> browser,
                          const CefString &message) override;

  // CefLifeSpanHandler methods
  void OnAfterCreated(CefRefPtr<CefBrowser> browser) override;
  bool DoClose(CefRefPtr<CefBrowser> browser) override;
  void OnBeforeClose(CefRefPtr<CefBrowser> browser) override;

  // CefLoadHandler methods
  void OnLoadError(CefRefPtr<CefBrowser> browser, CefRefPtr<CefFrame> frame,
                   ErrorCode errorCode, const CefString &errorText,
                   const CefString &failedUrl) override;

  void ShowMainWindow();
  void CloseAllBrowsers(bool force_close);
  bool IsClosing() const { return is_closing_; }

  static bool IsChromeRuntimeEnabled();

private:
  // Platform-specific implementation.
  void PlatformTitleChange(CefRefPtr<CefBrowser> browser,
                           const CefString &title);
  void PlatformShowWindow(CefRefPtr<CefBrowser> browser);

  // List of existing browser windows. Only accessed on the CEF UI thread.
  typedef std::list<CefRefPtr<CefBrowser>> BrowserList;
  BrowserList browser_list_;

  bool is_closing_ = false;

  // Include the default reference counting implementation.
  IMPLEMENT_REFCOUNTING(AgentHandler);
};

#endif // LKCEF_HANDLER_HPP
