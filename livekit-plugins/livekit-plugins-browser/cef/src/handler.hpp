#ifndef LKCEF_HANDLER_HPP
#define LKCEF_HANDLER_HPP

#include <list>

#include "dev_renderer.hpp"
#include "browser_handle.hpp"
#include "include/cef_client.h"
#include "include/wrapper/cef_helpers.h"

class DevToolsHandler : public CefClient {
 public:
  DevToolsHandler();
  ~DevToolsHandler();

  static DevToolsHandler* GetInstance();

 private:
  IMPLEMENT_REFCOUNTING(DevToolsHandler);
};

class AgentHandler : public CefClient,
                     public CefDisplayHandler,
                     public CefRenderHandler,
                     public CefAudioHandler,
                     public CefLifeSpanHandler,
                     public CefLoadHandler {
 public:
  AgentHandler(CefRefPtr<BrowserStore> browser_store, CefRefPtr<DevRenderer> dev_renderer);
  ~AgentHandler();

  static AgentHandler* GetInstance();

  CefRefPtr<CefDisplayHandler> GetDisplayHandler() override { return this; }
  CefRefPtr<CefRenderHandler> GetRenderHandler() override { return this; }
  CefRefPtr<CefAudioHandler> GetAudioHandler() override { return this; }
  CefRefPtr<CefLifeSpanHandler> GetLifeSpanHandler() override { return this; }
  CefRefPtr<CefLoadHandler> GetLoadHandler() override { return this; }

  // CefDisplayHandler methods
  void OnTitleChange(CefRefPtr<CefBrowser> browser,
                     const CefString& title) override;

  // CefRenderHandler methods
  void OnPaint(CefRefPtr<CefBrowser> browser,
               PaintElementType type,
               const RectList& dirtyRects,
               const void* buffer,
               int width,
               int height) override;

  void GetViewRect(CefRefPtr<CefBrowser> browser, CefRect& rect) override;

  // CefAudioHandler methods
  void OnAudioStreamPacket(CefRefPtr<CefBrowser> browser,
                           const float** data,
                           int frames,
                           int64_t pts) override;

  void OnAudioStreamStarted(CefRefPtr<CefBrowser> browser,
                            const CefAudioParameters& params,
                            int channels) override;

  void OnAudioStreamStopped(CefRefPtr<CefBrowser> browser) override;

  void OnAudioStreamError(CefRefPtr<CefBrowser> browser,
                          const CefString& message) override;

  // CefLifeSpanHandler methods

  bool OnBeforePopup(CefRefPtr<CefBrowser> browser,
                     CefRefPtr<CefFrame> frame,
                     const CefString& target_url,
                     const CefString& target_frame_name,
                     WindowOpenDisposition target_disposition,
                     bool user_gesture,
                     const CefPopupFeatures& popupFeatures,
                     CefWindowInfo& windowInfo,
                     CefRefPtr<CefClient>& client,
                     CefBrowserSettings& settings,
                     CefRefPtr<CefDictionaryValue>& extra_info,
                     bool* no_javascript_access) override;

  void OnAfterCreated(CefRefPtr<CefBrowser> browser) override;
  bool DoClose(CefRefPtr<CefBrowser> browser) override;
  void OnBeforeClose(CefRefPtr<CefBrowser> browser) override;

  // CefLoadHandler methods

  void OnLoadingStateChange(CefRefPtr<CefBrowser> browser,
                            bool isLoading,
                            bool canGoBack,
                            bool canGoForward) override;

  void CloseAllBrowsers(bool force_close);

 private:
  CefRefPtr<BrowserStore> browser_store_;
  CefRefPtr<DevRenderer> dev_renderer_;

  IMPLEMENT_REFCOUNTING(AgentHandler);
};

#endif  // LKCEF_HANDLER_HPP
