
#import <Cocoa/Cocoa.h>

#include <iostream>

#import <Cocoa/Cocoa.h>
#include <objc/runtime.h>

#include "app.hpp"
#include "handler.hpp"
#include "include/cef_application_mac.h"
#include "include/cef_command_line.h"
#include "include/wrapper/cef_library_loader.h"

BOOL g_handling_send_event = false;

@interface NSApplication (AgentsApplication) <CefAppProtocol>

- (BOOL)isHandlingSendEvent;
- (void)setHandlingSendEvent:(BOOL)handlingSendEvent;
- (void)_swizzled_sendEvent:(NSEvent*)event;
- (void)_swizzled_terminate:(id)sender;

@end

@implementation NSApplication (AgentsApplication)

// This selector is called very early during the application initialization.
+ (void)load {
  NSLog(@"AgentsApplication::load");
  // Swap NSApplication::sendEvent with _swizzled_sendEvent.
  Method original = class_getInstanceMethod(self, @selector(sendEvent));
  Method swizzled =
      class_getInstanceMethod(self, @selector(_swizzled_sendEvent));
  method_exchangeImplementations(original, swizzled);

  Method originalTerm = class_getInstanceMethod(self, @selector(terminate:));
  Method swizzledTerm =
      class_getInstanceMethod(self, @selector(_swizzled_terminate:));
  method_exchangeImplementations(originalTerm, swizzledTerm);
}

- (BOOL)isHandlingSendEvent {
  return g_handling_send_event;
}

- (void)setHandlingSendEvent:(BOOL)handlingSendEvent {
  g_handling_send_event = handlingSendEvent;
}

- (void)_swizzled_sendEvent:(NSEvent*)event {
  CefScopedSendingEvent sendingEventScoper;
  // Calls NSApplication::sendEvent due to the swizzling.
  [self _swizzled_sendEvent:event];
}

- (void)_swizzled_terminate:(id)sender {
  [self _swizzled_terminate:sender];
}

@end

// Entry point function for the browser process.
int RunAgentApp(CefRefPtr<AgentApp> app) {
  CefMainArgs main_args(0, nullptr);

  @autoreleasepool {
    [NSApplication sharedApplication];

    // If there was an invocation to NSApp prior to this method, then the NSApp
    // will not be a AgentsApplication, but will instead be an NSApplication.
    // This is undesirable and we must enforce that this doesn't happen.
    CHECK([NSApp isKindOfClass:[NSApplication class]]);

    std::string framework_lib = app->GetFrameworkPath() + "/Chromium Embedded Framework";
    if (!cef_load_library(framework_lib.c_str())) {
      std::cerr << "lkcef: Failed to load CEF library" << std::endl;
      return 1;
    }

    CefSettings settings{};
    settings.chrome_runtime = true;
    settings.external_message_pump = app->IsDevMode();
    settings.remote_debugging_port = app->GetRemoteDebuggingPort();
    CefString(&settings.root_cache_path).FromString(app->GetRootCachePath());
    CefString(&settings.framework_dir_path).FromString(app->GetFrameworkPath());
    CefString(&settings.main_bundle_path).FromString(app->GetMainBundlePath());
    CefString(&settings.browser_subprocess_path).FromString(app->GetSubprocessPath());

    settings.no_sandbox = true;  // No sandbox for MacOS, for livekit-agents,
                                 // we're only going to support Linux
    settings.windowless_rendering_enabled = true;

    // Initialize the CEF browser process. May return false if initialization
    // fails or if early exit is desired (for example, due to process singleton
    // relaunch behavior).
    if (!CefInitialize(main_args, settings, app.get(), nullptr)) {
      std::cerr << "lkcef: Failed to initialize CEF" << std::endl;
      // TODO(theomonnom): Use CefGetExitCode();
      return 1;
    }

    app->Run();
    CefShutdown();

    cef_unload_library();
  }  // @autoreleasepool

  return 0;
}
