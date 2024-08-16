
#import <Cocoa/Cocoa.h>

#include <iostream>

#include "app.hpp"
#include "handler.hpp"
#include "include/cef_application_mac.h"
#include "include/cef_command_line.h"
#include "include/wrapper/cef_library_loader.h"

// Receives notifications from the application.
@interface AgentsAppDelegate : NSObject <NSApplicationDelegate>

- (void)createApplication:(id)object;
- (void)tryToTerminateApplication:(NSApplication*)app;
@end

// Provide the CefAppProtocol implementation required by CEF.
@interface AgentsApplication : NSApplication <CefAppProtocol> {
 @private
  BOOL handlingSendEvent_;
}
@end

@implementation AgentsApplication
- (BOOL)isHandlingSendEvent {
  return handlingSendEvent_;
}

- (void)setHandlingSendEvent:(BOOL)handlingSendEvent {
  handlingSendEvent_ = handlingSendEvent;
}

- (void)sendEvent:(NSEvent*)event {
  CefScopedSendingEvent sendingEventScoper;
  [super sendEvent:event];
}

- (void)terminate:(id)sender {
  AgentsAppDelegate* delegate =
      static_cast<AgentsAppDelegate*>([NSApp delegate]);
  [delegate tryToTerminateApplication:self];
  // Return, don't exit. The application is responsible for exiting on its own.
}
@end

@implementation AgentsAppDelegate

// Create the application on the UI thread.
- (void)createApplication:(id)object {
  [[NSBundle mainBundle] loadNibNamed:@"MainMenu"
                                owner:NSApp
                      topLevelObjects:nil];

  // Set the delegate for application events.
  [[NSApplication sharedApplication] setDelegate:self];
}

- (void)tryToTerminateApplication:(NSApplication*)app {
}

- (NSApplicationTerminateReply)applicationShouldTerminate:
    (NSApplication*)sender {
  return NSTerminateNow;
}

// Called when the user clicks the app dock icon while the application is
// already running.
- (BOOL)applicationShouldHandleReopen:(NSApplication*)theApplication
                    hasVisibleWindows:(BOOL)flag {
  return NO;
}
@end

// Entry point function for the browser process.
int RunAgentApp(CefRefPtr<AgentApp> app) {
  CefMainArgs main_args(0, nullptr);

  @autoreleasepool {
    [AgentsApplication sharedApplication];

    // If there was an invocation to NSApp prior to this method, then the NSApp
    // will not be a AgentsApplication, but will instead be an NSApplication.
    // This is undesirable and we must enforce that this doesn't happen.
    CHECK([NSApp isKindOfClass:[AgentsApplication class]]);

    std::string framework_path =
        "/Users/theomonnom/livekit/agents/livekit-plugins/"
        "livekit-plugins-browser/cef/src/Debug/lkcef_app.app/Contents/"
        "Frameworks/Chromium Embedded Framework.framework";
    std::string main_bundle_path =
        "/Users/theomonnom/livekit/agents/livekit-plugins/"
        "livekit-plugins-browser/cef/src/Debug/lkcef_app.app";
    std::string subprocess_path =
        "/Users/theomonnom/livekit/agents/livekit-plugins/"
        "livekit-plugins-browser/cef/src/Debug/lkcef_app.app/Contents/"
        "Frameworks/lkcef Helper.app/Contents/MacOS/lkcef Helper";

    std::string framework_lib = framework_path + "/Chromium Embedded Framework";
    if (!cef_load_library(framework_lib.c_str())) {
      std::cerr << "lkcef: Failed to load CEF library" << std::endl;
      return 1;
    }

    CefSettings settings{};
    // settings.remote_debugging_port = 8088;
    CefString(&settings.framework_dir_path).FromString(framework_path);
    CefString(&settings.main_bundle_path).FromString(main_bundle_path);
    CefString(&settings.browser_subprocess_path).FromString(subprocess_path);

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

    // Create the application delegate.
    AgentsAppDelegate* delegate = [[AgentsAppDelegate alloc] init];
    // Set as the delegate for application events.
    NSApp.delegate = delegate;

    [delegate performSelectorOnMainThread:@selector(createApplication:)
                               withObject:nil
                            waitUntilDone:NO];

    app->Run();

    CefShutdown();
    cef_unload_library();

#if !__has_feature(objc_arc)
    [delegate release];
#endif  // !__has_feature(objc_arc)
    delegate = nil;
  }  // @autoreleasepool

  return 0;
}
