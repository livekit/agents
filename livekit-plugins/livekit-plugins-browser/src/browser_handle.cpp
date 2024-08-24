#include "browser_handle.hpp"

void BrowserHandle::SetSize(int width, int height) {
  width_ = width;
  height_ = height;

  if (browser_)
    browser_->GetHost()->WasResized();
}


void BrowserHandle::Close() {
  if (browser_)
    browser_->GetHost()->CloseBrowser(true);
}
