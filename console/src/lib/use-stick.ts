"use client";

import { useCallback, useEffect, useRef, useState, type DependencyList } from "react";

/** How far from the foot still counts as "at the foot", in px. */
const SLACK = 48;

/**
 * A scroller that sticks to its foot: while the person is at the bottom,
 * new content keeps the bottom in view; once they scroll up to read, the
 * scroller stays where they left it and `following` goes false until
 * `follow()` or their own scroll brings it back. The reply streaming in
 * grows the content many times a second, so the stick is measured on each
 * change of `deps`. `attach` is the scroller's ref callback; `overflowing`
 * says whether the content is taller than the scroller.
 */
export function useStick(deps: DependencyList) {
  const element = useRef<HTMLDivElement | null>(null);
  const observer = useRef<ResizeObserver | null>(null);
  const [following, setFollowing] = useState(true);
  const [overflowing, setOverflowing] = useState(false);
  const stuck = useRef(true);

  const measure = useCallback(() => {
    const scroller = element.current;
    if (!scroller) return;
    setOverflowing(scroller.scrollHeight > scroller.clientHeight + 1);
  }, []);

  const attach = useCallback(
    (node: HTMLDivElement | null) => {
      observer.current?.disconnect();
      observer.current = null;
      element.current = node;
      if (node && typeof ResizeObserver !== "undefined") {
        observer.current = new ResizeObserver(measure);
        observer.current.observe(node);
        if (node.firstElementChild) observer.current.observe(node.firstElementChild);
      }
    },
    [measure],
  );

  const onScroll = useCallback(() => {
    const scroller = element.current;
    if (!scroller) return;
    const atFoot = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight <= SLACK;
    stuck.current = atFoot;
    setFollowing(atFoot);
  }, []);

  const follow = useCallback(() => {
    const scroller = element.current;
    if (!scroller) return;
    stuck.current = true;
    setFollowing(true);
    scroller.scrollTop = scroller.scrollHeight;
  }, []);

  useEffect(() => {
    const scroller = element.current;
    if (scroller && stuck.current) scroller.scrollTop = scroller.scrollHeight;
    /* The content, not the hook's own values, is what moves the foot. */
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return { attach, following, overflowing, follow, onScroll };
}
