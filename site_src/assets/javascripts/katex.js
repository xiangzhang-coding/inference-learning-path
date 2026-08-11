// Mount KaTeX on every page. Material emits `document$` on initial load and on
// every navigation, so math re-renders correctly (we deliberately disabled
// navigation.instant — see ADR-0003). arithmatex `generic: true` wraps math in
// \( \) / \[ \] delimiters, which auto-render picks up.
//
// Guard: extra_javascript loads this before the KaTeX CDN bundles, so on the
// very first paint `renderMathInElement` may not exist yet — retry until it does.
function ilpRenderMath(root) {
  if (typeof renderMathInElement === "undefined") {
    setTimeout(() => ilpRenderMath(root), 50);
    return;
  }
  renderMathInElement(root, {
    delimiters: [
      { left: "$$", right: "$$", display: true },
      { left: "$", right: "$", display: false },
      { left: "\\(", right: "\\)", display: false },
      { left: "\\[", right: "\\]", display: true },
    ],
    throwOnError: false,
    strict: "ignore",
  });
}

document$.subscribe(({ body }) => {
  ilpRenderMath(body);
});
