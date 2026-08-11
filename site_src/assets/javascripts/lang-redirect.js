// Browser-language auto-redirect (ADR-0003).
//
// Static site => no server-side negotiation. On the first page load of a
// session we read navigator.language: a Chinese browser is sent to /zh/, every
// other browser stays on the English default. We flip a sessionStorage flag
// BEFORE redirecting so:
//   * the redirect fires at most once per session (debounce, no loop), and
//   * a later manual switch via the language selector is respected.
(function () {
  try {
    var KEY = "ilp-lang-choice";
    if (sessionStorage.getItem(KEY)) return; // already decided this session

    var path = window.location.pathname;
    var inZh = path === "/zh" || path.indexOf("/zh/") === 0;
    var lang = (navigator.language || navigator.userLanguage || "").toLowerCase();
    var prefersZh = lang.indexOf("zh") === 0;

    sessionStorage.setItem(KEY, "1"); // decide exactly once per session

    if (prefersZh && !inZh) {
      window.location.replace(
        "/zh" + path + window.location.search + window.location.hash
      );
    }
  } catch (e) {
    /* sessionStorage blocked (private mode etc.) — just stay put */
  }
})();
