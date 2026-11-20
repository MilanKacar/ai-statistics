// =====================================================================
// StatsLab: shared canvas-plotting + math utilities for embedded,
// in-chapter interactive modules (keeps each chapter's inline <script>
// short -- it just calls these instead of redefining them every time).
// =====================================================================
window.StatsLab = (function () {
  function cssVar(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  var colors = {
    accent: cssVar("--accent") || "#4c78a8", accent2: cssVar("--accent2") || "#f58518",
    accent3: cssVar("--accent3") || "#54a24b", accent4: cssVar("--accent4") || "#e45756",
    border: cssVar("--border") || "#ddd", ink: cssVar("--ink") || "#222", inkSoft: cssVar("--ink-soft") || "#666"
  };

  function makePlot(canvas, margin) {
    margin = margin || { l: 50, r: 20, t: 16, b: 34 };
    var ctx = canvas.getContext("2d");
    var W = canvas.width, H = canvas.height;
    return {
      ctx: ctx, W: W, H: H, margin: margin, xDomain: [0, 1], yDomain: [0, 1],
      setDomain: function (xd, yd) { this.xDomain = xd; this.yDomain = yd; },
      px: function (x) { var m = this.margin; return m.l + (x - this.xDomain[0]) / (this.xDomain[1] - this.xDomain[0]) * (this.W - m.l - m.r); },
      py: function (y) { var m = this.margin; return this.H - m.b - (y - this.yDomain[0]) / (this.yDomain[1] - this.yDomain[0]) * (this.H - m.t - m.b); },
      clear: function () { this.ctx.clearRect(0, 0, this.W, this.H); },
      axes: function (xLabel, yLabel) {
        var ctx = this.ctx, m = this.margin, ink = cssVar("--ink-soft"), border = cssVar("--border");
        ctx.strokeStyle = border; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(m.l, this.py(this.yDomain[0])); ctx.lineTo(this.W - m.r, this.py(this.yDomain[0])); ctx.stroke();
        ctx.beginPath(); ctx.moveTo(m.l, m.t); ctx.lineTo(m.l, this.H - m.b); ctx.stroke();
        ctx.fillStyle = ink; ctx.font = "11px -apple-system,sans-serif";
        for (var i = 0; i <= 4; i++) {
          var xv = this.xDomain[0] + i / 4 * (this.xDomain[1] - this.xDomain[0]);
          var px = this.px(xv);
          ctx.beginPath(); ctx.moveTo(px, this.py(this.yDomain[0])); ctx.lineTo(px, this.py(this.yDomain[0]) + 4); ctx.stroke();
          ctx.textAlign = "center"; ctx.fillText(xv.toFixed(Math.abs(xv) < 10 ? 1 : 0), px, this.py(this.yDomain[0]) + 16);
        }
        for (var j = 0; j <= 3; j++) {
          var yv = this.yDomain[0] + j / 3 * (this.yDomain[1] - this.yDomain[0]);
          var py = this.py(yv);
          ctx.beginPath(); ctx.moveTo(m.l - 4, py); ctx.lineTo(m.l, py); ctx.stroke();
          ctx.textAlign = "right"; ctx.fillText((Math.abs(yv) < 1 && yv !== 0) ? yv.toFixed(2) : yv.toFixed(1), m.l - 7, py + 3);
        }
        if (xLabel) { ctx.textAlign = "center"; ctx.fillText(xLabel, (this.W + m.l - m.r) / 2, this.H - 4); }
        if (yLabel) { ctx.save(); ctx.translate(12, (this.H - m.b + m.t) / 2); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center"; ctx.fillText(yLabel, 0, 0); ctx.restore(); }
      },
      line: function (points, color, width) {
        var ctx = this.ctx; ctx.strokeStyle = color; ctx.lineWidth = width || 2; ctx.beginPath();
        var self = this;
        points.forEach(function (p, i) { var x = self.px(p[0]), y = self.py(p[1]); i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y); });
        ctx.stroke();
      },
      fillArea: function (points, color) {
        var ctx = this.ctx; if (points.length < 2) return;
        var self = this;
        ctx.fillStyle = color; ctx.beginPath();
        ctx.moveTo(this.px(points[0][0]), this.py(this.yDomain[0]));
        points.forEach(function (p) { ctx.lineTo(self.px(p[0]), self.py(p[1])); });
        ctx.lineTo(this.px(points[points.length - 1][0]), this.py(this.yDomain[0]));
        ctx.closePath(); ctx.fill();
      },
      bars: function (points, color, widthFrac) {
        var ctx = this.ctx; widthFrac = widthFrac || 0.8;
        if (points.length < 1) return;
        var dx = points.length > 1 ? Math.abs(this.px(points[1][0]) - this.px(points[0][0])) * widthFrac : 40;
        ctx.fillStyle = color;
        var self = this;
        points.forEach(function (p) {
          var x = self.px(p[0]), y0 = self.py(0), y1 = self.py(p[1]);
          ctx.fillRect(x - dx / 2, Math.min(y0, y1), dx, Math.abs(y1 - y0));
        });
      },
      scatter: function (points, color, r) {
        var ctx = this.ctx; ctx.fillStyle = color;
        var self = this;
        points.forEach(function (p) { ctx.beginPath(); ctx.arc(self.px(p[0]), self.py(p[1]), r || 3, 0, 2 * Math.PI); ctx.fill(); });
      },
      vline: function (x, color, dashed) {
        var ctx = this.ctx; ctx.strokeStyle = color; ctx.lineWidth = 2;
        ctx.setLineDash(dashed ? [5, 4] : []);
        ctx.beginPath(); ctx.moveTo(this.px(x), this.py(this.yDomain[0])); ctx.lineTo(this.px(x), this.py(this.yDomain[1])); ctx.stroke();
        ctx.setLineDash([]);
      },
      hline: function (y, color, dashed, width) {
        var ctx = this.ctx; ctx.strokeStyle = color; ctx.lineWidth = width || 2;
        ctx.setLineDash(dashed ? [5, 4] : []);
        ctx.beginPath(); ctx.moveTo(this.px(this.xDomain[0]), this.py(y)); ctx.lineTo(this.px(this.xDomain[1]), this.py(y)); ctx.stroke();
        ctx.setLineDash([]);
      },
      arrow: function (x0, y0, x1, y1, color, width) {
        var ctx = this.ctx;
        var px0 = this.px(x0), py0 = this.py(y0), px1 = this.px(x1), py1 = this.py(y1);
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = width || 3;
        ctx.beginPath(); ctx.moveTo(px0, py0); ctx.lineTo(px1, py1); ctx.stroke();
        var angle = Math.atan2(py1 - py0, px1 - px0);
        ctx.beginPath();
        ctx.moveTo(px1, py1);
        ctx.lineTo(px1 - 10 * Math.cos(angle - 0.4), py1 - 10 * Math.sin(angle - 0.4));
        ctx.lineTo(px1 - 10 * Math.cos(angle + 0.4), py1 - 10 * Math.sin(angle + 0.4));
        ctx.closePath(); ctx.fill();
      }
    };
  }

  function erf(x) {
    var sign = x < 0 ? -1 : 1; x = Math.abs(x);
    var a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741, a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
    var t = 1 / (1 + p * x);
    var y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-x * x);
    return sign * y;
  }
  function normalCDF(x, mu, sigma) { return 0.5 * (1 + erf((x - mu) / (sigma * Math.sqrt(2)))); }
  function normalPDF(x, mu, sigma) { return Math.exp(-0.5 * Math.pow((x - mu) / sigma, 2)) / (sigma * Math.sqrt(2 * Math.PI)); }
  function randNormal(mu, sigma) {
    var u = Math.random() || 1e-9, v = Math.random();
    var z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    return mu + sigma * z;
  }
  function mean(arr) { return arr.reduce(function (a, b) { return a + b; }, 0) / arr.length; }
  function variance(arr, ddof) {
    ddof = ddof || 0;
    var m = mean(arr);
    return arr.reduce(function (a, b) { return a + (b - m) * (b - m); }, 0) / (arr.length - ddof);
  }
  function std(arr, ddof) { return Math.sqrt(variance(arr, ddof)); }
  function median(arr) {
    var s = arr.slice().sort(function (a, b) { return a - b; });
    var n = s.length;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  function percentile(arr, p) {
    var s = arr.slice().sort(function (a, b) { return a - b; });
    var idx = (p / 100) * (s.length - 1);
    var lo = Math.floor(idx), hi = Math.ceil(idx);
    return s[lo] + (s[hi] - s[lo]) * (idx - lo);
  }
  function logFactorial(n) { var s = 0; for (var i = 2; i <= n; i++) s += Math.log(i); return s; }
  function binomPMF(k, n, p) {
    if (k < 0 || k > n) return 0;
    var logC = logFactorial(n) - logFactorial(k) - logFactorial(n - k);
    return Math.exp(logC + k * Math.log(p || 1e-12) + (n - k) * Math.log(1 - p || 1e-12));
  }
  function poissonPMF(k, lambda) {
    if (k < 0) return 0;
    return Math.exp(k * Math.log(lambda) - lambda - logFactorial(k));
  }
  function histogram(data, nbins, range) {
    var lo = range ? range[0] : Math.min.apply(null, data);
    var hi = range ? range[1] : Math.max.apply(null, data);
    var width = (hi - lo) / nbins || 1;
    var counts = new Array(nbins).fill(0);
    data.forEach(function (v) {
      var idx = Math.floor((v - lo) / width);
      if (idx >= nbins) idx = nbins - 1;
      if (idx < 0) idx = 0;
      counts[idx]++;
    });
    return counts.map(function (c, i) { return [lo + width * (i + 0.5), c, width]; });
  }
  function solveLinearSystem(Ain, bin) {
    var n = bin.length;
    var A = Ain.map(function (r) { return r.slice(); });
    var b = bin.slice();
    for (var col = 0; col < n; col++) {
      var piv = col;
      for (var r = col + 1; r < n; r++) if (Math.abs(A[r][col]) > Math.abs(A[piv][col])) piv = r;
      var tmpA = A[col]; A[col] = A[piv]; A[piv] = tmpA;
      var tmpB = b[col]; b[col] = b[piv]; b[piv] = tmpB;
      if (Math.abs(A[col][col]) < 1e-12) continue;
      for (var rr = 0; rr < n; rr++) {
        if (rr === col) continue;
        var factor = A[rr][col] / A[col][col];
        for (var c = col; c < n; c++) A[rr][c] -= factor * A[col][c];
        b[rr] -= factor * b[col];
      }
    }
    return b.map(function (v, i) { return Math.abs(A[i][i]) < 1e-12 ? 0 : v / A[i][i]; });
  }
  function polyFit(xs, ys, degree) {
    var n = degree + 1;
    var XtX = Array.from({ length: n }, function () { return new Array(n).fill(0); });
    var Xty = new Array(n).fill(0);
    for (var i = 0; i < xs.length; i++) {
      var row = Array.from({ length: n }, function (_, j) { return Math.pow(xs[i], j); });
      for (var a = 0; a < n; a++) {
        Xty[a] += row[a] * ys[i];
        for (var b = 0; b < n; b++) XtX[a][b] += row[a] * row[b];
      }
    }
    return solveLinearSystem(XtX, Xty);
  }
  function polyEval(coeffs, x) {
    var s = 0, p = 1;
    for (var j = 0; j < coeffs.length; j++) { s += coeffs[j] * p; p *= x; }
    return s;
  }

  function tCriticalApprox(conf, df) {
    // Approximate t critical value via inverse-Normal (Acklam's approximation)
    // plus a small-sample correction. Good enough for a teaching visualization.
    var alpha = 1 - conf;
    var p = 1 - alpha / 2;
    var a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
              1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00];
    var b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
               6.680131188771972e+01, -1.328068155288572e+01];
    var c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
              -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00];
    var d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00];
    var plow = 0.02425, phigh = 1 - plow, q, r, zForConf;
    if (p < plow) {
      q = Math.sqrt(-2 * Math.log(p));
      zForConf = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    } else if (p <= phigh) {
      q = p - 0.5; r = q * q;
      zForConf = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
                 (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
    } else {
      q = Math.sqrt(-2 * Math.log(1 - p));
      zForConf = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
                 ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
    }
    if (df > 100) return zForConf;
    var inflate = 1 + (zForConf * zForConf + 1) / (4 * df);
    return zForConf * inflate;
  }

  return {
    colors: colors, makePlot: makePlot, erf: erf, normalCDF: normalCDF, normalPDF: normalPDF,
    randNormal: randNormal, mean: mean, variance: variance, std: std, median: median,
    percentile: percentile, logFactorial: logFactorial, binomPMF: binomPMF, poissonPMF: poissonPMF,
    histogram: histogram, solveLinearSystem: solveLinearSystem, polyFit: polyFit, polyEval: polyEval,
    tCriticalApprox: tCriticalApprox
  };
})();

(function () {
  "use strict";
  var CH = document.body.getAttribute("data-chapter") || "unknown";
  var STORAGE_PREFIX = "sa_notes::";
  var key = STORAGE_PREFIX + CH;

  function loadNotes() {
    try { return JSON.parse(localStorage.getItem(key) || "[]"); }
    catch (e) { return []; }
  }
  function saveNotes(arr) { localStorage.setItem(key, JSON.stringify(arr)); }
  function allChapterKeys() {
    return Object.keys(localStorage).filter(function (k) { return k.indexOf(STORAGE_PREFIX) === 0; });
  }
  function escapeHtml(s) {
    var d = document.createElement("div");
    d.textContent = s;
    return d.innerHTML;
  }
  function colorLabel(c) {
    return { yellow: "🟡 interesting", green: "🟢 understood",
             red: "🔴 confusing", blue: "🔵 review later" }[c] || c;
  }

  // ===================================================================
  // Image rail (desktop two-pane layout)
  // ===================================================================
  function setupRail() {
    var rail = document.getElementById("railContent");
    if (!rail) return;
    var figs = document.querySelectorAll(".article img.has-rail");
    if (figs.length === 0) {
      rail.innerHTML = '<p class="railEmpty">No figures in this chapter.</p>';
      return;
    }
    rail.innerHTML = "";
    figs.forEach(function (img) {
      var figId = img.getAttribute("data-figid");
      var captionP = img.nextElementSibling && img.nextElementSibling.classList.contains("img-caption")
        ? img.nextElementSibling : null;
      var wrap = document.createElement("div");
      wrap.className = "railFigure";
      wrap.id = "rail-" + figId;
      var cloneImg = img.cloneNode(true);
      cloneImg.classList.remove("has-rail");
      wrap.appendChild(cloneImg);
      if (captionP) {
        var cap = document.createElement("div");
        cap.className = "cap";
        cap.textContent = captionP.textContent;
        wrap.appendChild(cap);
      }
      rail.appendChild(wrap);

      // Click the rail thumbnail -> jump to the real, inline figure in the text.
      wrap.style.cursor = "pointer";
      wrap.addEventListener("click", function () {
        img.scrollIntoView({ behavior: "smooth", block: "center" });
        highlightRail(figId);
      });
      // Click the inline figure itself -> highlight its entry in the rail.
      img.style.cursor = "pointer";
      img.addEventListener("click", function () { highlightRail(figId); });
    });

    if ("IntersectionObserver" in window) {
      var obs = new IntersectionObserver(function (entries) {
        entries.forEach(function (e) {
          if (e.isIntersecting) highlightRail(e.target.getAttribute("data-figid"));
        });
      }, { rootMargin: "-40% 0px -50% 0px" });
      figs.forEach(function (img) { obs.observe(img); });
    }
  }
  function highlightRail(figId) {
    document.querySelectorAll(".railFigure").forEach(function (f) { f.classList.remove("active"); });
    var el = document.getElementById("rail-" + figId);
    if (el) el.classList.add("active");
  }

  // ===================================================================
  // Sidebar: mobile toggle + quick filter
  // ===================================================================
  function setupSidebar() {
    var sidebar = document.getElementById("sidebar");
    var overlay = document.getElementById("sidebarOverlay");
    var toggle = document.getElementById("btnSidebarToggle");
    if (toggle && sidebar) {
      toggle.addEventListener("click", function () {
        sidebar.classList.add("open");
        if (overlay) overlay.classList.add("open");
      });
    }
    if (overlay && sidebar) {
      overlay.addEventListener("click", function () {
        sidebar.classList.remove("open");
        overlay.classList.remove("open");
      });
    }
    var filter = document.getElementById("sidebarFilter");
    if (filter && sidebar) {
      filter.addEventListener("input", function () {
        var q = filter.value.trim().toLowerCase();
        sidebar.querySelectorAll("li[data-title]").forEach(function (li) {
          var match = li.getAttribute("data-title").indexOf(q) !== -1;
          li.classList.toggle("hidden", q.length > 0 && !match);
        });
        sidebar.querySelectorAll("details").forEach(function (d) {
          var anyVisible = d.querySelectorAll("li:not(.hidden)").length > 0;
          d.style.display = (q.length > 0 && !anyVisible) ? "none" : "";
          if (q.length > 0) d.setAttribute("open", "open");
        });
      });
    }
    var current = sidebar && sidebar.querySelector("li.current");
    if (current) current.scrollIntoView({ block: "center" });
  }

  // ===================================================================
  // Highlighting + notes
  // ===================================================================
  var selToolbar, notePopup, pendingSelectionRange = null;

  function initToolbar() {
    selToolbar = document.getElementById("selToolbar");
    notePopup = document.getElementById("notePopup");
    if (!selToolbar) return;

    document.addEventListener("mouseup", onSelectionChange);
    document.addEventListener("touchend", function () { setTimeout(onSelectionChange, 60); });

    selToolbar.querySelectorAll("button[data-color]").forEach(function (btn) {
      btn.addEventListener("click", function () { applyHighlight(btn.getAttribute("data-color")); });
    });
    var noteBtn = document.getElementById("btnAddNote");
    if (noteBtn) {
      noteBtn.addEventListener("click", function () {
        var markEl = applyHighlight("yellow");
        if (markEl) openNotePopupFor(markEl);
      });
    }

    document.addEventListener("mousedown", function (e) {
      if (selToolbar && !selToolbar.contains(e.target)) selToolbar.style.display = "none";
    });
  }

  function onSelectionChange() {
    if (!selToolbar) return;
    if (notePopup && notePopup.style.display === "block") return;
    var sel = window.getSelection();
    if (!sel || sel.isCollapsed || sel.rangeCount === 0) { selToolbar.style.display = "none"; return; }
    var range = sel.getRangeAt(0);
    var article = document.getElementById("article");
    if (!article || !article.contains(range.commonAncestorContainer)) { selToolbar.style.display = "none"; return; }
    if (range.toString().trim().length === 0) { selToolbar.style.display = "none"; return; }
    pendingSelectionRange = range.cloneRange();
    var rect = range.getBoundingClientRect();
    selToolbar.style.display = "flex";
    selToolbar.style.top = (window.scrollY + rect.top - 46) + "px";
    selToolbar.style.left = Math.max(8, window.scrollX + rect.left) + "px";
  }

  function findBlock(node) {
    while (node && node !== document.body) {
      if (node.nodeType === 1 && node.hasAttribute && node.hasAttribute("data-bid")) return node;
      node = node.parentNode;
    }
    return null;
  }

  function applyHighlight(color) {
    if (!pendingSelectionRange) return null;
    var range = pendingSelectionRange;
    if (selToolbar) selToolbar.style.display = "none";
    var block = findBlock(range.startContainer) || findBlock(range.commonAncestorContainer);
    if (!block) { window.getSelection().removeAllRanges(); return null; }

    var pre = document.createRange();
    pre.selectNodeContents(block);
    pre.setEnd(range.startContainer, range.startOffset);
    var startOffset = pre.toString().length;
    var selectedText = range.toString();
    var endOffset = startOffset + selectedText.length;

    var id = "hl-" + Date.now() + "-" + Math.floor(Math.random() * 10000);
    var mark = document.createElement("mark");
    mark.className = "hl hl-" + color;
    mark.id = id;
    try { range.surroundContents(mark); }
    catch (e) {
      var frag = range.extractContents();
      mark.appendChild(frag);
      range.insertNode(mark);
    }
    window.getSelection().removeAllRanges();

    var notes = loadNotes();
    notes.push({
      id: id, blockId: block.getAttribute("data-bid"), startOffset: startOffset, endOffset: endOffset,
      text: selectedText.slice(0, 200), color: color, note: "", ts: Date.now()
    });
    saveNotes(notes);

    mark.addEventListener("click", function (ev) { ev.stopPropagation(); openNotePopupFor(mark); });
    pendingSelectionRange = null;
    return mark;
  }

  function openNotePopupFor(mark) {
    if (!mark || !notePopup) return;
    var notes = loadNotes();
    var rec = null;
    for (var i = 0; i < notes.length; i++) if (notes[i].id === mark.id) { rec = notes[i]; break; }
    if (!rec) return;
    var rect = mark.getBoundingClientRect();
    notePopup.style.display = "block";
    notePopup.style.top = (window.scrollY + rect.bottom + 8) + "px";
    var left = window.scrollX + rect.left;
    var maxLeft = window.scrollX + document.documentElement.clientWidth - 280;
    notePopup.style.left = Math.max(8, Math.min(left, maxLeft)) + "px";
    var ta = notePopup.querySelector("textarea");
    ta.value = rec.note || "";

    notePopup.querySelector("#btnSaveNote").onclick = function () {
      rec.note = ta.value;
      saveNotes(notes);
      refreshNoteIcon(mark, rec);
      notePopup.style.display = "none";
    };
    notePopup.querySelector("#btnDeleteHl").onclick = function () {
      var remaining = notes.filter(function (n) { return n.id !== mark.id; });
      saveNotes(remaining);
      unwrapMark(mark);
      notePopup.style.display = "none";
    };
    notePopup.querySelector("#btnCloseNote").onclick = function () { notePopup.style.display = "none"; };
    ta.focus();
  }

  function unwrapMark(mark) {
    var parent = mark.parentNode;
    if (!parent) return;
    var icon = mark.nextSibling;
    while (mark.firstChild) parent.insertBefore(mark.firstChild, mark);
    parent.removeChild(mark);
    if (icon && icon.classList && icon.classList.contains("noteIcon")) icon.remove();
    parent.normalize();
  }

  function refreshNoteIcon(mark, rec) {
    var icon = mark.nextSibling;
    if (icon && icon.nodeType === 1 && icon.classList.contains("noteIcon")) icon.remove();
    if (rec.note && rec.note.trim().length) {
      var span = document.createElement("span");
      span.className = "noteIcon";
      span.textContent = "📝";
      span.title = rec.note;
      span.addEventListener("click", function (ev) { ev.stopPropagation(); openNotePopupFor(mark); });
      mark.parentNode.insertBefore(span, mark.nextSibling);
    }
  }

  function rangeFromOffsets(block, start, end) {
    var node, charCount = 0, range = document.createRange();
    var foundStart = false, foundEnd = false;
    var walker = document.createTreeWalker(block, NodeFilter.SHOW_TEXT, null);
    while ((node = walker.nextNode())) {
      var nextCount = charCount + node.length;
      if (!foundStart && start >= charCount && start <= nextCount) {
        range.setStart(node, start - charCount);
        foundStart = true;
      }
      if (!foundEnd && end >= charCount && end <= nextCount) {
        range.setEnd(node, end - charCount);
        foundEnd = true;
      }
      if (foundStart && foundEnd) break;
      charCount = nextCount;
    }
    if (!foundStart || !foundEnd) return null;
    return range;
  }

  function restoreHighlights() {
    var notes = loadNotes();
    var byBlock = {};
    notes.forEach(function (n) {
      (byBlock[n.blockId] = byBlock[n.blockId] || []).push(n);
    });
    Object.keys(byBlock).forEach(function (bid) {
      var block = document.querySelector('[data-bid="' + bid + '"]');
      if (!block) return;
      var recs = byBlock[bid].slice().sort(function (a, b) { return b.startOffset - a.startOffset; });
      recs.forEach(function (rec) {
        var range = rangeFromOffsets(block, rec.startOffset, rec.endOffset);
        if (!range) return;
        var mark = document.createElement("mark");
        mark.className = "hl hl-" + rec.color;
        mark.id = rec.id;
        try { range.surroundContents(mark); }
        catch (e) {
          var frag = range.extractContents();
          mark.appendChild(frag);
          range.insertNode(mark);
        }
        mark.addEventListener("click", function (ev) { ev.stopPropagation(); openNotePopupFor(mark); });
        if (rec.note && rec.note.trim().length) {
          var span = document.createElement("span");
          span.className = "noteIcon";
          span.textContent = "📝";
          span.title = rec.note;
          span.addEventListener("click", function (ev) { ev.stopPropagation(); openNotePopupFor(mark); });
          mark.parentNode.insertBefore(span, mark.nextSibling);
        }
      });
    });
  }

  // ===================================================================
  // Notes panel + export/import
  // ===================================================================
  function setupNotesPanel() {
    var btn = document.getElementById("btnNotesPanel");
    var panel = document.getElementById("notesPanel");
    var overlay = document.getElementById("notesOverlay");
    var closeBtn = document.getElementById("btnCloseNotesPanel");
    if (!btn || !panel) return;

    function render() {
      var list = document.getElementById("notesList");
      list.innerHTML = "";
      var notes = loadNotes().slice().sort(function (a, b) { return b.ts - a.ts; });
      if (notes.length === 0) {
        list.innerHTML = '<p class="railEmpty">No highlights or notes in this chapter yet. Select some text to start.</p>';
        return;
      }
      notes.forEach(function (n) {
        var card = document.createElement("div");
        card.className = "noteCard";
        card.innerHTML =
          '<div class="snippet">“' + escapeHtml(n.text) + '”</div>' +
          (n.note ? '<div class="noteText">' + escapeHtml(n.note) + "</div>" : "") +
          '<div class="meta"><span>' + colorLabel(n.color) + '</span><span><a data-act="jump">Jump to &rarr;</a> &nbsp; <a data-act="del">Remove</a></span></div>';
        card.querySelector('[data-act="jump"]').addEventListener("click", function () {
          var el = document.getElementById(n.id);
          if (el) { el.scrollIntoView({ behavior: "smooth", block: "center" }); }
          panel.classList.remove("open"); overlay.classList.remove("open");
        });
        card.querySelector('[data-act="del"]').addEventListener("click", function () {
          var remaining = loadNotes().filter(function (x) { return x.id !== n.id; });
          saveNotes(remaining);
          var el = document.getElementById(n.id);
          if (el) unwrapMark(el);
          render();
        });
        list.appendChild(card);
      });
    }

    btn.addEventListener("click", function () { render(); panel.classList.add("open"); overlay.classList.add("open"); });
    if (closeBtn) closeBtn.addEventListener("click", function () { panel.classList.remove("open"); overlay.classList.remove("open"); });
    if (overlay) overlay.addEventListener("click", function () { panel.classList.remove("open"); overlay.classList.remove("open"); });

    var exportBtn = document.getElementById("btnExportNotes");
    if (exportBtn) exportBtn.addEventListener("click", exportAll);
    var importBtn = document.getElementById("btnImportNotes");
    var importFile = document.getElementById("importFile");
    if (importBtn && importFile) {
      importBtn.addEventListener("click", function () { importFile.click(); });
      importFile.addEventListener("change", importAll);
    }
  }

  function exportAll() {
    var data = {};
    allChapterKeys().forEach(function (k) { data[k] = JSON.parse(localStorage.getItem(k)); });
    var blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    var url = URL.createObjectURL(blob);
    var a = document.createElement("a");
    a.href = url; a.download = "statistics-book-notes.json";
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  }
  function importAll(e) {
    var file = e.target.files[0];
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      try {
        var data = JSON.parse(reader.result);
        Object.keys(data).forEach(function (k) {
          if (k.indexOf(STORAGE_PREFIX) === 0) localStorage.setItem(k, JSON.stringify(data[k]));
        });
        alert("Notes imported. Reloading page...");
        location.reload();
      } catch (err) { alert("Could not read that file as valid notes JSON."); }
    };
    reader.readAsText(file);
    e.target.value = "";
  }

  document.addEventListener("DOMContentLoaded", function () {
    setupSidebar();
    setupRail();
    initToolbar();
    restoreHighlights();
    setupNotesPanel();
  });
})();
