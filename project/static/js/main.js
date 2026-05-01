/* ── Global utility functions used across all pages ─────────── */

/**
 * Format a number as US currency: $1,234,567
 */
function money(val) {
  if (val === null || val === undefined || isNaN(val)) return "$--";
  return "$" + Number(val).toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0
  });
}

/**
 * Format a large integer with commas: 10,291
 */
function fmt(val) {
  if (val === null || val === undefined) return "--";
  return Number(val).toLocaleString("en-US");
}

/**
 * Format a percentage: 12.4%
 */
function pct(val, decimals = 1) {
  if (val === null || val === undefined || isNaN(val)) return "--%";
  return Number(val).toFixed(decimals) + "%";
}

/**
 * Format a decimal ratio (0–1) as percentage: 0.124 → "12.4%"
 */
function ratio(val, decimals = 1) {
  return pct(val * 100, decimals);
}

/**
 * Set active nav link based on current path
 */
document.addEventListener("DOMContentLoaded", () => {
  const path = window.location.pathname;
  document.querySelectorAll(".nav-link").forEach(link => {
    const href = link.getAttribute("href");
    if (href === path || (path !== "/" && href !== "/" && path.startsWith(href))) {
      link.classList.add("active");
    } else {
      link.classList.remove("active");
    }
  });
});

/**
 * Global Chart.js default settings
 */
if (typeof Chart !== "undefined") {
  Chart.defaults.font.family  = "'Segoe UI', system-ui, sans-serif";
  Chart.defaults.font.size    = 12;
  Chart.defaults.color        = "#6B7280";
  Chart.defaults.plugins.tooltip.backgroundColor = "#1A1A2E";
  Chart.defaults.plugins.tooltip.titleColor      = "#fff";
  Chart.defaults.plugins.tooltip.bodyColor       = "#d1d5db";
  Chart.defaults.plugins.tooltip.padding         = 10;
  Chart.defaults.plugins.tooltip.cornerRadius    = 8;
  Chart.defaults.plugins.tooltip.callbacks = {
    ...Chart.defaults.plugins.tooltip.callbacks,
    label: function (context) {
      let label = context.dataset.label || "";
      if (label) label += ": ";
      const val = context.parsed.y ?? context.parsed.x ?? context.raw;
      if (typeof val === "number" && val > 1000) {
        label += "$" + val.toLocaleString("en-US", { maximumFractionDigits: 0 });
      } else {
        label += val;
      }
      return label;
    }
  };
}