import { COASTLINE } from "./ne-coast.js";

const FIELD_INFO = {
  temperature: { label: "Temperature", unit: "°C" },
  wind_speed: { label: "Wind speed", unit: "m/s" },
  specific_humidity: { label: "Specific humidity", unit: "g/kg" },
};

const COLOR_STOPS = [
  [0.00, [39, 91, 121]],
  [0.25, [66, 139, 164]],
  [0.50, [145, 190, 190]],
  [0.74, [220, 207, 157]],
  [1.00, [213, 132, 85]],
];

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function formatValue(value) {
  if (!Number.isFinite(value)) return "—";
  return Number(value.toPrecision(4)).toString();
}

function colorAt(amount) {
  const value = Math.max(0, Math.min(1, amount));
  let index = 1;
  while (index < COLOR_STOPS.length - 1 && COLOR_STOPS[index][0] < value) index += 1;
  const [lowAt, low] = COLOR_STOPS[index - 1];
  const [highAt, high] = COLOR_STOPS[index];
  const mix = (value - lowAt) / (highAt - lowAt);
  return `rgb(${low.map((component, i) => Math.round(component + (high[i] - component) * mix)).join(",")})`;
}

function longitudeDelta(from, to) {
  const delta = (to - from) * Math.PI / 180;
  return Math.atan2(Math.sin(delta), Math.cos(delta)) * 180 / Math.PI;
}

function coordinateBounds(coordinates, minimum, maximum, longitude = false) {
  const values = coordinates.map(Number);
  const bounds = [];
  if (values.length === 1) return [[minimum, maximum]];
  const span = Math.abs(values[values.length - 1] - values[0]);
  const cyclic = longitude && span >= 350 && span < 359.5;
  for (let i = 0; i < values.length; i += 1) {
    const current = values[i];
    const previous = i > 0 ? values[i - 1] : cyclic ? values[values.length - 1] : null;
    const next = i < values.length - 1 ? values[i + 1] : cyclic ? values[0] : null;
    const previousStep = previous === null
      ? longitude ? -longitudeDelta(current, next) : current - next
      : longitude ? longitudeDelta(current, previous) : previous - current;
    const nextStep = next === null
      ? -previousStep
      : longitude ? longitudeDelta(current, next) : next - current;
    let low = current + previousStep / 2;
    let high = current + nextStep / 2;
    if (low > high) [low, high] = [high, low];
    bounds.push([Math.max(minimum, low), Math.min(maximum, high)]);
  }
  return bounds;
}

export function createGlobe(container) {
  const root = element("div", "globe-view");
  const canvas = element("canvas", "globe-canvas");
  canvas.setAttribute("role", "img");
  canvas.setAttribute("aria-label", "Interactive sampled atmospheric field globe");
  canvas.tabIndex = 0;
  const empty = element("div", "globe-empty", "No frame at the selected forecast time.");
  const controls = element("div", "globe-controls");
  const zoomOut = element("button", "globe-zoom", "−");
  zoomOut.type = "button";
  zoomOut.setAttribute("aria-label", "Zoom out");
  const zoomIn = element("button", "globe-zoom", "+");
  zoomIn.type = "button";
  zoomIn.setAttribute("aria-label", "Zoom in");
  const reset = element("button", "globe-reset", "Reset view");
  reset.type = "button";
  const zoomValue = element("span", "globe-zoom-value", "100%");
  zoomValue.setAttribute("aria-live", "polite");
  controls.append(zoomOut, zoomValue, zoomIn, reset);
  const hint = element("p", "globe-hint", "Drag to rotate · use +/− to zoom · arrow keys rotate");
  const status = element("p", "globe-status");
  status.setAttribute("aria-live", "polite");
  const legend = element("section", "globe-legend");
  legend.setAttribute("aria-label", "Field color legend");
  const legendTitle = element("strong", "globe-legend-title", "Field values");
  const legendUnit = element("span", "globe-legend-unit");
  const scale = element("div", "globe-legend-scale");
  scale.setAttribute("aria-hidden", "true");
  const labels = element("div", "globe-legend-labels");
  const minLabel = element("span", "globe-legend-min");
  const maxLabel = element("span", "globe-legend-max");
  labels.append(minLabel, maxLabel);
  const legendNote = element("span", "globe-legend-note", "Each tile shows one sampled grid value · no interpolation");
  legend.append(legendTitle, legendUnit, scale, labels, legendNote);
  root.append(canvas, empty, controls, hint, status, legend);
  container.replaceChildren(root);

  const context = canvas.getContext("2d");
  let frameData = null;
  let fieldId = "temperature";
  let range = null;
  let width = 0;
  let height = 0;
  let pixelRatio = 1;
  let centerLongitude = 0;
  let centerLatitude = 0;
  let zoom = 1;
  let drag = null;
  let destroyed = false;

  function updateZoomLabel() {
    zoomValue.textContent = `${Math.round(zoom * 100)}%`;
  }

  function draw() {
    if (!context || !width || !height || destroyed) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, width, height);
    const cx = width / 2;
    const cy = height / 2;
    const radius = Math.max(1, Math.min(width, height) * 0.39 * zoom);
    context.beginPath();
    context.arc(cx, cy, radius, 0, Math.PI * 2);
    context.fillStyle = "#0d2332";
    context.fill();
    context.save();
    context.beginPath();
    context.arc(cx, cy, radius, 0, Math.PI * 2);
    context.clip();

    const sinCenter = Math.sin(centerLatitude);
    const cosCenter = Math.cos(centerLatitude);
    const project = (longitude, latitude) => {
      const lat = latitude * Math.PI / 180;
      const delta = longitude * Math.PI / 180 - centerLongitude;
      const cosLat = Math.cos(lat);
      const cosDelta = Math.cos(delta);
      return [
        cosLat * Math.sin(delta),
        Math.sin(lat) * cosCenter - cosLat * cosDelta * sinCenter,
        Math.sin(lat) * sinCenter + cosLat * cosDelta * cosCenter,
      ];
    };
    const screenPoint = ([x, y]) => [cx + radius * x, cy - radius * y];
    const clipHemisphere = (polygon) => {
      const clipped = [];
      for (let i = 0; i < polygon.length; i += 1) {
        const current = polygon[i];
        const previous = polygon[(i + polygon.length - 1) % polygon.length];
        const currentVisible = current[2] >= 0;
        const previousVisible = previous[2] >= 0;
        if (currentVisible !== previousVisible) {
          const fraction = previous[2] / (previous[2] - current[2]);
          const intersection = previous.map((component, axis) => component + (current[axis] - component) * fraction);
          const length = Math.hypot(intersection[0], intersection[1], intersection[2]);
          clipped.push(intersection.map((component) => component / length));
        }
        if (currentVisible) clipped.push(current);
      }
      return clipped;
    };
    const drawLine = (points, color, lineWidth) => {
      context.strokeStyle = color;
      context.lineWidth = lineWidth;
      context.beginPath();
      for (let i = 1; i < points.length; i += 1) {
        const start = points[i - 1];
        const end = points[i];
        const startVisible = start[2] >= 0;
        const endVisible = end[2] >= 0;
        if (!startVisible && !endVisible) continue;
        let a = start;
        let b = end;
        if (startVisible !== endVisible) {
          const fraction = start[2] / (start[2] - end[2]);
          const intersection = start.map((component, axis) => component + (end[axis] - component) * fraction);
          const length = Math.hypot(intersection[0], intersection[1], intersection[2]);
          intersection.forEach((component, axis) => { intersection[axis] = component / length; });
          if (!startVisible) a = intersection;
          else b = intersection;
        }
        const [x1, y1] = screenPoint(a);
        const [x2, y2] = screenPoint(b);
        context.moveTo(x1, y1);
        context.lineTo(x2, y2);
      }
      context.stroke();
    };

    const validFrame = frameData && range && FIELD_INFO[fieldId];
    if (validFrame) {
      const { grid, frame } = frameData;
      const latitudes = grid?.latitudes;
      const longitudes = grid?.longitudes;
      const values = frame?.fields?.[fieldId];
      if (Array.isArray(latitudes) && Array.isArray(longitudes) && Array.isArray(values)
        && values.length === latitudes.length * longitudes.length) {
        const latitudeBounds = coordinateBounds(latitudes, -90, 90);
        const longitudeBounds = coordinateBounds(longitudes, -Infinity, Infinity, true);
        const span = range.max - range.min;
        for (let latIndex = 0; latIndex < latitudes.length; latIndex += 1) {
          const [south, north] = latitudeBounds[latIndex];
          if (!(north > south)) continue;
          for (let lonIndex = 0; lonIndex < longitudes.length; lonIndex += 1) {
            const value = Number(values[latIndex * longitudes.length + lonIndex]);
            const [west, east] = longitudeBounds[lonIndex];
            if (!Number.isFinite(value) || !(east > west)) continue;
            const corners = [
              [west, south], [east, south], [east, north], [west, north],
            ].map(([lon, lat]) => project(lon, lat));
            const visible = clipHemisphere(corners);
            if (visible.length < 3) continue;
            context.beginPath();
            const [x, y] = screenPoint(visible[0]);
            context.moveTo(x, y);
            for (let i = 1; i < visible.length; i += 1) {
              const [px, py] = screenPoint(visible[i]);
              context.lineTo(px, py);
            }
            context.closePath();
            context.fillStyle = colorAt(span > 0 ? (value - range.min) / span : 0.5);
            context.fill();
          }
        }
      }
    }

    context.lineWidth = Math.max(0.55, radius * 0.002);
    for (let latitude = -60; latitude <= 60; latitude += 30) {
      const points = [];
      for (let longitude = -180; longitude <= 180; longitude += 3) {
        points.push(project(longitude, latitude));
      }
      drawLine(points, "rgba(207, 225, 228, 0.19)", Math.max(0.5, radius * 0.0018));
    }
    for (let longitude = -180; longitude < 180; longitude += 30) {
      const points = [];
      for (let latitude = -90; latitude <= 90; latitude += 3) {
        points.push(project(longitude, latitude));
      }
      drawLine(points, "rgba(207, 225, 228, 0.19)", Math.max(0.5, radius * 0.0018));
    }

    context.lineCap = "round";
    context.lineJoin = "round";
    for (const line of COASTLINE) {
      const points = line.map(([longitude, latitude]) => project(longitude / 10, latitude / 10));
      drawLine(points, "rgba(244, 239, 218, 0.82)", Math.max(0.7, radius * 0.0026));
    }
    context.restore();
    context.beginPath();
    context.arc(cx, cy, radius, 0, Math.PI * 2);
    context.lineWidth = Math.max(1, radius * 0.004);
    context.strokeStyle = "rgba(205, 225, 235, 0.5)";
    context.stroke();
  }

  function resize() {
    if (destroyed) return;
    const bounds = root.getBoundingClientRect();
    width = Math.max(1, bounds.width);
    height = Math.max(1, bounds.height);
    pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * pixelRatio);
    canvas.height = Math.round(height * pixelRatio);
    draw();
  }

  function setFrame(data) {
    if (destroyed) return;
    frameData = data && data.grid && data.frame ? data : null;
    fieldId = data?.field || "temperature";
    range = data?.range && Number.isFinite(data.range.min) && Number.isFinite(data.range.max)
      ? data.range
      : null;
    const info = FIELD_INFO[fieldId];
    if (!frameData) {
      empty.hidden = false;
      status.textContent = "No frame at the selected forecast time.";
      canvas.setAttribute("aria-label", "No sampled frame is available at the selected forecast time.");
      legendTitle.textContent = "Field values";
      legendUnit.textContent = "";
      minLabel.textContent = "";
      maxLabel.textContent = "";
      draw();
      return;
    }
    empty.hidden = true;
    if (!info || !range) {
      status.textContent = "Frame unavailable: field metadata or shared color range is missing.";
      frameData = null;
      draw();
      return;
    }
    legendTitle.textContent = info.label;
    legendUnit.textContent = `(${info.unit} · 850 hPa)`;
    minLabel.textContent = `${formatValue(range.min)} ${info.unit}`;
    maxLabel.textContent = `${formatValue(range.max)} ${info.unit}`;
    const kind = data.frame.kind === "initial_reanalysis" ? "Initial reanalysis" : "Prediction";
    status.textContent = `${kind} · ${info.label} · 850 hPa (${info.unit}); sampled model-grid values.`;
    canvas.setAttribute("aria-label", `${kind}, ${info.label} in ${info.unit} at 850 hectopascals. Sampled model-grid values on a rotatable globe.`);
    draw();
  }

  function changeZoom(factor) {
    zoom = Math.max(0.72, Math.min(2.8, zoom * factor));
    updateZoomLabel();
    draw();
  }

  zoomIn.addEventListener("click", () => changeZoom(1.2));
  zoomOut.addEventListener("click", () => changeZoom(1 / 1.2));
  reset.addEventListener("click", () => {
    centerLongitude = 0;
    centerLatitude = 0;
    zoom = 1;
    updateZoomLabel();
    draw();
  });
  canvas.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    drag = { x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drag) return;
    const scale = Math.max(1, Math.min(width, height) * 0.39 * zoom);
    centerLongitude -= (event.clientX - drag.x) / scale;
    centerLatitude = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, centerLatitude + (event.clientY - drag.y) / scale));
    drag = { x: event.clientX, y: event.clientY };
    draw();
  });
  canvas.addEventListener("pointerup", () => { drag = null; });
  canvas.addEventListener("pointercancel", () => { drag = null; });
  canvas.addEventListener("keydown", (event) => {
    const rotation = 10 * Math.PI / 180;
    if (event.key === "ArrowLeft") centerLongitude -= rotation;
    else if (event.key === "ArrowRight") centerLongitude += rotation;
    else if (event.key === "ArrowUp") centerLatitude = Math.min(Math.PI / 2, centerLatitude + rotation);
    else if (event.key === "ArrowDown") centerLatitude = Math.max(-Math.PI / 2, centerLatitude - rotation);
    else return;
    event.preventDefault();
    draw();
  });

  const observer = typeof ResizeObserver === "function" ? new ResizeObserver(resize) : null;
  if (observer) observer.observe(root);
  else window.addEventListener("resize", resize);
  updateZoomLabel();
  resize();

  return {
    setFrame,
    resize,
    destroy() {
      if (destroyed) return;
      destroyed = true;
      observer?.disconnect();
      if (!observer) window.removeEventListener("resize", resize);
      container.replaceChildren();
    },
  };
}
