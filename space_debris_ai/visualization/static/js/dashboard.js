// A.R.I.E.S Dashboard JavaScript
// AetherOS-style interactive visualizations

const COLORS = {
    bg: '#050508',
    panel: 'rgba(12, 14, 24, 0.85)',
    cyan: '#800000',
    blue: '#a52a2a',
    orange: '#ff8c42',
    green: '#22c55e',
    purple: '#8b4513',
    textPrimary: '#e2e8f0',
    textSecondary: '#64748b',
    grid: 'rgba(100, 116, 139, 0.15)',
};

const EARTH_RADIUS = 6371.0; // km

// Экваториальная орбита: круг в плоскости XY (z=0), бесконечное движение корабля по экватору
const EQUATORIAL_ORBIT_RADIUS_KM = 7000; // км от центра Земли (~630 км над поверхностью)
const EQUATORIAL_ORBIT_POINTS = 80;       // точек на круг для плавной траектории
function getEquatorialOrbitPositions(numPoints) {
    numPoints = numPoints || EQUATORIAL_ORBIT_POINTS;
    var R = EQUATORIAL_ORBIT_RADIUS_KM;
    var out = [];
    for (var i = 0; i < numPoints; i++) {
        var t = (i / numPoints) * 2 * Math.PI;
        out.push([R * Math.cos(t), R * Math.sin(t), 0]);
    }
    return out;
}

// Common layout settings for Plotly
const commonLayout = {
    paper_bgcolor: COLORS.panel,
    plot_bgcolor: COLORS.panel,
    font: {
        family: 'Courier New, monospace',
        color: COLORS.textPrimary,
        size: 11,
    },
    margin: { t: 30, r: 30, b: 50, l: 60 },
    showlegend: true,
    legend: {
        bgcolor: COLORS.panel,
        bordercolor: COLORS.grid,
        borderwidth: 1,
        font: { size: 10 },
    },
};

// Last loaded mission data (for resize redraw without refetch)
let lastMissionData = null;

// Radar animation: sweep angle (rad), drift offset (px), animation id
let radarSweepAngle = 0;
let radarAnimationId = null;
let radarDriftOffset = 0;
const RADAR_DRIFT_SPEED = 0.12;

// Orbit chart: animate ship along trajectory (trace index for "Корабль")
let orbitShipAnimationId = null;
let orbitChartUserDragging = false; // true while mouse down on plot → skip restyle so rotation is smooth
const ORBIT_SHIP_LOOP_MS = 120000; // один полный круг по экватору за 2 минуты
let debrisRemovedFromMap = {};  // object_id -> true: утилизация — скрыть с карты
let debrisAvoided = {};        // object_id -> true: обход — рисовать сзади
// Мусор «к утилизации»: исчезнет со сферы спустя время после выбора утилизации (object_id -> { addedAt: timestamp })
var debrisPendingRemoval = {};
var DEBRIS_REMOVAL_DELAY_MS = 6500; // утилизация срабатывает через ~6.5 сек после выбора, пока шаттл движется по орбите
var dismissedAnomalyKeys = {}; // ключи устранённых аномалий (одно «Устранить» — одна аномалия)
var noAnomalyPanelDismissed = false; // закрыла зелёное окно — больше не показывать
var dangerPanelShowTimeoutId = null;  // задержка перед показом красного окна (не так быстро)
var DANGER_PANEL_DELAY_MS = 6000;    // 6 сек до показа панели опасности
/** Периодический resync панели опасности при удерживаемом магните/вибрации (мс) */
var _lastArduinoDangerResyncMs = 0;
/** Расстояние до выбранного мусора из симуляции (км), пока попап открыт */
var _orbitPopupSimDistanceKm = null;
var _popupArduinoIntervalId = null;
/** Попап «мусор» открыт — обновлять ARDUINO LIVE по SSE/fetch без проверки style.display */
var _orbitPopupArduinoOpen = false;
var _popupArduinoLogPollTick = 0;
const ORBIT_SHIP_FPS = 5; // update ship at 5fps when not dragging; paused while user rotates
// Отложенный рендер орбиты: пока пользователь держит мышь на сфере — не обновляем; после отпускания — один раз отрисуем
var orbitChartDeferredRender = false;

/** Симуляция SOC: 100% → 90% со скоростью 1 процент в минуту (затем удержание на 90%). */
var _simBatteryStartMs = Date.now();
var _simBatteryIntervalId = null;

function getSimulatedBatteryPercent() {
    var minutes = (Date.now() - _simBatteryStartMs) / 60000;
    var pct = 100 - minutes;
    if (pct < 90) pct = 90;
    if (pct > 100) pct = 100;
    return pct;
}

function formatBatteryDisplayPct(p) {
    var s = p.toFixed(1);
    return s.endsWith('.0') ? Math.round(p) + '%' : s + '%';
}

function applySimulatedBatteryUI() {
    var pct = getSimulatedBatteryPercent();
    var w = Math.min(100, Math.max(0, pct));
    var batFill = document.getElementById('battery-fill');
    var batPct = document.getElementById('battery-pct');
    var vtSoc = document.getElementById('vt-soc');
    if (batFill) batFill.style.width = w + '%';
    if (batPct) batPct.textContent = formatBatteryDisplayPct(pct);
    if (vtSoc) vtSoc.textContent = pct.toFixed(1) + '%';
}

function scheduleOrbitChartRender(data) {
    if (orbitChartUserDragging) {
        orbitChartDeferredRender = true;
        return;
    }
    renderOrbitChart(data);
}

// Удалить конкретный объект мусора с 3D-сферы без полной перерисовки
function removeDebrisFromOrbitPlot(objectId) {
    var orbitChartEl = document.getElementById('orbit-chart');
    if (!orbitChartEl || !orbitChartEl.data) return;
    var oidStr = String(objectId);
    var data = orbitChartEl.data;
    for (var ti = 0; ti < data.length; ti++) {
        var trace = data[ti];
        if (!trace || !trace.customdata || !trace.x || !trace.y || !trace.z) continue;
        var cdArr = trace.customdata;
        var newX = [], newY = [], newZ = [], newCd = [], newText = [];
        var newSizes = null;
        var hasSizes = trace.marker && Array.isArray(trace.marker.size);
        if (hasSizes) newSizes = [];
        var changed = false;
        for (var pi = 0; pi < cdArr.length; pi++) {
            var cd = cdArr[pi];
            var cdOid = (cd && cd.length >= 8) ? cd[7] : null;
            if (cdOid != null && String(cdOid) === oidStr) {
                changed = true; // пропускаем эту точку — она «исчезает»
                continue;
            }
            newX.push(trace.x[pi]);
            newY.push(trace.y[pi]);
            newZ.push(trace.z[pi]);
            newCd.push(cdArr[pi]);
            if (trace.text && Array.isArray(trace.text)) newText.push(trace.text[pi]);
            if (hasSizes) newSizes.push(trace.marker.size[pi]);
        }
        if (changed) {
            var update = { x: [newX], y: [newY], z: [newZ], customdata: [newCd] };
            if (newText.length) update.text = [newText];
            if (hasSizes) update['marker.size'] = [newSizes];
            try { Plotly.restyle(orbitChartEl, update, [ti]); } catch (e) {}
        }
    }
}

document.addEventListener('mouseup', function() {
    orbitChartUserDragging = false;
    if (orbitChartDeferredRender && lastMissionData) {
        orbitChartDeferredRender = false;
        renderOrbitChart(lastMissionData);
    }
    startOrbitShipAnimation();
});
document.addEventListener('mouseleave', function() {
    orbitChartUserDragging = false;
    if (orbitChartDeferredRender && lastMissionData) {
        orbitChartDeferredRender = false;
        renderOrbitChart(lastMissionData);
    }
    startOrbitShipAnimation();
});

function startOrbitShipAnimation() {
    var orbitChartEl = document.getElementById('orbit-chart');
    if (!orbitChartEl || !orbitChartEl.data || orbitChartEl.data.length < 4) return;
    if (orbitShipAnimationId != null) return;
    var lastShipRestyleTime = 0;
    var restyleInterval = ORBIT_SHIP_FPS > 0 ? 1000 / ORBIT_SHIP_FPS : Infinity;
    // Траектория — экватор (круг в XY), бесконечный цикл
    var basePositions = getEquatorialOrbitPositions();
    var fullPath = basePositions.concat(basePositions.slice(1));
    var pathLen = fullPath.length;
    function runOrbitShipAnimation() {
        if (!orbitChartEl || !orbitChartEl.data || orbitChartEl.data.length < 4) {
            orbitShipAnimationId = null;
            return;
        }
        var duration = ORBIT_SHIP_LOOP_MS;
        var t = (Date.now() % duration) / duration;
        var idx = t * (pathLen - 1);
        var i0 = Math.min(Math.floor(idx), pathLen - 2);
        var i1 = i0 + 1;
        var frac = idx - Math.floor(idx);
        var p0 = fullPath[i0], p1 = fullPath[i1];
        var x = p0[0] + (p1[0] - p0[0]) * frac;
        var y = p0[1] + (p1[1] - p0[1]) * frac;
        var z = p0[2] + (p1[2] - p0[2]) * frac;
        var now = Date.now();
        // Утилизация: убрать мусор со сферы через небольшую задержку после выбора (пока шаттл движется)
        for (var oid in debrisPendingRemoval) {
            if (!debrisPendingRemoval.hasOwnProperty(oid)) continue;
            var info = debrisPendingRemoval[oid];
            if (!info || !info.addedAt) continue;
            if (now - info.addedAt >= DEBRIS_REMOVAL_DELAY_MS) {
                removeDebrisFromOrbitPlot(oid);
                delete debrisPendingRemoval[oid];
            }
        }
        if (!orbitChartUserDragging && restyleInterval < Infinity && (now - lastShipRestyleTime >= restyleInterval)) {
            lastShipRestyleTime = now;
            try {
                Plotly.restyle(orbitChartEl, { x: [[x]], y: [[y]], z: [[z]] }, [3]);
            } catch (e) {}
        }
        orbitShipAnimationId = requestAnimationFrame(runOrbitShipAnimation);
    }
    orbitShipAnimationId = requestAnimationFrame(runOrbitShipAnimation);
}

function getDebrisTypeLabel(type) {
    var map = { fragment: 'Fragment', rocket_body: 'Rocket body', satellite: 'Satellite fragment', tool: 'Tool', panel: 'Panel (solar/MLI)' };
    return (map[type] != null ? map[type] : type) || 'Unknown';
}
function getAnomalyReasonLabel(type) {
    var map = {
        TELEMETRY_ANOMALY: 'Telemetry anomaly',
        SENSOR_DEVIATION: 'Sensor deviation',
        COMM_LOSS: 'Communication loss',
        ORIENTATION_ANOMALY: 'Orientation anomaly',
        GYRO_DRIFT: 'Gyro drift',
        POWER_FLUCTUATION: 'Power fluctuation',
        MAGNETIC_FIELD: 'Magnetic field detected (Hall / magnetic sensor, Arduino)',
        VIBRATION_SPIKE: 'Hull vibration (Arduino)',
        TEMP_ANOMALY: 'Temperature out of safe range (Arduino)',
        PROXIMITY_ALERT: 'Object too close to spacecraft',
        PROXIMITY_THERMAL_ANOMALY: 'Ultrasonic range < 100 km and temperature > 30 °C (Arduino)',
    };
    return (map[type] != null ? map[type] : type) || 'Sensor anomaly';
}
function formatTrajectory(velocity) {
    if (!velocity || !Array.isArray(velocity) || velocity.length < 3) return '—';
    return velocity[0].toFixed(4) + ', ' + velocity[1].toFixed(4) + ', ' + velocity[2].toFixed(4) + ' km/s';
}

/** Generate a seed in [0, 2e9] from time (ms) and random component. */
const MAX_SEED = 2000000000;
function generateSeed() {
    const timePart = Date.now() % (MAX_SEED + 1);
    const randomPart = Math.floor(Math.random() * 1000);
    return (timePart + randomPart) % (MAX_SEED + 1);
}

function getAlternatingStreamSeed() {
    return generateSeed();
}

const STREAM_REFRESH_INTERVAL_MS = 45000;
var streamRefreshIntervalId = null;

function startStreamRefresh() {
    if (streamRefreshIntervalId != null) return;
    streamRefreshIntervalId = setInterval(function() {
        var seed = getAlternatingStreamSeed();
        loadMissionData(seed).catch(function() {});
    }, STREAM_REFRESH_INTERVAL_MS);
}

/** Fetch mission data with the given seed and render the dashboard. */
async function loadMissionData(seed) {
    // #region agent log
    debugLog937bb1('dashboard.js:loadMissionData:entry', 'loadMissionData called', { seed: seed }, 'start-btn-issue', 'H3');
    // #endregion
    const url = '/api/mission-data' + (seed != null ? '?seed=' + encodeURIComponent(seed) : '');
    const response = await fetch(url);
    // #region agent log
    debugLog937bb1('dashboard.js:loadMissionData:response', 'loadMissionData fetch response', { ok: response.ok, status: response.status, url: url }, 'start-btn-issue', 'H4');
    // #endregion
    if (!response.ok) throw new Error('Mission data request failed');
    const data = await response.json();
    lastMissionData = data;

    // Орбиту перерисовываем только при первом показе — сфера не обновляется по таймеру; корабль двигается по lastMissionData в анимации
    var orbitChartEl = document.getElementById('orbit-chart');
    var orbitChartNeedsInitialRender = !orbitChartEl || !orbitChartEl.data || orbitChartEl.data.length === 0;
    if (orbitChartNeedsInitialRender) scheduleOrbitChartRender(data);
    else {
        var obstaclesCountEl = document.getElementById('orbit-obstacles-count');
        if (obstaclesCountEl && data.debris) {
            var list = data.debris.filter(function(d) { return !debrisRemovedFromMap[String(d.object_id)]; });
            obstaclesCountEl.textContent = list.length;
        }
    }

    updateVelocityGauge(data);
    updateReadouts(data);
    updateTrajectoryFooter(data);
    drawRadar(data, radarSweepAngle);
    if (radarAnimationId != null) cancelAnimationFrame(radarAnimationId);
    radarAnimationId = requestAnimationFrame(runRadarAnimation);
    updateHealthBars(data);
    updateCopilotCards(data);
    updateThreatCount(data);
    updateBatteryShields(data);
    updateDangerStatus(data);
    await fetchArduinoLive();
    updateDangerWarnings(data);

    if (data.virtual_sensors && typeof data.virtual_sensors === 'object') {
        var speedsM = data.speeds || [];
        var speedKmhM = speedsM.length ? speedsM[speedsM.length - 1] * 3600 : null;
        var altM = (data.spacecraft_status || {}).altitude;
        _virtualSensorsLive = Object.assign({}, data.virtual_sensors, {
            _meta: { seq: null, sim_time: null, seed: data.seed, from_mission: true },
            _motion: speedKmhM != null ? { speed_kmh: speedKmhM, altitude_km: altM != null ? altM : null } : undefined,
        });
        updateVirtualSensorsUI(_virtualSensorsLive);
    }

    if (data.seed != null) {
        const seedEl = document.getElementById('seed-value');
        const blockEl = document.getElementById('current-seed');
        if (seedEl) seedEl.textContent = data.seed;
        if (blockEl) blockEl.style.display = 'inline';
    }
    if (_orbitPopupArduinoOpen) updatePopupWithArduino();
    return data;
}

function startLiveTime() {
    const el = document.getElementById('live-time');
    if (!el) return;
    function tick() {
        const d = new Date();
        el.textContent = d.toTimeString().slice(0, 8);
    }
    tick();
    setInterval(tick, 1000);
}

/** Orbital speed (km/h) + altitude (km) for the Velocity gauge and ALT readout. */
function applyVelocityPanel(speedKmh, altitudeKm) {
    if (speedKmh == null || typeof speedKmh !== 'number' || isNaN(speedKmh)) return;
    var gaugeValue = document.getElementById('gauge-value');
    var gaugeArc = document.getElementById('gauge-arc');
    if (gaugeValue) gaugeValue.textContent = speedKmh.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
    if (gaugeArc) {
        var maxKmh = 40000;
        var ratio = Math.min(1, Math.max(0, speedKmh / maxKmh));
        gaugeArc.style.strokeDashoffset = String(245 * (1 - ratio));
    }
    var altEl = document.getElementById('readout-alt');
    if (altEl && altitudeKm != null && typeof altitudeKm === 'number' && !isNaN(altitudeKm)) {
        altEl.textContent = altitudeKm.toFixed(1) + ' km';
    }
}

function updateVelocityGauge(data) {
    const speeds = data.speeds || [];
    const speedKmh = speeds.length ? speeds[speeds.length - 1] * 3600 : 0;
    const s = data.spacecraft_status || {};
    applyVelocityPanel(speedKmh, s.altitude != null ? s.altitude : null);
}

function updateReadouts(data) {
    const s = data.spacecraft_status || {};
    const altEl = document.getElementById('readout-alt');
    if (altEl && s.altitude != null) altEl.textContent = s.altitude.toFixed(1) + ' km';
}

function updateTrajectoryFooter(data) {
    const times = data.times || [];
    const positions = data.positions || [];
    const etaEl = document.getElementById('eta');
    const distEl = document.getElementById('dist');
    const headingEl = document.getElementById('heading');
    if (etaEl && times.length) etaEl.textContent = (times[times.length - 1] / 60).toFixed(2) + ' min ETA';
    if (distEl && positions.length >= 2) {
        const p0 = positions[0], p1 = positions[positions.length - 1];
        const d = Math.sqrt((p1[0]-p0[0])**2 + (p1[1]-p0[1])**2 + (p1[2]-p0[2])**2);
        distEl.textContent = d.toFixed(0) + ' km DIST';
    }
    if (headingEl) headingEl.textContent = 'NNE 024° HEADING';
}

function drawRadar(data, sweepAngle) {
    const canvas = document.getElementById('radar-canvas');
    const debrisPos = data.debris_positions || [];
    const positions = data.positions || [];
    if (!canvas || !canvas.getContext) return;

    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    const w = canvas.width = rect.width || canvas.offsetWidth || 218;
    const h = canvas.height = rect.height || canvas.offsetHeight || 218;
    const cx = w / 2, cy = h / 2, r = Math.min(w, h) / 2 - 8;

    ctx.clearRect(0, 0, w, h);
    ctx.save();
    ctx.beginPath();
    ctx.arc(cx, cy, r + 2, 0, Math.PI * 2);
    ctx.clip();

    // Concentric circles
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.25)';
    ctx.lineWidth = 1;
    for (let i = 1; i <= 4; i++) {
        ctx.beginPath();
        ctx.arc(cx, cy, r * i / 4, 0, Math.PI * 2);
        ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.5)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    // Sweep trail (PPI-style fading cone)
    if (sweepAngle != null) {
        const sweepWidth = (30 * Math.PI) / 180;
        const startAngle = sweepAngle - sweepWidth;
        const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        grad.addColorStop(0, 'rgba(0, 212, 255, 0.08)');
        grad.addColorStop(0.7, 'rgba(0, 212, 255, 0.03)');
        grad.addColorStop(1, 'rgba(0, 212, 255, 0)');
        ctx.fillStyle = grad;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.arc(cx, cy, r + 2, startAngle, sweepAngle);
        ctx.closePath();
        ctx.fill();
    }

    // Crosshairs
    ctx.strokeStyle = 'rgba(0, 212, 255, 0.2)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx, cy + r);
    ctx.moveTo(cx - r, cy);
    ctx.lineTo(cx + r, cy);
    ctx.stroke();

    // Sweep line
    if (sweepAngle != null) {
        const sx = cx + Math.sin(sweepAngle) * r;
        const sy = cy - Math.cos(sweepAngle) * r;
        ctx.strokeStyle = 'rgba(0, 212, 255, 0.9)';
        ctx.lineWidth = 2;
        ctx.shadowColor = COLORS.cyan;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(sx, sy);
        ctx.stroke();
        ctx.shadowBlur = 0;
    }

    const scPos = positions.length ? positions[positions.length - 1] : [0, 0, 0];

    // Spacecraft at center
    ctx.fillStyle = COLORS.cyan;
    ctx.strokeStyle = COLORS.cyan;
    ctx.shadowColor = COLORS.cyan;
    ctx.shadowBlur = 8;
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(cx, cy, 4, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    const safeDebris = [];
    const dangerousDebris = [];
    debrisPos.forEach(p => {
        const dx = p[0] - scPos[0], dy = p[1] - scPos[1], dz = (p[2] || 0) - (scPos[2] || 0);
        const dist = Math.sqrt(dx*dx + dy*dy + dz*dz) || 1;
        if (dist < 0.1) dangerousDebris.push({pos: p, dist: dist, dx: dx, dy: dy});
        else safeDebris.push({pos: p, dist: dist, dx: dx, dy: dy});
    });

    const driftSpan = 2 * r + 50;
    const base = cy - r - 25;
    function wrapY(y) {
        let yy = ((y - base) % driftSpan + driftSpan) % driftSpan + base;
        return yy;
    }
    function hash(dx, dy, d) {
        return Math.abs((dx * 7.3 + dy * 4.1 + (d || 0) * 11) % 1);
    }

    safeDebris.forEach(function(d, idx) {
        const dist2d = Math.sqrt(d.dx*d.dx + d.dy*d.dy) || 1;
        const maxRange = 500;
        const scale = Math.min(1, maxRange / d.dist);
        let x = cx + (d.dx / dist2d) * r * scale * 0.75;
        let y = cy - (d.dy / dist2d) * r * scale * 0.75;
        const h = hash(d.dx, d.dy, d.dist);
        const speedMul = 0.6 + h * 0.6;
        y = wrapY(y + radarDriftOffset * speedMul);
        x += Math.sin(radarDriftOffset * 0.02 + h * 6.28) * 4 + Math.cos(radarDriftOffset * 0.035 + idx * 0.5) * 3;
        ctx.fillStyle = COLORS.green;
        ctx.shadowColor = COLORS.green;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    });

    dangerousDebris.forEach(function(d, idx) {
        const dist2d = Math.sqrt(d.dx*d.dx + d.dy*d.dy) || 1;
        const maxRange = 500;
        const scale = Math.min(1, maxRange / d.dist);
        let x = cx + (d.dx / dist2d) * r * scale * 0.75;
        let y = cy - (d.dy / dist2d) * r * scale * 0.75;
        const h = hash(d.dx, d.dy, d.dist);
        const speedMul = 0.6 + (1 - h * 0.5);
        y = wrapY(y + radarDriftOffset * speedMul);
        x += Math.sin(radarDriftOffset * 0.025 + h * 4) * 5 + Math.cos(radarDriftOffset * 0.04 + idx * 0.8) * 4;
        ctx.fillStyle = '#ff4444';
        ctx.strokeStyle = '#ff4444';
        ctx.shadowColor = '#ff4444';
        ctx.shadowBlur = 12;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(x - 6, y - 6);
        ctx.lineTo(x + 6, y + 6);
        ctx.moveTo(x + 6, y - 6);
        ctx.lineTo(x - 6, y + 6);
        ctx.stroke();
        ctx.beginPath();
        ctx.arc(x, y, 8, 0, Math.PI * 2);
        ctx.stroke();
        ctx.shadowBlur = 0;
    });

    const seed = data.seed != null ? Number(data.seed) : 42;
    const ambientCount = 16;
    for (let i = 0; i < ambientCount; i++) {
        const angle = (i * 0.4 + seed * 0.0001) % (Math.PI * 2);
        const radius = (0.25 + (i % 5) / 5 * 0.6) * r;
        let ax = cx + Math.sin(angle) * radius;
        let ay = cy - Math.cos(angle) * radius;
        const h = (Math.sin(i * 2.1 + seed * 0.01) * 0.5 + 0.5);
        const speedMul = 0.5 + h * 0.6;
        ay = wrapY(ay + radarDriftOffset * speedMul);
        ax += Math.sin(radarDriftOffset * 0.018 + i * 1.2) * 5 + Math.cos(radarDriftOffset * 0.028 + i * 0.7) * 4;
        ctx.fillStyle = 'rgba(0, 212, 255, 0.5)';
        ctx.shadowColor = 'rgba(0, 212, 255, 0.35)';
        ctx.shadowBlur = 4;
        ctx.beginPath();
        ctx.arc(ax, ay, 2.5, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    }

    ctx.restore();
}

function runRadarAnimation() {
    if (!lastMissionData) {
        radarAnimationId = null;
        return;
    }
    const dashboard = document.getElementById('dashboard');
    if (!dashboard || dashboard.classList.contains('dashboard-hidden')) {
        radarAnimationId = null;
        return;
    }
    radarSweepAngle += 0.012;
    if (radarSweepAngle >= Math.PI * 2) radarSweepAngle -= Math.PI * 2;
    radarDriftOffset += RADAR_DRIFT_SPEED;
    drawRadar(lastMissionData, radarSweepAngle);
    radarAnimationId = requestAnimationFrame(runRadarAnimation);
}

function updateHealthBars(data) {
    const s = data.spacecraft_status || {};
    const fuels = data.fuels || [];
    const conf = data.confidences || [];
    const fuelPct = fuels.length ? (fuels[fuels.length - 1] / (fuels[0] || 1)) * 100 : 100;
    const fusionPct = conf.length ? conf[conf.length - 1] * 100 : 100;
    const setBar = (id, pct) => {
        const bar = document.getElementById(id);
        if (bar) bar.style.width = Math.min(100, Math.max(0, pct)) + '%';
    };
    const setPct = (id, pct) => {
        const el = document.getElementById(id);
        if (el) el.textContent = Math.round(pct) + '%';
    };
    setBar('bar-prop', fuelPct);
    setPct('pct-prop', fuelPct);
    setBar('bar-ai', fusionPct);
    setPct('pct-ai', fusionPct);
}

function updateCopilotCards(data) {
    const cards = document.getElementById('copilot-cards');
    if (!cards) return;
    const s = data.spacecraft_status || {};
    const lines = [
        'Optimal trajectory confirmed. Fuel efficiency nominal. T+00:12',
        'Micro-debris field detected ahead. Recommending correction. T+00:34',
        'Fusion harmonics nominal. Thermal protection nominal. T+01:07',
    ];
    if (s.altitude != null) lines[0] = 'Altitude ' + s.altitude.toFixed(0) + ' km. Trajectory nominal. T+00:12';
    if (s.debris_remaining != null) lines[1] = s.debris_remaining + ' debris in range. Recommending correction. T+00:34';
    cards.innerHTML = lines.map((text, i) => {
        const cls = ['card-blue', 'card-orange', 'card-green'][i];
        return '<div class="copilot-card ' + cls + '">' + text + '</div>';
    }).join('');
}

function updateThreatCount(data) {
    const n = (data.debris_positions && data.debris_positions.length) || (data.spacecraft_status && data.spacecraft_status.debris_remaining) || 0;
    const el = document.getElementById('threat-count');
    if (el) el.textContent = n;
}

function updateBatteryShields(data) {
    const conf = data.confidences || [];
    const shieldPct = conf.length ? Math.round(conf[conf.length - 1] * 47 + 50) : 47;
    const shieldEl = document.getElementById('shields-pct');
    if (shieldEl) shieldEl.textContent = Math.min(100, shieldPct);
}

// Terminal / logs — append real-time style lines
function appendTerminalLine(text, className) {
    const out = document.getElementById('terminal-output');
    if (!out) return;
    const line = document.createElement('div');
    line.className = 'line' + (className ? ' ' + className : '');
    line.textContent = text;
    out.appendChild(line);
    out.scrollTop = out.scrollHeight;
}

function updateTerminal(data) {
    const steps = data.times ? data.times.length : 0;
    appendTerminalLine('[ORBIT] Track updated. Points: ' + steps, 'sys');
    appendTerminalLine('[TELEMETRY] ECI stream received.', 'sys');
    appendTerminalLine('[FUSION] Confidence nominal.', 'ok');
    if (data.spacecraft_status) {
        const s = data.spacecraft_status;
        appendTerminalLine('[RES] Altitude ' + (s.altitude != null ? s.altitude.toFixed(1) : '—') + ' km | Fuel ' + (s.fuel != null ? s.fuel.toFixed(1) : '—') + ' kg | Debris ' + (s.debris_remaining != null ? s.debris_remaining : '—'), 'line');
    }
    appendTerminalLine('[SYS] Mission dashboard ready.', 'ok');
}

function updateStatus(data) {
    const set = (id, text) => {
        const el = document.getElementById(id);
        if (el) el.textContent = text;
    };
    set('status-network', 'OK');
    set('status-compute', 'LOW');
    set('status-model', 'LOADED');
    set('status-runtime', 'NOMINAL');
    set('status-fusion', 'ACTIVE');
    if (data.spacecraft_status) {
        const s = data.spacecraft_status;
        if (s.altitude != null) set('status-runtime', 'ALT ' + s.altitude.toFixed(0) + ' km');
    }
}

// 3D Orbit visualization
function renderOrbitChart(data) {
    // Траектория корабля — экватор (круг в плоскости XY), бесконечное движение
    const positions = getEquatorialOrbitPositions();
    const debrisPos = data.debris_positions || [];
    var allDebris = data.debris || [];
    var debrisList = allDebris.filter(function(d) { return !debrisRemovedFromMap[String(d.object_id)]; });
    const scPos = positions[0]; // позиция корабля для линий до мусора (анимация двигает маркер по экватору)

    // Generate Earth sphere
    const earthTheta = Array.from({length: 20}, (_, i) => i * Math.PI / 19);
    const earthPhi = Array.from({length: 40}, (_, i) => i * 2 * Math.PI / 39);

    const earthX = [], earthY = [], earthZ = [];
    earthTheta.forEach(theta => {
        const row_x = [], row_y = [], row_z = [];
        earthPhi.forEach(phi => {
            row_x.push(EARTH_RADIUS * Math.sin(theta) * Math.cos(phi));
            row_y.push(EARTH_RADIUS * Math.sin(theta) * Math.sin(phi));
            row_z.push(EARTH_RADIUS * Math.cos(theta));
        });
        earthX.push(row_x);
        earthY.push(row_y);
        earthZ.push(row_z);
    });

    const traces = [
        // Earth
        {
            type: 'surface',
            x: earthX,
            y: earthY,
            z: earthZ,
            colorscale: [[0, '#1a3a52'], [1, '#1a3a52']],
            showscale: false,
            opacity: 0.5,
            hoverinfo: 'skip',
        },
        // Spacecraft trajectory — semi-transparent line
        {
            type: 'scatter3d',
            mode: 'lines',
            x: positions.map(p => p[0]),
            y: positions.map(p => p[1]),
            z: positions.map(p => p[2]),
            line: {
                color: 'rgba(128, 0, 0, 0.45)',
                width: 3,
            },
            name: 'Trajectory',
        },
        // Start marker
        {
            type: 'scatter3d',
            mode: 'markers',
            x: [positions[0][0]],
            y: [positions[0][1]],
            z: [positions[0][2]],
            marker: {
                size: 6,
                color: COLORS.green,
                symbol: 'circle',
                line: { color: '#fff', width: 1 },
            },
            name: 'Start',
        },
        // Our spacecraft (Корабль) — prominent marker
        {
            type: 'scatter3d',
            mode: 'markers',
            x: [scPos[0]],
            y: [scPos[1]],
            z: [scPos[2]],
            marker: {
                size: 12,
                color: COLORS.cyan,
                symbol: 'diamond',
                line: { color: '#fff', width: 2 },
            },
            name: 'Spacecraft',
        },
    ];

    // Distance lines: ship to nearest N debris (max 5)
    const N_DISTANCE_LINES = 5;
    if (debrisList.length > 0) {
        const sorted = debrisList.slice().sort((a, b) => a.distance_km - b.distance_km);
        const nearest = sorted.slice(0, N_DISTANCE_LINES);
        const lineX = [], lineY = [], lineZ = [];
        nearest.forEach(d => {
            const p = d.position;
            lineX.push(scPos[0], p[0], null);
            lineY.push(scPos[1], p[1], null);
            lineZ.push(scPos[2], p[2], null);
        });
        if (lineX.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'lines',
                x: lineX,
                y: lineY,
                z: lineZ,
                line: { color: 'rgba(255, 140, 66, 0.5)', width: 1, dash: 'dot' },
                name: 'Distance to debris',
                hoverinfo: 'skip',
            });
        }
    }

    // Debris trajectory lines (velocity direction, ~6 km ahead)
    const TRAJ_SCALE = 6;
    if (debrisList.length > 0 && debrisList[0].velocity) {
        const trajX = [], trajY = [], trajZ = [];
        debrisList.forEach(d => {
            const p = d.position, v = d.velocity;
            if (!v || v.length < 3) return;
            trajX.push(p[0], p[0] + v[0] * TRAJ_SCALE, null);
            trajY.push(p[1], p[1] + v[1] * TRAJ_SCALE, null);
            trajZ.push(p[2], p[2] + v[2] * TRAJ_SCALE, null);
        });
        if (trajX.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'lines',
                x: trajX,
                y: trajY,
                z: trajZ,
                line: { color: 'rgba(255, 140, 66, 0.4)', width: 1 },
                name: 'Debris trajectory',
                hoverinfo: 'skip',
            });
        }
    }

    // Add debris: обход (сзади) — сначала; затем остальной мусор
    const sizeToMarker = (size) => Math.max(2, Math.min(14, 2 + (size || 0.5) * 2));
    const avoidedDebris = debrisList.filter(d => debrisAvoided[String(d.object_id)]);
    const normalDebris = debrisList.filter(d => !debrisAvoided[String(d.object_id)]);
    const safeNormal = normalDebris.filter(d => d.distance_km >= 0.1);
    const dangerousNormal = normalDebris.filter(d => d.distance_km < 0.1);

    if (avoidedDebris.length > 0) {
        traces.push({
            type: 'scatter3d',
            mode: 'markers',
            x: avoidedDebris.map(d => d.position[0]),
            y: avoidedDebris.map(d => d.position[1]),
            z: avoidedDebris.map(d => d.position[2]),
            customdata: avoidedDebris.map(d => [d.size, d.mass, d.debris_type, d.material || 'Aluminum', d.velocity, d.distance_km, d.relative_velocity_km_s, d.object_id]),
            marker: {
                size: avoidedDebris.map(d => sizeToMarker(d.size)),
                color: COLORS.orange,
                opacity: 0.45,
            },
            text: avoidedDebris.map(d => '(avoided) ' + (d.distance_km * 1000).toFixed(0) + ' m'),
            hoverinfo: 'text',
            name: 'Debris (avoided)',
        });
    }

    if (normalDebris.length > 0) {
        if (safeNormal.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: safeNormal.map(d => d.position[0]),
                y: safeNormal.map(d => d.position[1]),
                z: safeNormal.map(d => d.position[2]),
                customdata: safeNormal.map(d => [d.size, d.mass, d.debris_type, d.material || 'Aluminum', d.velocity, d.distance_km, d.relative_velocity_km_s, d.object_id]),
                marker: {
                    size: safeNormal.map(d => sizeToMarker(d.size)),
                    color: COLORS.orange,
                    opacity: 0.7,
                },
                text: safeNormal.map(d => `${(d.distance_km * 1000).toFixed(0)} m · ${d.debris_type} · ${d.material || ''}`),
                hoverinfo: 'text',
                name: `Debris (${safeNormal.length})`,
            });
        }
        if (dangerousNormal.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: dangerousNormal.map(d => d.position[0]),
                y: dangerousNormal.map(d => d.position[1]),
                z: dangerousNormal.map(d => d.position[2]),
                customdata: dangerousNormal.map(d => [d.size, d.mass, d.debris_type, d.material || 'Aluminum', d.velocity, d.distance_km, d.relative_velocity_km_s, d.object_id]),
                marker: {
                    size: dangerousNormal.map(d => sizeToMarker(d.size)),
                    color: '#ff4444',
                    opacity: 0.9,
                    symbol: 'x',
                    line: { color: '#fff', width: 2 },
                },
                text: dangerousNormal.map(d => `${(d.distance_km * 1000).toFixed(0)} m · ${d.debris_type} · ${d.material || ''}`),
                hoverinfo: 'text',
                name: `Dangerous debris (${dangerousNormal.length})`,
            });
        }
    }
    if (debrisList.length === 0 && debrisPos.length > 0) {
        const safeDebris = [];
        const dangerousDebris = [];
        debrisPos.forEach(p => {
            const dist = Math.sqrt((p[0]-scPos[0])**2 + (p[1]-scPos[1])**2 + (p[2]-scPos[2])**2);
            if (dist < 0.1) dangerousDebris.push(p);
            else safeDebris.push(p);
        });
        if (safeDebris.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: safeDebris.map(p => p[0]),
                y: safeDebris.map(p => p[1]),
                z: safeDebris.map(p => p[2]),
                marker: { size: 3, color: COLORS.orange, opacity: 0.6 },
                name: `Debris (${safeDebris.length})`,
            });
        }
        if (dangerousDebris.length > 0) {
            traces.push({
                type: 'scatter3d',
                mode: 'markers',
                x: dangerousDebris.map(p => p[0]),
                y: dangerousDebris.map(p => p[1]),
                z: dangerousDebris.map(p => p[2]),
                marker: { size: 12, color: '#ff4444', opacity: 0.9, symbol: 'x', line: { color: '#fff', width: 2 } },
                name: `Dangerous debris (${dangerousDebris.length})`,
            });
        }
    }

    var orbitChartEl = document.getElementById('orbit-chart');
    var savedCamera = null;
    if (orbitChartEl && orbitChartEl._fullLayout && orbitChartEl._fullLayout.scene && orbitChartEl._fullLayout.scene.camera) {
        savedCamera = JSON.parse(JSON.stringify(orbitChartEl._fullLayout.scene.camera));
    }
    var sceneLayout = {
        xaxis: { title: 'X (km)', gridcolor: COLORS.grid, backgroundcolor: COLORS.panel, color: COLORS.textSecondary },
        yaxis: { title: 'Y (km)', gridcolor: COLORS.grid, backgroundcolor: COLORS.panel, color: COLORS.textSecondary },
        zaxis: { title: 'Z (km)', gridcolor: COLORS.grid, backgroundcolor: COLORS.panel, color: COLORS.textSecondary },
        bgcolor: COLORS.panel,
        aspectmode: 'cube',
        dragmode: 'orbit',
        hovermode: 'closest',
    };
    if (savedCamera) sceneLayout.camera = savedCamera;
    var layout = Object.assign({}, commonLayout, {
        scene: sceneLayout,
        margin: { t: 10, r: 10, b: 10, l: 10 },
        width: 700,
        height: 280,
    });
    // Фиксированный размер графика — не менять при перерисовке, чтобы сфера всегда вращалась
    var plotOpts = { responsive: false, scrollZoom: true, displayModeBar: true, modeBarButtonsToRemove: ['lasso2d', 'select2d'] };
    function afterPlotUpdate() {
        // После react сфера может перестать крутиться: принудительно включаем орбитальное вращение
        try { Plotly.relayout(orbitChartEl, { 'scene.dragmode': 'orbit' }); } catch (e) {}
    }
    if (orbitChartEl.data && orbitChartEl.data.length > 0) {
        var reactPromise = Plotly.react(orbitChartEl, traces, layout, plotOpts);
        if (reactPromise && typeof reactPromise.then === 'function') {
            reactPromise.then(afterPlotUpdate, afterPlotUpdate);
        } else {
            setTimeout(afterPlotUpdate, 50);
        }
    } else {
        var newPlotPromise = Plotly.newPlot(orbitChartEl, traces, layout, plotOpts);
        if (newPlotPromise && typeof newPlotPromise.then === 'function') {
            newPlotPromise.then(afterPlotUpdate, afterPlotUpdate);
        } else {
            setTimeout(afterPlotUpdate, 50);
        }
    }

    // Панель «Преград» не скрываем никогда; только обновляем счётчик (при утилизации/обходе — на 1 меньше)
    var obstaclesCountEl = document.getElementById('orbit-obstacles-count');
    if (obstaclesCountEl) obstaclesCountEl.textContent = debrisList.length;

    // Сфера должна ВСЕГДА крутиться: вешаем mousedown один раз, иначе после утилизации вращение «застывает»
    if (!orbitChartEl._orbitChartMouseBound) {
        orbitChartEl._orbitChartMouseBound = true;
        orbitChartEl.addEventListener('mousedown', function(ev) {
            if (!ev.target || !ev.target.closest || ev.target.closest('.modebar-container')) return;
            orbitChartUserDragging = true;
            if (orbitShipAnimationId != null) {
                cancelAnimationFrame(orbitShipAnimationId);
                orbitShipAnimationId = null;
            }
        }, true);
    }

    // Ship at 5fps when idle; loop is stopped while dragging and restarted on mouseup
    if (orbitShipAnimationId != null) {
        cancelAnimationFrame(orbitShipAnimationId);
        orbitShipAnimationId = null;
    }
    startOrbitShipAnimation();

    // Click on debris: show info + Arduino live data
    orbitChartEl.on('plotly_click', function(data) {
        if (!data.points || data.points.length === 0) return;
        const pt = data.points[0];
        const cd = pt.customdata;
        if (!cd || !Array.isArray(cd) || cd.length < 2) return;
        const baseSize = cd[0], debrisType = cd[2], material = cd[3];
        const velocity = cd[4], distanceKm = cd[5], objectId = cd[7];
        const popup = document.getElementById('orbit-debris-popup');
        if (popup) {
            var sd = (distanceKm != null && distanceKm !== '') ? Number(distanceKm) : null;
            _orbitPopupSimDistanceKm = (sd != null && !isNaN(sd)) ? sd : null;
            _orbitPopupArduinoOpen = true;
            popup.style.display = 'block';
            var sizeEl = document.getElementById('orbit-popup-size');
            if (sizeEl) sizeEl.textContent = (baseSize || 0.5).toFixed(2);
            var matEl = document.getElementById('orbit-popup-material');
            if (matEl) matEl.textContent = material || 'Aluminum';
            var typeEl = document.getElementById('orbit-popup-type');
            if (typeEl) typeEl.textContent = getDebrisTypeLabel(debrisType);
            var trajEl = document.getElementById('orbit-popup-trajectory');
            if (trajEl) trajEl.textContent = formatTrajectory(velocity);
            updatePopupWithArduino();
            fetchArduinoLive().then(function () {
                if (_orbitPopupArduinoOpen) updatePopupWithArduino();
            });
            startPopupArduinoPolling();
        }
    });

    var popupCloseBtn = document.getElementById('orbit-popup-close');
    if (popupCloseBtn && !window._orbitPopupCloseBound) {
        window._orbitPopupCloseBound = true;
        popupCloseBtn.addEventListener('click', function() {
            var p = document.getElementById('orbit-debris-popup');
            if (p) p.style.display = 'none';
            _orbitPopupSimDistanceKm = null;
            _orbitPopupArduinoOpen = false;
            stopPopupArduinoPolling();
        });
    }
}

var TEMP_FACTOR = 0.8;
var TEMP_WARN_RAW = 32;
var PROXIMITY_WARN_KM = 100;
/** Combined Arduino anomaly: distance < 100 km and raw temperature > 30 °C */
var COMPOUND_PROXIMITY_TEMP_C = 30;

function updateDangerStatus(data) {
    var dangerLevels = data.danger_levels || [];
    var currentDanger = dangerLevels.length > 0 ? dangerLevels[dangerLevels.length - 1] : 0.0;

    var dangerBar = document.getElementById('danger-level-bar');
    var dangerValue = document.getElementById('danger-level-value');
    if (dangerBar) {
        dangerBar.style.width = (currentDanger * 100) + '%';
        dangerBar.style.background = currentDanger > 0.7 ? '#ff4444' :
                                     currentDanger > 0.4 ? '#ff8c42' :
                                     currentDanger > 0.1 ? '#ffaa00' : COLORS.green;
    }

    var d = _arduinoLive || {};
    var magOn = arduinoSensorIsTrue(d.magnetic) || arduinoSensorIsTrue(d.hall);
    var vibOn = arduinoSensorIsTrue(d.vibration);
    var tempRaw = arduinoLiveNum(d.temperature);
    var tempVal = tempRaw != null ? tempRaw * TEMP_FACTOR : null;
    var tempWarn = tempRaw != null && tempRaw > TEMP_WARN_RAW;

    var closestDistKm = arduinoLiveNum(d.distance_km);
    var proximityWarn = closestDistKm != null && closestDistKm < PROXIMITY_WARN_KM;
    var compoundProxTemp =
        closestDistKm != null &&
        closestDistKm < PROXIMITY_WARN_KM &&
        tempRaw != null &&
        tempRaw > COMPOUND_PROXIMITY_TEMP_C;

    var activeCount = (magOn ? 1 : 0) + (vibOn ? 1 : 0);
    if (compoundProxTemp) {
        activeCount += 2;
    } else {
        if (tempWarn) activeCount += 1;
        if (proximityWarn) activeCount += 1;
    }

    if (dangerValue) {
        var dangerLabel;
        if (activeCount >= 3 || currentDanger >= 0.6) dangerLabel = 'high';
        else if (activeCount >= 1 || currentDanger >= 0.2) dangerLabel = 'medium';
        else dangerLabel = 'low';
        dangerValue.textContent = dangerLabel;
        dangerValue.style.color = dangerLabel === 'high' ? '#ff4444' :
                                  dangerLabel === 'medium' ? '#ff8c42' : COLORS.green;
    }

    var magEl = document.getElementById('danger-magnetic');
    if (magEl) {
        magEl.textContent = magOn ? 'DETECTED' : 'Clear';
        magEl.style.color = magOn ? '#ff4444' : COLORS.green;
    }
    var tempEl = document.getElementById('danger-temperature');
    if (tempEl) {
        if (tempVal == null) {
            tempEl.textContent = '—';
            tempEl.style.color = '';
        } else if (compoundProxTemp) {
            tempEl.textContent = tempVal.toFixed(1) + ' °C — CRITICAL (prox+temp)';
            tempEl.style.color = '#ff4444';
        } else if (tempWarn) {
            tempEl.textContent = tempVal.toFixed(1) + ' °C — WARNING';
            tempEl.style.color = '#ff8c42';
        } else {
            tempEl.textContent = tempVal.toFixed(1) + ' °C — OK';
            tempEl.style.color = COLORS.green;
        }
    }
    var vibEl = document.getElementById('danger-vibration');
    if (vibEl) {
        vibEl.textContent = vibOn ? 'DETECTED' : 'None';
        vibEl.style.color = vibOn ? '#ff8c42' : COLORS.green;
    }
    var proxEl = document.getElementById('danger-proximity');
    if (proxEl) {
        if (closestDistKm != null) {
            if (compoundProxTemp) {
                proxEl.textContent = 'CRITICAL';
                proxEl.style.color = '#ff4444';
            } else {
                proxEl.textContent = proximityWarn ? 'Not' : 'Safe';
                proxEl.style.color = proximityWarn ? '#ff4444' : COLORS.green;
            }
        } else {
            proxEl.textContent = 'Safe';
            proxEl.style.color = COLORS.green;
        }
    }

    var sysStatus = document.getElementById('sys-status');
    var statusDot = document.getElementById('status-dot');
    var warningsCount = document.getElementById('warnings-count');
    var warningsText = document.getElementById('warnings-text');
    var warningsNum = document.getElementById('warnings-num');

    var statusText = 'SYS NOMINAL';
    var statusColor = COLORS.green;
    if (activeCount >= 3) { statusText = 'STATUS: CRITICAL'; statusColor = '#ff4444'; }
    else if (activeCount >= 1) { statusText = 'STATUS: CAUTION'; statusColor = '#ff8c42'; }

    if (sysStatus) { sysStatus.textContent = statusText; sysStatus.style.color = statusColor; }
    if (statusDot) { statusDot.style.background = statusColor; statusDot.style.boxShadow = '0 0 10px ' + statusColor; }
    if (activeCount > 0) {
        if (warningsCount) warningsCount.style.display = 'inline';
        if (warningsText) warningsText.style.display = 'inline';
        if (warningsNum) warningsNum.textContent = activeCount;
    } else {
        if (warningsCount) warningsCount.style.display = 'none';
        if (warningsText) warningsText.style.display = 'none';
    }
}

/** Обложка с сидом: панели аномалий не показываем (только после Start/Add). */
function isCoverVisible() {
    var c = document.getElementById('cover');
    if (!c) return false;
    var s = window.getComputedStyle(c);
    return s.display !== 'none' && s.visibility !== 'hidden';
}

/** Контекст для панели опасности до загрузки миссии (обложка) — те же правила Arduino, что и в миссии */
function dangerWarningsDataContext() {
    if (lastMissionData) return lastMissionData;
    return {
        debris: [],
        collision_warnings: [],
        anomaly_detections: [],
        low_fuel_warnings: [],
        danger_levels: [0],
    };
}

// Update danger warnings panel
function updateDangerWarnings(data) {
    const dangerPanel = document.getElementById('danger-panel');
    const dangerContent = document.getElementById('danger-content');
    
    if (!dangerPanel || !dangerContent) return;
    
    var warnings = [];
    var rawAnomalies = [];
    var d = _arduinoLive || {};
    var nowSec = Date.now() / 1000;
    var closestDistW = arduinoLiveNum(d.distance_km);
    var tempRawW = arduinoLiveNum(d.temperature);
    var proximityUnder100 = closestDistW != null && closestDistW < PROXIMITY_WARN_KM;
    var compoundProxTemp =
        proximityUnder100 && tempRawW != null && tempRawW > COMPOUND_PROXIMITY_TEMP_C;

    // Reset suppression when all sensors return to normal
    try {
        var hasMagAnomaly = arduinoSensorIsTrue(d.magnetic) || arduinoSensorIsTrue(d.hall);
        var hasVibAnomaly = arduinoSensorIsTrue(d.vibration);
        var hasTempAnomaly = tempRawW != null && tempRawW > TEMP_WARN_RAW;
        if (!hasMagAnomaly && !hasVibAnomaly && !hasTempAnomaly && !compoundProxTemp) {
            _arduinoAnomaliesSuppressed = false;
        }
    } catch (e) { /* ignore */ }

    // 1) Magnetic field (rising-edge events + sustained)
    try {
        for (var hi = _hallMagneticAnomalyEvents.length - 1; hi >= 0; hi--) {
            var hev = _hallMagneticAnomalyEvents[hi];
            if (!hev || dismissedAnomalyKeys[hev.id]) continue;
            rawAnomalies.unshift({ time: hev.time, score: 0.85, type: 'MAGNETIC_FIELD', stableKey: hev.id });
        }
        if (!_arduinoAnomaliesSuppressed && (arduinoSensorIsTrue(d.magnetic) || arduinoSensorIsTrue(d.hall))) {
            var anyOpenHall = false;
            for (var hiu = 0; hiu < _hallMagneticAnomalyEvents.length; hiu++) {
                if (_hallMagneticAnomalyEvents[hiu] && !dismissedAnomalyKeys[_hallMagneticAnomalyEvents[hiu].id]) { anyOpenHall = true; break; }
            }
            if (!anyOpenHall && !dismissedAnomalyKeys['arduino_magnetic_sustained']) {
                rawAnomalies.push({ time: nowSec, score: 0.85, type: 'MAGNETIC_FIELD', stableKey: 'arduino_magnetic_sustained' });
            }
        }
    } catch (e) { /* ignore */ }

    // 2) Vibration
    try {
        if (!_arduinoAnomaliesSuppressed && hasVibAnomaly) {
            rawAnomalies.push({ time: nowSec, score: 0.8, type: 'VIBRATION_SPIKE' });
        }
    } catch (e) { /* ignore */ }

    // 3) Combined: ultrasonic < 100 km AND temperature > 30 °C (single high-severity anomaly)
    try {
        if (
            !_arduinoAnomaliesSuppressed &&
            compoundProxTemp &&
            !dismissedAnomalyKeys['arduino_proximity_temp_compound']
        ) {
            rawAnomalies.push({
                time: nowSec,
                score: 0.95,
                type: 'PROXIMITY_THERMAL_ANOMALY',
                stableKey: 'arduino_proximity_temp_compound',
            });
        }
    } catch (e) { /* ignore */ }

    // 4) Temperature anomaly (standalone — skipped when combined rule is active)
    try {
        if (
            !_arduinoAnomaliesSuppressed &&
            hasTempAnomaly &&
            !compoundProxTemp &&
            !dismissedAnomalyKeys['arduino_temp_warn']
        ) {
            rawAnomalies.push({ time: nowSec, score: 0.7, type: 'TEMP_ANOMALY', stableKey: 'arduino_temp_warn' });
        }
    } catch (e) { /* ignore */ }

    // 5) Proximity — ultrasonic distance < 100 km (standalone — skipped when combined rule is active)
    if (
        closestDistW != null &&
        closestDistW < PROXIMITY_WARN_KM &&
        !compoundProxTemp &&
        !dismissedAnomalyKeys['proximity_warn']
    ) {
        rawAnomalies.push({ time: nowSec, score: 0.75, type: 'PROXIMITY_ALERT', stableKey: 'proximity_warn' });
    }

    var visibleAnomalies = [];
    rawAnomalies.forEach(function(a, idx) {
        var key = a.stableKey != null ? String(a.stableKey) : ('anomaly_' + (a.time != null ? a.time : 0) + '_' + (a.score != null ? a.score : 0) + '_' + idx);
        if (dismissedAnomalyKeys[key]) return;
        visibleAnomalies.push(a);
        var anomalyTitle;
        if (a.type === 'MAGNETIC_FIELD') anomalyTitle = 'ANOMALY! Magnetic field detected.';
        else if (a.type === 'VIBRATION_SPIKE') anomalyTitle = 'ANOMALY! Hull vibration detected.';
        else if (a.type === 'TEMP_ANOMALY') anomalyTitle = 'ANOMALY! Temperature out of range.';
        else if (a.type === 'PROXIMITY_ALERT') anomalyTitle = 'DANGER! Object too close (' + (closestDistW != null ? closestDistW.toFixed(1) : '?') + ' km).';
        else if (a.type === 'PROXIMITY_THERMAL_ANOMALY') {
            anomalyTitle =
                'ANOMALY! Ultrasonic < 100 km and temperature > 30 °C (' +
                (closestDistW != null ? closestDistW.toFixed(1) : '?') +
                ' km, ' +
                (tempRawW != null ? tempRawW.toFixed(1) : '?') +
                ' °C).';
        } else anomalyTitle = 'ANOMALY! Level: ' + ((a.score || 0) * 100).toFixed(1) + '%';
        warnings.push({
            icon: '🚨',
            text: anomalyTitle,
            severity: (a.score != null && a.score > 0.7) ? 'HIGH' : 'MEDIUM',
            time: a.time != null ? a.time : 0,
            isAnomaly: true,
            reason: getAnomalyReasonLabel(a.type),
            anomalyKey: key,
            immediatePanel: true,
        });
    });
    
    var needsImmediatePanel = warnings.length > 0;
    var liveArduinoUrgent =
        !_arduinoAnomaliesSuppressed &&
        (hasMagAnomaly || hasVibAnomaly || hasTempAnomaly || compoundProxTemp);
    if (warnings.length > 0) {
        if (needsImmediatePanel || liveArduinoUrgent) {
            if (dangerPanelShowTimeoutId) {
                clearTimeout(dangerPanelShowTimeoutId);
                dangerPanelShowTimeoutId = null;
            }
            dangerPanel.style.display = 'block';
        } else if (!dangerPanelShowTimeoutId) {
            dangerPanelShowTimeoutId = setTimeout(function() {
                dangerPanelShowTimeoutId = null;
                if (dangerPanel) dangerPanel.style.display = 'block';
            }, DANGER_PANEL_DELAY_MS);
        }
        var anomalyWarnings = warnings.filter(function(w) { return w.isAnomaly; });
        var html = warnings.map(function(w) {
            var severityClass = w.severity === 'HIGH' ? 'danger-high' : 'danger-medium';
            var reasonBlock = w.isAnomaly && w.reason
                ? '<div class="danger-warning-reason">Cause: ' + w.reason + '</div>'
                : '';
            var btnBlock = w.isAnomaly && w.anomalyKey
                ? '<button type="button" class="danger-warning-resolve" data-anomaly-key="' + w.anomalyKey + '">Resolve</button>'
                : '';
            return '<div class="danger-warning ' + severityClass + '" data-anomaly-key="' + (w.anomalyKey || '') + '">' +
                '<span class="danger-warning-icon">' + w.icon + '</span>' +
                '<div class="danger-warning-body">' +
                '<span class="danger-warning-text">' + w.text + '</span>' +
                reasonBlock +
                btnBlock +
                '</div>' +
                '<span class="danger-warning-time">T+' + (w.time != null ? w.time.toFixed(1) : '0') + 's</span>' +
                '</div>';
        }).join('');
        if (anomalyWarnings.length > 0) {
            html += '<div class="danger-resolve-all-wrap"><button type="button" class="danger-warning-resolve-all" id="danger-resolve-all-btn">Resolve all</button></div>';
        }
        dangerContent.innerHTML = html;
        dangerContent.querySelectorAll('.danger-warning-resolve[data-anomaly-key]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var key = btn.getAttribute('data-anomaly-key');
                if (key) dismissedAnomalyKeys[key] = true;
                updateDangerWarnings(dangerWarningsDataContext());
            });
        });
        var resolveAllBtn = document.getElementById('danger-resolve-all-btn');
        if (resolveAllBtn) {
            resolveAllBtn.addEventListener('click', function() {
                // Помечаем все текущие аномалии (включая созданные по данным Arduino) как «устранённые»
                anomalyWarnings.forEach(function(w) {
                    if (w.anomalyKey) dismissedAnomalyKeys[w.anomalyKey] = true;
                });
                // Временно подавляем генерацию новых Arduino‑аномалий, пока состояние датчиков не вернётся в норму
                _arduinoAnomaliesSuppressed = true;
                updateDangerWarnings(dangerWarningsDataContext());
            });
        }
        var dataForStatus = Object.assign({}, data, { anomaly_detections: visibleAnomalies });
        updateDangerStatus(dataForStatus);
    } else {
        if (dangerPanelShowTimeoutId) {
            clearTimeout(dangerPanelShowTimeoutId);
            dangerPanelShowTimeoutId = null;
        }
        dangerPanel.style.display = 'none';
        var dataForStatusElse = Object.assign({}, data, { anomaly_detections: visibleAnomalies });
        updateDangerStatus(dataForStatusElse);
    }

    var noAnomalyPanel = document.getElementById('no-anomaly-panel');
    if (noAnomalyPanel && !noAnomalyPanelDismissed) {
        if (typeof visibleAnomalies !== 'undefined' ? visibleAnomalies.length === 0 : (data.anomaly_detections || []).length === 0) {
            noAnomalyPanel.style.display = 'block';
        } else {
            noAnomalyPanel.style.display = 'none';
        }
    }

    if (isCoverVisible()) {
        if (dangerPanelShowTimeoutId) {
            clearTimeout(dangerPanelShowTimeoutId);
            dangerPanelShowTimeoutId = null;
        }
        dangerPanel.style.display = 'none';
        if (noAnomalyPanel) noAnomalyPanel.style.display = 'none';
    }
}

// ── Arduino live sensor data (SSE push + fetch fallback) ───────────────────
var _arduinoLive = {
    distance_km: null,
    distance_cm: null,
    magnetic: null,
    hall: null,
    temperature: null,
    humidity: null,
    vibration: null,
    updated_at: null,
    error: null,
};
var _arduinoPrevMagnetic = null;
var _arduinoPrevHall = null;
var _arduinoPrevVibration = null;
/** События по фронту магнитного датчика (отдельные аномалии): { id, time } */
var _hallMagneticAnomalyEvents = [];
var HALL_MAG_EVENTS_MAX = 40;
var _arduinoAnomaliesSuppressed = false;
/** Previous frame: ultrasonic < 100 km and temp > 30 °C (for rising-edge danger refresh) */
var _arduinoPrevCompoundPT = false;
var _arduinoPollingId = null;
var _arduinoEventSource = null;
/** Последний снимок из /api/arduino/logs/latest — те же поля distance_*, что в таблице лога */
var _arduinoLatestLogSnapshot = null;
/** Фоновая симуляция: последний кадр с /api/virtual-sensors/stream */
var _virtualSensorsLive = null;
var _virtualSensorsEventSource = null;

function debugLog937bb1(location, message, data, runId, hypothesisId) {
    // #region agent log
    fetch('http://127.0.0.1:7369/ingest/c03c92ff-9516-4b93-9c98-35857e7435f1',{method:'POST',headers:{'Content-Type':'application/json','X-Debug-Session-Id':'937bb1'},body:JSON.stringify({sessionId:'937bb1',runId:runId||'start-btn-issue',hypothesisId:hypothesisId||'HX',location:location,message:message,data:data||{},timestamp:Date.now()})}).catch(()=>{});
    // #endregion
}

function arduinoLiveNum(v) {
    if (v == null || v === '') return null;
    var n = typeof v === 'number' ? v : Number(v);
    return isNaN(n) ? null : n;
}

/** Как колонка Distance в ARDUINO SENSOR LOG: см приоритетнее км (строка без суффикса «Arduino»). */
function formatArduinoUltrasonicReading(obj) {
    if (!obj || typeof obj !== 'object') return null;
    var cm = arduinoLiveNum(obj.distance_cm);
    if (cm != null) return cm.toFixed(1) + ' cm';
    var km = arduinoLiveNum(obj.distance_km);
    if (km != null) return km.toFixed(1) + ' km';
    return null;
}

/** Датчик Arduino «включён» (true / 1 / "true" и т.п.) — для магнита и вибрации */
function arduinoSensorIsTrue(v) {
    if (v === true || v === 1) return true;
    if (typeof v === 'string') {
        var s = v.trim().toLowerCase();
        return s === '1' || s === 'true' || s === 'yes' || s === 'on';
    }
    return false;
}

function fetchArduinoLive() {
    return fetch('/api/arduino/live')
        .then(function (r) {
            if (!r.ok) {
                var msg = r.status === 404
                    ? 'No route /api/arduino (install pyserial, restart server)'
                    : ('HTTP ' + r.status);
                _arduinoLive = Object.assign({}, _arduinoLive, { error: msg });
                if (_orbitPopupArduinoOpen) updatePopupWithArduino();
                return null;
            }
            return r.json();
        })
        .then(function (data) {
            if (data && typeof data === 'object') applyArduinoLiveData(data);
            return data;
        })
        .catch(function () {
            _arduinoLive = Object.assign({}, _arduinoLive, { error: 'Network / server unavailable' });
            if (_orbitPopupArduinoOpen) updatePopupWithArduino();
        });
}

function applyArduinoLiveData(data) {
    if (!data || typeof data !== 'object') return;
    _arduinoLive = Object.assign({}, _arduinoLive, data);

    // Магнит / вибрация: сразу пересчитать панель (без 6 с задержки и без ожидания loadMissionData).
    try {
        var prevMag = _arduinoPrevMagnetic;
        var prevHall = _arduinoPrevHall;
        var prevVib = _arduinoPrevVibration;
        var nowMag = _arduinoLive && arduinoSensorIsTrue(_arduinoLive.magnetic);
        var nowHall = _arduinoLive && arduinoSensorIsTrue(_arduinoLive.hall);
        var nowVib = _arduinoLive && arduinoSensorIsTrue(_arduinoLive.vibration);
        var prevField = arduinoSensorIsTrue(prevMag) || arduinoSensorIsTrue(prevHall);
        var liveField = nowMag || nowHall;
        _arduinoPrevMagnetic = _arduinoLive ? _arduinoLive.magnetic : null;
        _arduinoPrevHall = _arduinoLive ? _arduinoLive.hall : null;
        _arduinoPrevVibration = _arduinoLive ? _arduinoLive.vibration : null;
        var fieldRising = !prevField && liveField;
        var vibRising = !arduinoSensorIsTrue(prevVib) && nowVib;
        if (fieldRising || vibRising) {
            _arduinoAnomaliesSuppressed = false;
        }
        if (fieldRising) {
            var hallId = 'hall_mag_' + Date.now() + '_' + Math.random().toString(36).slice(2, 10);
            _hallMagneticAnomalyEvents.push({ id: hallId, time: Date.now() / 1000 });
            while (_hallMagneticAnomalyEvents.length > HALL_MAG_EVENTS_MAX) {
                _hallMagneticAnomalyEvents.shift();
            }
        }
        var tLiveRaw = _arduinoLive ? arduinoLiveNum(_arduinoLive.temperature) : null;
        var distLive = _arduinoLive ? arduinoLiveNum(_arduinoLive.distance_km) : null;
        var compoundLive =
            distLive != null &&
            distLive < PROXIMITY_WARN_KM &&
            tLiveRaw != null &&
            tLiveRaw > COMPOUND_PROXIMITY_TEMP_C;
        var compoundRising = compoundLive && !_arduinoPrevCompoundPT;
        _arduinoPrevCompoundPT = compoundLive;
        var tempAnomLive = (tLiveRaw != null && tLiveRaw > TEMP_WARN_RAW) || compoundLive;
        var anySensorActive = liveField || nowVib || tempAnomLive;
        var mustRefreshDanger = fieldRising || vibRising || compoundRising;
        var _tRes = Date.now();
        if (anySensorActive && (_tRes - _lastArduinoDangerResyncMs >= 400)) {
            _lastArduinoDangerResyncMs = _tRes;
            mustRefreshDanger = true;
        }
        if (mustRefreshDanger) {
            if (dangerPanelShowTimeoutId) {
                clearTimeout(dangerPanelShowTimeoutId);
                dangerPanelShowTimeoutId = null;
            }
            updateDangerWarnings(dangerWarningsDataContext());
        }
    } catch (e) {
        // не даём сбойнуть всей панели, если что-то пошло не так
    }
    if (_orbitPopupArduinoOpen) {
        updatePopupWithArduino();
    }
    updateDangerStatus(dangerWarningsDataContext());
}

function startArduinoPolling() {
    if (_arduinoEventSource || _arduinoPollingId) return;
    if (typeof EventSource !== 'undefined') {
        try {
            _arduinoEventSource = new EventSource('/api/arduino/stream');
            _arduinoEventSource.onmessage = function (ev) {
                try {
                    applyArduinoLiveData(JSON.parse(ev.data));
                } catch (e) { /* ignore */ }
            };
            _arduinoEventSource.onerror = function () {
                /* браузер переподключает EventSource сам */
            };
            fetchArduinoLive();
            return;
        } catch (e) {
            _arduinoEventSource = null;
        }
    }
    fetchArduinoLive();
    _arduinoPollingId = setInterval(fetchArduinoLive, 80);
}

function stopArduinoPolling() {
    if (_arduinoEventSource) {
        _arduinoEventSource.close();
        _arduinoEventSource = null;
    }
    if (_arduinoPollingId) {
        clearInterval(_arduinoPollingId);
        _arduinoPollingId = null;
    }
}

function _vtNum(v, fallback) {
    if (v == null || v === '') return fallback != null ? fallback : null;
    var n = typeof v === 'number' ? v : Number(v);
    return isNaN(n) ? (fallback != null ? fallback : null) : n;
}

/** Push virtual sensor dict (from SSE or mission snapshot) into the SIM panel + shared readouts. */
function updateVirtualSensorsUI(vs) {
    if (!vs || typeof vs !== 'object') return;
    var meta = vs._meta || {};
    var foot = document.getElementById('vt-footer-text');
    var metaEl = document.getElementById('vt-meta');
    var simT = _vtNum(meta.sim_time, null);
    if (meta.seq != null && foot) {
        foot.textContent = '#' + meta.seq + '  t=' + (simT != null ? simT.toFixed(1) : '—') + 's';
    } else if (foot) {
        foot.textContent = 'snapshot';
    }
    if (metaEl) {
        if (meta.seq != null) {
            metaEl.textContent = 'Frame #' + meta.seq + ' · sim t=' + (simT != null ? simT.toFixed(2) : '—') + 's · seed ' + (meta.seed != null ? meta.seed : '—');
        } else {
            metaEl.textContent = 'Mission snapshot — live stream will override when connected';
        }
    }
    var dot = document.getElementById('vt-live-dot');
    if (dot) {
        dot.style.background = '#22c55e';
        dot.style.boxShadow = '0 0 6px #22c55e';
    }

    var motion = vs._motion;
    if (motion) {
        var sk = _vtNum(motion.speed_kmh, null);
        var ak = _vtNum(motion.altitude_km, null);
        if (sk != null) applyVelocityPanel(sk, ak);
    }

    var setTxt = function (id, text) {
        var el = document.getElementById(id);
        if (el) el.textContent = text;
    };

    var gps = vs.gps;
    if (gps) {
        setTxt('vt-gps', gps.fix_valid ? (String(gps.num_satellites) + ' sats') : 'No fix');
    } else {
        setTxt('vt-gps', '—');
    }

    var pow = vs.power;
    if (pow) {
        var sp = _vtNum(pow.solar_power, null);
        setTxt('vt-solar', sp != null ? sp.toFixed(0) + ' W' : '—');
        setTxt('vt-eclipse', pow.eclipse_state ? 'Yes' : 'No');
    }

    var rad = vs.radiation;
    if (rad) {
        var dr = _vtNum(rad.dose_rate, null);
        setTxt('vt-dose', dr != null ? dr.toFixed(3) + ' µSv/h' : '—');
        var envRad = document.getElementById('env-rad');
        if (envRad && dr != null) envRad.textContent = dr.toFixed(3) + ' µSv/h';
        var pf = _vtNum(rad.particle_flux, null);
        var envPart = document.getElementById('env-particle');
        if (envPart && pf != null) envPart.textContent = pf.toFixed(0) + ' p/cm³';
    }

    var sun = vs.sun_sensor;
    if (sun) {
        var si = _vtNum(sun.intensity, null);
        var envSolar = document.getElementById('env-solar');
        if (envSolar && si != null) envSolar.textContent = si.toFixed(0) + ' W/m²';
    }

    var radar = vs.radar;
    if (radar && radar.num_targets != null) {
        setTxt('vt-radar', String(radar.num_targets));
    } else {
        setTxt('vt-radar', '—');
    }

    var prox = vs.proximity;
    if (prox) {
        if (prox.target_detected && _vtNum(prox.range_to_target, null) != null) {
            setTxt('vt-prox', _vtNum(prox.range_to_target, 0).toFixed(2) + ' m');
        } else {
            setTxt('vt-prox', 'No target');
        }
    } else {
        setTxt('vt-prox', '—');
    }

    var therm = vs.thermal;
    if (therm) {
        var mt = _vtNum(therm.mean_temperature, null);
        setTxt('vt-thermal', mt != null ? mt.toFixed(1) + ' °C' : '—');
        var envTemp = document.getElementById('env-temp');
        if (envTemp && mt != null) envTemp.textContent = mt.toFixed(0) + ' °C';
        var readoutTemp = document.getElementById('readout-temp');
        if (readoutTemp && mt != null) readoutTemp.textContent = mt.toFixed(0) + ' °C';
    }

    var dangerProx = document.getElementById('danger-proximity');
    if (dangerProx && vs.proximity) {
        var pr = _vtNum(vs.proximity.range_to_target, null);
        if (vs.proximity.target_detected && pr != null && pr < 15) {
            dangerProx.textContent = 'Not';
            dangerProx.style.color = '#ff4444';
        } else {
            dangerProx.textContent = 'Safe';
            dangerProx.style.color = COLORS.green;
        }
    }

    var accel = vs.accelerometer;
    if (accel && accel.shock_detected) {
        var dv = document.getElementById('danger-vibration');
        if (dv) {
            dv.textContent = 'Shock (sim)';
            dv.style.color = '#ff5555';
        }
    }
}

function startVirtualSensorsStream() {
    if (_virtualSensorsEventSource || typeof EventSource === 'undefined') return;
    try {
        _virtualSensorsEventSource = new EventSource('/api/virtual-sensors/stream');
        _virtualSensorsEventSource.onopen = function () {
            var foot = document.getElementById('vt-footer-text');
            if (foot) foot.textContent = 'connected…';
        };
        _virtualSensorsEventSource.onmessage = function (ev) {
            try {
                var o = JSON.parse(ev.data);
                if (o && typeof o === 'object') {
                    _virtualSensorsLive = o;
                    updateVirtualSensorsUI(o);
                }
            } catch (e) { /* ignore */ }
            if (_orbitPopupArduinoOpen) updatePopupWithArduino();
        };
        _virtualSensorsEventSource.onerror = function () {
            var foot = document.getElementById('vt-footer-text');
            if (foot) foot.textContent = 'reconnecting…';
            var dot = document.getElementById('vt-live-dot');
            if (dot) {
                dot.style.background = '#f97316';
                dot.style.boxShadow = '0 0 6px #f97316';
            }
        };
    } catch (e) {
        _virtualSensorsEventSource = null;
    }
}

function startArduinoAutoConnect() {
    var p = '';
    try {
        p = (localStorage.getItem('arduino_serial_port') || '').trim();
    } catch (e) { /* ignore */ }
    var body = p ? JSON.stringify({ port: p }) : '{}';
    fetch('/api/arduino/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: body,
    }).catch(function () {});
}

function updatePopupWithArduino() {
    var ultraEl = document.getElementById('orbit-popup-ultrasonic');
    var dotEl = document.getElementById('ard-live-dot');
    var d = _arduinoLive;
    var hasData = !!(d && d.updated_at != null);
    var err = d && d.error;

    if (dotEl) {
        if (err) {
            dotEl.style.background = '#f97316';
            dotEl.style.boxShadow = '0 0 6px #f97316';
            dotEl.title = String(err);
        } else {
            dotEl.title = '';
            dotEl.style.background = hasData ? '#22c55e' : '#64748b';
            dotEl.style.boxShadow = hasData ? '0 0 6px #22c55e' : 'none';
        }
    }

    if (ultraEl) {
        var line = formatArduinoUltrasonicReading(d);
        if (!line && _arduinoLatestLogSnapshot) {
            line = formatArduinoUltrasonicReading(_arduinoLatestLogSnapshot);
        }
        if (!line) {
            var vs = _virtualSensorsLive || (lastMissionData && lastMissionData.virtual_sensors);
            var prox = vs && vs.proximity;
            if (prox && prox.target_detected && prox.range_to_target != null) {
                var rm = Number(prox.range_to_target);
                if (!isNaN(rm)) line = rm.toFixed(2) + ' m (sim)';
            }
        }
        ultraEl.textContent = line || '—';
    }
}

function fetchArduinoLogLatestForPopup() {
    return fetch('/api/arduino/logs/latest')
        .then(function (r) { return r.ok ? r.json() : null; })
        .then(function (obj) {
            if (obj && typeof obj === 'object') _arduinoLatestLogSnapshot = obj;
            if (_orbitPopupArduinoOpen) updatePopupWithArduino();
        })
        .catch(function () {});
}

function startPopupArduinoPolling() {
    stopPopupArduinoPolling();
    _popupArduinoLogPollTick = 0;
    fetchArduinoLive();
    fetchArduinoLogLatestForPopup();
    _popupArduinoIntervalId = setInterval(function () {
        fetchArduinoLive();
        _popupArduinoLogPollTick += 1;
        if (_popupArduinoLogPollTick % 8 === 0) {
            fetchArduinoLogLatestForPopup();
        }
    }, 50);
}

function stopPopupArduinoPolling() {
    if (_popupArduinoIntervalId) {
        clearInterval(_popupArduinoIntervalId);
        _popupArduinoIntervalId = null;
    }
    _popupArduinoLogPollTick = 0;
}
// ─────────────────────────────────────────────────────────────────────────────

// Initialize on page load: cover visible, dashboard hidden; no auto-load.
function initCover() {
    startLiveTime();
    // #region agent log
    debugLog937bb1('dashboard.js:initCover:entry', 'initCover entry', { readyState: document.readyState }, 'start-btn-issue', 'H1');
    // #endregion
    document.addEventListener('click', function(e) {
        if (e.target && (e.target.id === 'no-anomaly-close' || (e.target.closest && e.target.closest('#no-anomaly-close')))) {
            noAnomalyPanelDismissed = true;
            var p = document.getElementById('no-anomaly-panel');
            if (p) p.style.display = 'none';
        }
    });

    var cover = document.getElementById('cover');
    var dashboard = document.getElementById('dashboard');
    var startBtn = document.getElementById('cover-start-btn');
    var coverStatus = document.getElementById('cover-status');
    var seedBlock = document.getElementById('cover-seed-block');
    var generatedSeedEl = document.getElementById('generated-seed-value');
    var seedInput = document.getElementById('seed-input');
    // #region agent log
    debugLog937bb1('dashboard.js:initCover:elements', 'cover elements lookup', { hasCover: !!cover, hasDashboard: !!dashboard, hasStartBtn: !!startBtn, hasSeedInput: !!seedInput }, 'start-btn-issue', 'H1');
    // #endregion

    function setCoverStatus(text, isError) {
        if (!coverStatus) return;
        coverStatus.textContent = text || '';
        coverStatus.classList.toggle('error', !!isError);
    }

    function setLoading(loading) {
        if (startBtn) {
            startBtn.disabled = loading;
            startBtn.textContent = loading ? 'Generating…' : 'Start';
        }
        if (!loading) setCoverStatus('');
    }

    function setAddLoading(loading) {
        var addBtn = document.getElementById('cover-add-btn');
        if (addBtn) {
            addBtn.disabled = loading;
            addBtn.textContent = loading ? 'Loading…' : 'Add';
        }
        if (!loading) setCoverStatus('');
    }

    function showSeedOnCover(seed) {
        if (generatedSeedEl) generatedSeedEl.textContent = seed;
        if (seedBlock) seedBlock.style.display = 'block';
        if (seedInput) seedInput.value = String(seed);
    }

    function launchFromCover(seedValue) {
        var seedStart = seedValue;
        // #region agent log
        debugLog937bb1('dashboard.js:launchFromCover:entry', 'launchFromCover invoked', { seedStart: seedStart }, 'start-btn-issue', 'H2');
        // #endregion
        setLoading(true);
        setCoverStatus('Launching with seed ' + (seedStart != null ? seedStart : 'default') + '…');
        loadMissionData(seedStart)
            .then(function() {
                if (cover) cover.style.display = 'none';
                if (dashboard) {
                    dashboard.classList.remove('dashboard-hidden');
                    dashboard.classList.add('dashboard-visible');
                }
                startStreamRefresh();
            })
            .catch(function(err) {
                console.error('Error loading mission data:', err);
                // #region agent log
                debugLog937bb1('dashboard.js:launchFromCover:catch', 'launchFromCover caught error', { error: String(err && err.message ? err.message : err) }, 'start-btn-issue', 'H4');
                // #endregion
                setCoverStatus('Failed to load mission data. Try again.', true);
            })
            .finally(setLoading);
    }

    // Direct listener on Seed/Start so they always work (some environments miss delegation)
    var seedBtn = document.getElementById('cover-seed-btn');
    if (seedBtn) {
        seedBtn.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            var seed = generateSeed();
            showSeedOnCover(seed);
            setCoverStatus('Seed generated. Press Start to launch with this seed.');
        });
    }
    if (startBtn) {
        startBtn.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            var rawStart = (seedInput && (seedInput.value || '').trim()) || '';
            var seedStart = rawStart === '' ? null : parseInt(rawStart, 10);
            // #region agent log
            debugLog937bb1('dashboard.js:startBtn:click', 'startBtn direct click', { rawStart: rawStart, parsedSeed: seedStart, invalidSeed: (rawStart !== '' && isNaN(seedStart)) }, 'start-btn-issue', 'H2');
            // #endregion
            if (rawStart !== '' && isNaN(seedStart)) {
                setCoverStatus('Enter a seed (integer) or press Seed then Start.', true);
                return;
            }
            launchFromCover(seedStart);
        });
    }

    // One listener on cover: delegate clicks; use closest() so click on button text still works
    if (cover) {
        cover.addEventListener('click', function(ev) {
            var target = ev.target;
            if (!target || !target.closest) return;
            var clicked = target.closest('#cover-seed-btn') || target.closest('#cover-seed-copy') || target.closest('#cover-add-btn') || target.closest('#cover-start-btn');
            if (!clicked) return;

            if (clicked.id === 'cover-seed-btn') {
                var seed = generateSeed();
                showSeedOnCover(seed);
                setCoverStatus('Seed generated. Press Start to launch with this seed.');
                return;
            }

            if (clicked.id === 'cover-seed-copy') {
                var value = generatedSeedEl ? generatedSeedEl.textContent : '';
                if (!value) return;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(value).then(function() {
                        setCoverStatus('Seed copied.', false);
                    }).catch(function() {
                        setCoverStatus('Failed to copy.', true);
                    });
                } else {
                    setCoverStatus('Clipboard not available in this browser.', true);
                }
                return;
            }

            if (clicked.id === 'cover-add-btn') {
                var raw = (seedInput && (seedInput.value || '').trim()) || '';
                var seedAdd = parseInt(raw, 10);
                if (raw === '' || isNaN(seedAdd)) {
                    setCoverStatus('Enter a seed (integer).', true);
                    return;
                }
                setAddLoading(true);
                setCoverStatus('Loading mission data…');
                loadMissionData(seedAdd)
                    .then(function() {
                        if (cover) cover.style.display = 'none';
                        if (dashboard) {
                            dashboard.classList.remove('dashboard-hidden');
                            dashboard.classList.add('dashboard-visible');
                        }
                        startStreamRefresh();
                    })
                    .catch(function(err) {
                        console.error('Error loading mission data:', err);
                        setCoverStatus('Failed to load mission data. Try again.', true);
                    })
                    .finally(setAddLoading);
                return;
            }

            if (clicked.id === 'cover-start-btn') {
                var rawStart = (seedInput && (seedInput.value || '').trim()) || '';
                var seedStart = rawStart === '' ? null : parseInt(rawStart, 10);
                if (rawStart !== '' && isNaN(seedStart)) {
                    setCoverStatus('Enter a seed (integer) or press Seed then Start.', true);
                    return;
                }
                launchFromCover(seedStart);
            }
        });
    }

    // Redraw radar on window resize
    var resizeTimeout;
    window.addEventListener('resize', function() {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(function() {
            if (lastMissionData) drawRadar(lastMissionData, radarSweepAngle);
        }, 250);
    });
}

// ── Sensor Log Panel ──────────────────────────────────────────────────────────

(function () {
    var _logCurrentFile = null;
    var _logRefreshInterval = null;

    function _parseLogLine(line) {
        var m = line.match(
            /^\[(.+?)\]\s+dist=(.+?)\s*\|\s*mag=(.+?)\s*\|\s*hall=(.+?)\s*\|\s*temp=(.+?)\s*\|\s*hum=.+?\s*\|\s*vib=(.+)$/
        );
        if (m) {
            return { time: m[1], dist: m[2], mag: m[3].trim(), hall: m[4].trim(), temp: m[5], vib: m[6].trim() };
        }
        m = line.match(/^\[(.+?)\]\s+dist=(.+?)\s*\|\s*mag=(.+?)\s*\|\s*temp=(.+?)\s*\|\s*hum=.+?\s*\|\s*vib=(.+)$/);
        if (!m) return null;
        return { time: m[1], dist: m[2], mag: m[3].trim(), hall: null, temp: m[4], vib: m[5].trim() };
    }

    function _renderTable(entries, lines) {
        var tbody = document.getElementById('sensor-log-tbody');
        var empty = document.getElementById('sensor-log-empty');
        if (!tbody) return;

        var parsed = [];
        if (Array.isArray(entries) && entries.length > 0) {
            for (var i = entries.length - 1; i >= 0; i--) {
                var e = entries[i] || {};
                var t = (e.timestamp || '').toString();
                var dist = formatArduinoUltrasonicReading(e) || '—';
                var mag = (e.magnetic === true) ? 'DETECTED' : ((e.magnetic === false) ? 'clear' : '—');
                var hall = (e.hall === true) ? 'DETECTED' : ((e.hall === false) ? 'clear' : null);
                var temp = (typeof e.temperature_c === 'number') ? (e.temperature_c.toFixed(1) + ' °C') : '—';
                var vib = (e.vibration === true) ? 'YES' : ((e.vibration === false) ? 'no' : '—');
                parsed.push({ time: t, dist: dist, mag: mag, hall: hall, temp: temp, vib: vib });
                if (parsed.length >= 100) break;
            }
        } else if (Array.isArray(lines)) {
            for (var j = lines.length - 1; j >= 0; j--) {
                var p = _parseLogLine(lines[j]);
                if (p) parsed.push(p);
                if (parsed.length >= 100) break;
            }
        }

        if (parsed.length === 0) {
            tbody.innerHTML = '';
            if (empty) empty.style.display = '';
            return;
        }
        if (empty) empty.style.display = 'none';

        var html = '';
        for (var k = 0; k < parsed.length; k++) {
            var r = parsed[k];
            var magCell = r.hall != null
                ? ((r.mag === 'DETECTED' || r.hall === 'DETECTED') ? 'DETECTED' : r.mag)
                : r.mag;
            var magClass = magCell === 'DETECTED' ? 'log-mag-yes' : 'log-mag-no';
            var vibClass = r.vib === 'YES' ? 'log-vib-yes' : 'log-vib-no';
            html += '<tr>' +
                '<td>' + r.time + '</td>' +
                '<td>' + r.dist + '</td>' +
                '<td class="' + magClass + '">' + magCell + '</td>' +
                '<td>' + r.temp + '</td>' +
                '<td class="' + vibClass + '">' + r.vib + '</td>' +
                '</tr>';
        }
        tbody.innerHTML = html;
    }

    function _loadLogFile(filename) {
        if (!filename) return;
        fetch('/api/arduino/logs/' + encodeURIComponent(filename) + '?lines=200')
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (!data) return;
                _renderTable(data.entries || null, data.lines || null);
            })
            .catch(function () {});
    }

    function _loadLatest() {
        var el = document.getElementById('sensor-log-latest');
        if (!el) return;
        fetch('/api/arduino/logs/latest')
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (obj) {
                if (!obj) { el.style.display = 'none'; return; }
                _arduinoLatestLogSnapshot = obj;
                if (typeof _orbitPopupArduinoOpen !== 'undefined' && _orbitPopupArduinoOpen) {
                    updatePopupWithArduino();
                }
                var lines = [
                    'Timestamp : ' + (obj.timestamp || '—'),
                    'Distance  : ' + (formatArduinoUltrasonicReading(obj) || '—'),
                    'Magnetic  : ' + (obj.magnetic === true ? 'DETECTED' : (obj.magnetic === false ? 'clear' : '—')),
                    'Hall      : ' + (obj.hall === true ? 'DETECTED' : (obj.hall === false ? 'clear' : '—')),
                    'Temp      : ' + (typeof obj.temperature_c === 'number' ? (obj.temperature_c.toFixed(1) + ' °C') : '—'),
                    'Humidity  : ' + (typeof obj.humidity_pct === 'number' ? (obj.humidity_pct.toFixed(1) + ' %') : '—'),
                    'Vibration : ' + (obj.vibration === true ? 'YES' : (obj.vibration === false ? 'no' : '—'))
                ];
                el.textContent = lines.join('\n');
                el.style.display = '';
            })
            .catch(function () { el.style.display = 'none'; });
    }

    function _loadIndex() {
        fetch('/api/arduino/logs')
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (data) {
                if (!data) return;

                var dirEl = document.getElementById('sensor-log-dir');
                if (dirEl && data.logs_dir) dirEl.textContent = data.logs_dir;

                var sel = document.getElementById('sensor-log-file-select');
                if (!sel) return;
                var files = data.files || [];

                if (files.length === 0) {
                    sel.innerHTML = '<option value="">— no files —</option>';
                    _logCurrentFile = null;
                    return;
                }

                var prevVal = sel.value || (files[0] && files[0].filename);
                sel.innerHTML = '';
                files.forEach(function (f) {
                    var opt = document.createElement('option');
                    opt.value = f.filename;
                    opt.textContent = f.filename + ' (' + Math.ceil(f.size_bytes / 1024) + ' KB)';
                    sel.appendChild(opt);
                });
                // Restore selection or default to newest
                sel.value = prevVal || files[0].filename;
                _logCurrentFile = sel.value;
                _loadLogFile(_logCurrentFile);
            })
            .catch(function () {});
    }

    function _refreshAll() {
        _loadLatest();
        if (_logCurrentFile) {
            _loadLogFile(_logCurrentFile);
        } else {
            _loadIndex();
        }
    }

    function initSensorLog() {
        var refreshBtn = document.getElementById('sensor-log-refresh');
        var sel = document.getElementById('sensor-log-file-select');

        if (refreshBtn) {
            refreshBtn.addEventListener('click', function () {
                _loadIndex();
                _loadLatest();
            });
        }

        if (sel) {
            sel.addEventListener('change', function () {
                _logCurrentFile = sel.value;
                _loadLogFile(_logCurrentFile);
            });
        }

        _loadIndex();
        _loadLatest();

        // Auto-refresh every 5 seconds
        _logRefreshInterval = setInterval(_refreshAll, 5000);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initSensorLog);
    } else {
        initSensorLog();
    }
})();

function startSimulatedBatteryTicker() {
    applySimulatedBatteryUI();
    if (_simBatteryIntervalId != null) clearInterval(_simBatteryIntervalId);
    _simBatteryIntervalId = setInterval(applySimulatedBatteryUI, 1000);
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
        initCover();
        startSimulatedBatteryTicker();
        startVirtualSensorsStream();
        startArduinoAutoConnect();
        startArduinoPolling();
    });
} else {
    initCover();
    startSimulatedBatteryTicker();
    startVirtualSensorsStream();
    startArduinoAutoConnect();
    startArduinoPolling();
}

