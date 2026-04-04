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
var _orbitPopupRelVelKmS = null;
/** Расстояние до выбранного мусора из симуляции (км), пока попап открыт */
var _orbitPopupSimDistanceKm = null;
var _popupArduinoIntervalId = null;
/** Попап «мусор» открыт — обновлять ARDUINO LIVE по SSE/fetch без проверки style.display */
var _orbitPopupArduinoOpen = false;
const ORBIT_SHIP_FPS = 5; // update ship at 5fps when not dragging; paused while user rotates
// Отложенный рендер орбиты: пока пользователь держит мышь на сфере — не обновляем; после отпускания — один раз отрисуем
var orbitChartDeferredRender = false;

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
    var map = { fragment: 'Обломок', rocket_body: 'Корпус ракеты', satellite: 'Фрагмент спутника', tool: 'Инструмент', panel: 'Панель (солнечная/MLI)' };
    return (map[type] != null ? map[type] : type) || 'Неизвестно';
}
function getAnomalyReasonLabel(type) {
    var map = {
        TELEMETRY_ANOMALY: 'Телеметрическая аномалия',
        SENSOR_DEVIATION: 'Отклонение в датчиках',
        COMM_LOSS: 'Сбой связи',
        ORIENTATION_ANOMALY: 'Аномалия ориентации',
        GYRO_DRIFT: 'Дрейф гироскопа',
        POWER_FLUCTUATION: 'Колебания питания',
        MAGNETIC_FIELD: 'Аномалия магнитного поля (Arduino)',
        VIBRATION_SPIKE: 'Вибрация корпуса (Arduino)',
    };
    return (map[type] != null ? map[type] : type) || 'Аномалия телеметрии';
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

/** Seeds without anomalies (seed % 7 in {0, 1}). */
const NO_ANOMALY_SEEDS = [0, 1, 7, 8, 14, 15, 21, 22, 28, 29, 35, 36, 42, 43, 49, 50, 56, 57, 63, 64, 70, 71, 77, 78, 84, 85, 91, 92, 98, 99];
/** Seeds with anomalies (seed % 7 not in {0, 1}). */
const WITH_ANOMALY_SEEDS = [2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 16, 17, 18, 19, 20, 23, 24, 25, 26, 27, 30, 31, 32, 33, 34, 37, 38, 39, 40, 41, 44, 45, 46, 47, 48];

/** Time-based alternating seed for stream: 50% no-anomaly, 50% with-anomaly. */
function getAlternatingStreamSeed() {
    var periodSec = 45;
    var tick = Math.floor(Date.now() / 1000 / periodSec);
    var useNoAnomaly = (tick % 2) === 0;
    var arr = useNoAnomaly ? NO_ANOMALY_SEEDS : WITH_ANOMALY_SEEDS;
    return arr[tick % arr.length];
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
    const url = '/api/mission-data' + (seed != null ? '?seed=' + encodeURIComponent(seed) : '');
    const response = await fetch(url);
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

    if (document.getElementById('position-chart')) renderPositionChart(data);
    if (document.getElementById('velocity-chart')) renderVelocityChart(data);
    if (document.getElementById('resources-chart')) renderResourcesChart(data);
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
    updateDangerWarnings(data);

    if (data.seed != null) {
        const seedEl = document.getElementById('seed-value');
        const blockEl = document.getElementById('current-seed');
        if (seedEl) seedEl.textContent = data.seed;
        if (blockEl) blockEl.style.display = 'inline';
    }
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

function updateVelocityGauge(data) {
    const speeds = data.speeds || [];
    const speedKmh = speeds.length ? speeds[speeds.length - 1] * 3600 : 0;
    const gaugeValue = document.getElementById('gauge-value');
    const gaugeArc = document.getElementById('gauge-arc');
    if (gaugeValue) gaugeValue.textContent = speedKmh.toFixed(0).replace(/\B(?=(\d{3})+(?!\d))/g, ' ');
    if (gaugeArc) {
        const maxKmh = 40000;
        const ratio = Math.min(1, speedKmh / maxKmh);
        gaugeArc.style.strokeDashoffset = 245 * (1 - ratio);
    }
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
    const fuels = data.fuels || [];
    const conf = data.confidences || [];
    const fuelPct = fuels.length && fuels[0] ? (fuels[fuels.length - 1] / fuels[0]) * 100 : 89;
    const shieldPct = conf.length ? Math.round(conf[conf.length - 1] * 47 + 50) : 47;
    const batFill = document.getElementById('battery-fill');
    const batPct = document.getElementById('battery-pct');
    const shieldEl = document.getElementById('shields-pct');
    if (batFill) batFill.style.width = Math.min(100, fuelPct) + '%';
    if (batPct) batPct.textContent = Math.round(fuelPct) + '%';
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
            name: 'Корабль',
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
                name: 'Расстояние до мусора',
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
                name: 'Траектория мусора',
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
            text: avoidedDebris.map(d => '(обход) ' + (d.distance_km * 1000).toFixed(0) + ' m'),
            hoverinfo: 'text',
            name: 'Мусор (обход, сзади)',
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
                name: `Мусор (${safeNormal.length})`,
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
                name: `Опасный мусор (${dangerousNormal.length})`,
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
                name: `Опасный мусор (${dangerousDebris.length})`,
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
        const velocity = cd[4], distanceKm = cd[5], relVelKmS = cd[6], objectId = cd[7];
        const popup = document.getElementById('orbit-debris-popup');
        if (popup) {
            _orbitPopupRelVelKmS = relVelKmS;
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
            _orbitPopupRelVelKmS = null;
            _orbitPopupSimDistanceKm = null;
            _orbitPopupArduinoOpen = false;
            stopPopupArduinoPolling();
        });
    }
}

// Position telemetry
function renderPositionChart(data) {
    const traces = [
        {
            x: data.times,
            y: data.positions.map(p => p[0]),
            name: 'X',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.blue, width: 2 },
        },
        {
            x: data.times,
            y: data.positions.map(p => p[1]),
            name: 'Y',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.cyan, width: 2 },
        },
        {
            x: data.times,
            y: data.positions.map(p => p[2]),
            name: 'Z',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.purple, width: 2 },
        },
    ];
    
    const layout = {
        ...commonLayout,
        xaxis: {
            title: 'Time (s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis: {
            title: 'Position (km)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
    };
    
    Plotly.newPlot('position-chart', traces, layout, {responsive: true});
}

// Velocity and fusion confidence
function renderVelocityChart(data) {
    const traces = [
        {
            x: data.times,
            y: data.speeds,
            name: 'Speed (km/s)',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.green, width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(63, 185, 80, 0.1)',
        },
        {
            x: data.times,
            y: data.confidences,
            name: 'Fusion Confidence',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.purple, width: 2, dash: 'dash' },
            yaxis: 'y2',
            fill: 'tozeroy',
            fillcolor: 'rgba(165, 165, 255, 0.1)',
        },
    ];
    
    const layout = {
        ...commonLayout,
        xaxis: {
            title: 'Time (s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis: {
            title: 'Speed (km/s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis2: {
            title: 'Confidence',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'transparent',
            range: [0, 1.05],
            color: COLORS.purple,
        },
    };
    
    Plotly.newPlot('velocity-chart', traces, layout, {responsive: true});
}

// Resources (fuel and debris count) with danger overlay
function renderResourcesChart(data) {
    const traces = [
        {
            x: data.times,
            y: data.fuels,
            name: 'Fuel (kg)',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.orange, width: 2 },
            fill: 'tozeroy',
            fillcolor: 'rgba(255, 140, 66, 0.1)',
        },
        {
            x: data.times,
            y: data.debris_counts,
            name: 'Debris Count',
            type: 'scatter',
            mode: 'lines',
            line: { color: COLORS.cyan, width: 2, dash: 'dot' },
            yaxis: 'y2',
            fill: 'tozeroy',
            fillcolor: 'rgba(0, 212, 255, 0.1)',
        },
    ];
    
    // Add danger level if available
    if (data.danger_levels && data.danger_levels.length > 0) {
        const dangerColor = data.danger_levels[data.danger_levels.length - 1] > 0.7 ? '#ff4444' :
                           data.danger_levels[data.danger_levels.length - 1] > 0.4 ? '#ff8c42' :
                           data.danger_levels[data.danger_levels.length - 1] > 0.1 ? '#ffaa00' : COLORS.green;
        
        traces.push({
            x: data.times,
            y: data.danger_levels,
            name: 'Уровень опасности',
            type: 'scatter',
            mode: 'lines',
            line: { color: dangerColor, width: 2.5 },
            yaxis: 'y3',
            fill: 'tozeroy',
            fillcolor: dangerColor.replace(')', ', 0.2)').replace('rgb', 'rgba'),
        });
    }
    
    const layout = {
        ...commonLayout,
        xaxis: {
            title: 'Time (s)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis: {
            title: 'Fuel (kg)',
            gridcolor: COLORS.grid,
            color: COLORS.textSecondary,
        },
        yaxis2: {
            title: 'Debris Count',
            overlaying: 'y',
            side: 'right',
            gridcolor: 'transparent',
            color: COLORS.cyan,
        },
        yaxis3: {
            title: 'Уровень опасности',
            overlaying: 'y',
            side: 'right',
            position: 0.95,
            gridcolor: 'transparent',
            range: [0, 1.05],
            color: data.danger_levels && data.danger_levels.length > 0 && data.danger_levels[data.danger_levels.length - 1] > 0.1 ? '#ff4444' : COLORS.textSecondary,
        },
    };
    
    Plotly.newPlot('resources-chart', traces, layout, {responsive: true});
}

// Update danger status
function updateDangerStatus(data) {
    const dangerLevels = data.danger_levels || [];
    const currentDanger = dangerLevels.length > 0 ? dangerLevels[dangerLevels.length - 1] : 0.0;
    
    // Update danger gauge
    const dangerBar = document.getElementById('danger-level-bar');
    const dangerValue = document.getElementById('danger-level-value');
    if (dangerBar) {
        dangerBar.style.width = (currentDanger * 100) + '%';
        if (currentDanger > 0.7) {
            dangerBar.style.background = '#ff4444';
        } else if (currentDanger > 0.4) {
            dangerBar.style.background = '#ff8c42';
        } else if (currentDanger > 0.1) {
            dangerBar.style.background = '#ffaa00';
        } else {
            dangerBar.style.background = COLORS.green;
        }
    }
    const anomalies = data.anomaly_detections || [];
    if (dangerValue) {
        var dangerLabel;
        if (currentDanger >= 0.6) {
            dangerLabel = 'высокая';
        } else if (currentDanger >= 0.2 || anomalies.length > 0) {
            dangerLabel = 'средний';
        } else {
            dangerLabel = 'низкий';
        }
        dangerValue.textContent = dangerLabel;
        dangerValue.style.color = currentDanger > 0.7 ? '#ff4444' :
                                  currentDanger > 0.4 ? '#ff8c42' :
                                  currentDanger > 0.1 ? '#ffaa00' : COLORS.green;
    }
    
    // Update stats (значения после подписи: уровень опасности, аномалии, топливо)
    const collisionCount = document.getElementById('collision-count');
    const anomalyCount = document.getElementById('anomaly-count');
    const lowFuelCount = document.getElementById('low-fuel-count');
    const fuelMassEl = document.getElementById('fuel-mass-value');
    if (collisionCount) collisionCount.textContent = (data.collision_warnings || []).length;
    if (anomalyCount) {
        if (anomalies.length === 0) {
            anomalyCount.textContent = '0';
        } else {
            var maxScore = Math.max.apply(null, anomalies.map(function(a) { return a.score != null ? a.score : 0; }));
            anomalyCount.textContent = 'АНОМАЛИЯ ' + (maxScore * 100).toFixed(0) + '%';
        }
    }
    if (lowFuelCount) lowFuelCount.textContent = (data.low_fuel_warnings || []).length;
    if (fuelMassEl) {
        if (anomalies.length > 0) {
            fuelMassEl.textContent = 'заканчивается';
        } else {
            const fuel = data.spacecraft_status && data.spacecraft_status.fuel != null
                ? data.spacecraft_status.fuel
                : (data.fuels && data.fuels.length > 0 ? data.fuels[data.fuels.length - 1] : null);
            fuelMassEl.textContent = fuel != null ? Number(fuel).toFixed(1) : '—';
        }
    }
    
    // Update footer status
    const sysStatus = document.getElementById('sys-status');
    const statusDot = document.getElementById('status-dot');
    const warningsCount = document.getElementById('warnings-count');
    const warningsText = document.getElementById('warnings-text');
    const warningsNum = document.getElementById('warnings-num');
    
    const totalWarnings = (data.collision_warnings || []).length + 
                         (data.anomaly_detections || []).length + 
                         (data.low_fuel_warnings || []).length;
    
    let statusText = 'SYS NOMINAL';
    let statusColor = COLORS.green;
    
    if (currentDanger > 0.7) {
        statusText = 'STATUS: CRITICAL';
        statusColor = '#ff4444';
    } else if (currentDanger > 0.4) {
        statusText = 'STATUS: WARNING';
        statusColor = '#ff8c42';
    } else if (currentDanger > 0.1) {
        statusText = 'STATUS: CAUTION';
        statusColor = '#ffaa00';
    }
    
    if (sysStatus) {
        sysStatus.textContent = statusText;
        sysStatus.style.color = statusColor;
    }
    
    if (statusDot) {
        statusDot.style.background = statusColor;
        statusDot.style.boxShadow = '0 0 10px ' + statusColor;
    }
    
    if (totalWarnings > 0) {
        if (warningsCount) warningsCount.style.display = 'inline';
        if (warningsText) warningsText.style.display = 'inline';
        if (warningsNum) warningsNum.textContent = totalWarnings;
    } else {
        if (warningsCount) warningsCount.style.display = 'none';
        if (warningsText) warningsText.style.display = 'none';
    }
}

// Update danger warnings panel
function updateDangerWarnings(data) {
    const dangerPanel = document.getElementById('danger-panel');
    const dangerContent = document.getElementById('danger-content');
    
    if (!dangerPanel || !dangerContent) return;
    
    const warnings = [];
    
    // Client-side: if any debris is very close, add collision warning
    const COLLISION_WARNING_KM = 0.05;
    if (data.debris && Array.isArray(data.debris) && data.debris.length > 0) {
        var closest = data.debris.reduce(function(acc, d) {
            var dist = (d && typeof d.distance_km === 'number') ? d.distance_km : 1e9;
            return dist < acc.dist ? { dist: dist, d: d } : acc;
        }, { dist: 1e9, d: null });
        if (closest.dist < COLLISION_WARNING_KM && closest.d) {
            warnings.push({
                icon: '⚠️',
                text: 'Предупреждение о столкновении: объект на расстоянии ' + (closest.dist * 1000).toFixed(0) + ' m',
                severity: 'HIGH',
                time: 0,
            });
        }
    }
    
    // Collision warnings from backend
    (data.collision_warnings || []).forEach(w => {
        warnings.push({
            icon: '⚠️',
            text: `СТОЛКНОВЕНИЕ! Вероятность: ${(w.probability * 100).toFixed(1)}%`,
            severity: w.severity,
            time: w.time,
        });
    });
    
    // Аномалии: считаем только неустранённые для отображения
    var rawAnomalies = data.anomaly_detections || [];

    // Дополнительные аномалии на основе живых датчиков Arduino:
    // если обнаружено магнитное поле или вибрация, считаем это аномалией.
    // Флаг _arduinoAnomaliesSuppressed временно отключает их, пока оператор нажал «Устранить все».
    try {
        if (typeof _arduinoLive !== 'undefined' && _arduinoLive) {
            var hasMagAnomaly = _arduinoLive.magnetic === true;
            var hasVibAnomaly = _arduinoLive.vibration === true;
            if (!hasMagAnomaly && !hasVibAnomaly) {
                // как только датчики вернулись в норму — снова разрешаем показывать Arduino‑аномалии
                _arduinoAnomaliesSuppressed = false;
            }
            if (!_arduinoAnomaliesSuppressed && (hasMagAnomaly || hasVibAnomaly)) {
                var nowSec = Date.now() / 1000;
                var baseScore = 0.7;
                if (hasMagAnomaly && hasVibAnomaly) {
                    baseScore = 0.9;
                } else if (hasMagAnomaly || hasVibAnomaly) {
                    baseScore = 0.8;
                }
                if (hasMagAnomaly) {
                    rawAnomalies.push({
                        time: nowSec,
                        score: baseScore,
                        type: 'MAGNETIC_FIELD',
                    });
                }
                if (hasVibAnomaly) {
                    rawAnomalies.push({
                        time: nowSec + 0.1,
                        score: baseScore,
                        type: 'VIBRATION_SPIKE',
                    });
                }
            }
        }
    } catch (e) {
        // игнорируем ошибки, чтобы не ломать основной поток отрисовки
    }
    var visibleAnomalies = [];
    rawAnomalies.forEach(function(a, idx) {
        var key = 'anomaly_' + (a.time != null ? a.time : 0) + '_' + (a.score != null ? a.score : 0) + '_' + idx;
        if (dismissedAnomalyKeys[key]) return;
        visibleAnomalies.push(a);
        warnings.push({
            icon: '🚨',
            text: 'АНОМАЛИЯ! Уровень: ' + ((a.score != null ? a.score : 0) * 100).toFixed(1) + '%',
            severity: (a.score != null && a.score > 0.7) ? 'HIGH' : 'MEDIUM',
            time: a.time != null ? a.time : 0,
            isAnomaly: true,
            reason: getAnomalyReasonLabel(a.type),
            anomalyKey: key,
            immediatePanel: a.type === 'VIBRATION_SPIKE',
        });
    });
    
    // Low fuel warnings
    (data.low_fuel_warnings || []).forEach(f => {
        warnings.push({
            icon: '⛽',
            text: `НИЗКИЙ УРОВЕНЬ ТОПЛИВА! ${(f.fuel_ratio * 100).toFixed(1)}%`,
            severity: 'MEDIUM',
            time: f.time,
        });
    });
    
    var needsImmediatePanel = warnings.some(function(w) { return w.immediatePanel; });
    if (warnings.length > 0) {
        if (needsImmediatePanel) {
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
                ? '<div class="danger-warning-reason">Причина: ' + w.reason + '</div>'
                : '';
            var btnBlock = w.isAnomaly && w.anomalyKey
                ? '<button type="button" class="danger-warning-resolve" data-anomaly-key="' + w.anomalyKey + '">Устранить</button>'
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
            html += '<div class="danger-resolve-all-wrap"><button type="button" class="danger-warning-resolve-all" id="danger-resolve-all-btn">Устранить все</button></div>';
        }
        dangerContent.innerHTML = html;
        dangerContent.querySelectorAll('.danger-warning-resolve[data-anomaly-key]').forEach(function(btn) {
            btn.addEventListener('click', function() {
                var key = btn.getAttribute('data-anomaly-key');
                if (key) dismissedAnomalyKeys[key] = true;
                if (lastMissionData) updateDangerWarnings(lastMissionData);
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
                if (lastMissionData) updateDangerWarnings(lastMissionData);
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
}

// ── Arduino live sensor data (SSE push + fetch fallback) ───────────────────
var _arduinoLive = {
    distance_km: null,
    magnetic: null,
    temperature: null,
    humidity: null,
    vibration: null,
    updated_at: null,
    error: null,
};
var _arduinoPrevMagnetic = null;
var _arduinoPrevVibration = null;
var _arduinoAnomaliesSuppressed = false;
var _arduinoPollingId = null;
var _arduinoEventSource = null;

function arduinoLiveNum(v) {
    if (v == null || v === '') return null;
    var n = typeof v === 'number' ? v : Number(v);
    return isNaN(n) ? null : n;
}

function fetchArduinoLive() {
    return fetch('/api/arduino/live')
        .then(function (r) {
            if (!r.ok) {
                var msg = r.status === 404
                    ? 'Нет маршрута /api/arduino (установите pyserial, перезапустите сервер)'
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
            _arduinoLive = Object.assign({}, _arduinoLive, { error: 'Сеть / сервер недоступны' });
            if (_orbitPopupArduinoOpen) updatePopupWithArduino();
        });
}

function applyArduinoLiveData(data) {
    if (!data || typeof data !== 'object') return;
    _arduinoLive = Object.assign({}, _arduinoLive, data);

    // При первом появлении магнитного поля / вибрации с Arduino сразу обновляем панель опасности,
    // без ожидания следующего loadMissionData (~45 с) и без 6 с задержки для вибрации.
    try {
        var prevMag = _arduinoPrevMagnetic;
        var prevVib = _arduinoPrevVibration;
        var nowMag = _arduinoLive && _arduinoLive.magnetic === true;
        var nowVib = _arduinoLive && _arduinoLive.vibration === true;
        _arduinoPrevMagnetic = _arduinoLive ? _arduinoLive.magnetic : null;
        _arduinoPrevVibration = _arduinoLive ? _arduinoLive.vibration : null;
        var magRising = !prevMag && nowMag;
        var vibRising = !prevVib && nowVib;
        if (magRising || vibRising) {
            if (lastMissionData) {
                if (dangerPanelShowTimeoutId) {
                    clearTimeout(dangerPanelShowTimeoutId);
                    dangerPanelShowTimeoutId = null;
                }
                updateDangerWarnings(lastMissionData);
            } else {
                var dangerPanel = document.getElementById('danger-panel');
                var dangerContent = document.getElementById('danger-content');
                if (dangerPanel && dangerContent) {
                    var tSec = Date.now() / 1000;
                    var text;
                    var reasonLine;
                    if (magRising && vibRising) {
                        text = 'АНОМАЛИЯ! Магнитное поле и вибрация (Arduino).';
                        reasonLine = getAnomalyReasonLabel('MAGNETIC_FIELD') + ' · ' + getAnomalyReasonLabel('VIBRATION_SPIKE');
                    } else if (magRising) {
                        text = 'АНОМАЛИЯ! Обнаружено магнитное поле (Arduino).';
                        reasonLine = getAnomalyReasonLabel('MAGNETIC_FIELD');
                    } else {
                        text = 'АНОМАЛИЯ! Обнаружена вибрация корпуса (Arduino).';
                        reasonLine = getAnomalyReasonLabel('VIBRATION_SPIKE');
                    }
                    var html = '<div class="danger-warning danger-high">' +
                        '<span class="danger-warning-icon">🚨</span>' +
                        '<div class="danger-warning-body">' +
                        '<span class="danger-warning-text">' + text + '</span>' +
                        (reasonLine ? '<div class="danger-warning-reason">Причина: ' + reasonLine + '</div>' : '') +
                        '</div>' +
                        '<span class="danger-warning-time">T+' + tSec.toFixed(1) + 's</span>' +
                        '</div>';
                    dangerContent.innerHTML = html;
                    dangerPanel.style.display = 'block';
                }
            }
        }
    } catch (e) {
        // не даём сбойнуть всей панели, если что-то пошло не так
    }
    if (_orbitPopupArduinoOpen) {
        updatePopupWithArduino();
    }
}

function updateOrbitPopupTtc() {
    var ttcEl = document.getElementById('orbit-popup-ttc');
    if (!ttcEl || !_orbitPopupArduinoOpen) return;
    var distKm = _arduinoLive.distance_km != null ? _arduinoLive.distance_km : _orbitPopupSimDistanceKm;
    var rv = _orbitPopupRelVelKmS;
    if (distKm != null && rv != null && rv > 1e-6) {
        var ttcSec = distKm / rv;
        ttcEl.textContent = ttcSec >= 60 ? (ttcSec / 60).toFixed(1) + ' min' : ttcSec.toFixed(1) + ' s';
    } else {
        ttcEl.textContent = '—';
    }
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
    var distEl = document.getElementById('orbit-popup-distance');
    var magEl = document.getElementById('orbit-popup-magnetic');
    var tempEl = document.getElementById('orbit-popup-temp');
    var vibEl = document.getElementById('orbit-popup-vibration');
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

    // Только датчик расстояния (ультразвук → км в скетче); симуляция орбиты не подмешиваем
    if (distEl) {
        var distKm = arduinoLiveNum(d.distance_km);
        distEl.textContent = (distKm != null) ? distKm.toFixed(1) + ' km' : '—';
    }
    if (magEl) {
        if (d.magnetic === true) {
            magEl.textContent = 'Обнаружено';
            magEl.style.color = '#ff5555';
        } else if (d.magnetic === false) {
            magEl.textContent = 'Нет';
            magEl.style.color = '#22c55e';
        } else {
            magEl.textContent = '—';
            magEl.style.color = '';
        }
    }
    if (tempEl) {
        var t = arduinoLiveNum(d.temperature);
        tempEl.textContent = (t != null) ? t.toFixed(1) + ' °C' : '—';
    }
    if (vibEl) {
        if (d.vibration === true) {
            vibEl.textContent = 'Да';
            vibEl.style.color = '#f97316';
        } else if (d.vibration === false) {
            vibEl.textContent = 'Нет';
            vibEl.style.color = '#22c55e';
        } else {
            vibEl.textContent = '—';
            vibEl.style.color = '';
        }
    }
    updateOrbitPopupTtc();
}

function startPopupArduinoPolling() {
    stopPopupArduinoPolling();
    fetchArduinoLive();
    _popupArduinoIntervalId = setInterval(function () {
        fetchArduinoLive();
    }, 50);
}

function stopPopupArduinoPolling() {
    if (_popupArduinoIntervalId) {
        clearInterval(_popupArduinoIntervalId);
        _popupArduinoIntervalId = null;
    }
}
// ─────────────────────────────────────────────────────────────────────────────

// Initialize on page load: cover visible, dashboard hidden; no auto-load.
function initCover() {
    startLiveTime();
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

    // Direct listener on Seed button so it always works (some environments miss delegation)
    var seedBtn = document.getElementById('cover-seed-btn');
    if (seedBtn) {
        seedBtn.addEventListener('click', function(ev) {
            ev.preventDefault();
            ev.stopPropagation();
            var seed = generateSeed();
            showSeedOnCover(seed);
            setCoverStatus('Сид сгенерирован. Нажмите Start для запуска сайта с этим сидом.');
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
                setCoverStatus('Сид сгенерирован. Нажмите Start для запуска сайта с этим сидом.');
                return;
            }

            if (clicked.id === 'cover-seed-copy') {
                var value = generatedSeedEl ? generatedSeedEl.textContent : '';
                if (!value) return;
                if (navigator.clipboard && navigator.clipboard.writeText) {
                    navigator.clipboard.writeText(value).then(function() {
                        setCoverStatus('Сид скопирован.', false);
                    }).catch(function() {
                        setCoverStatus('Не удалось скопировать.', true);
                    });
                } else {
                    setCoverStatus('Копирование недоступно в этом браузере.', true);
                }
                return;
            }

            if (clicked.id === 'cover-add-btn') {
                var raw = (seedInput && (seedInput.value || '').trim()) || '';
                var seedAdd = parseInt(raw, 10);
                if (raw === '' || isNaN(seedAdd)) {
                    setCoverStatus('Введите сид (целое число).', true);
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
                    setCoverStatus('Введите сид (целое число) или нажмите Seed и затем Start.', true);
                    return;
                }
                setLoading(true);
                setCoverStatus('Запуск с сидом ' + (seedStart != null ? seedStart : 'по умолчанию') + '…');
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
                        setCoverStatus('Failed to load mission data. Try again.', true);
                    })
                    .finally(setLoading);
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
        // "[2026-04-03 12:34:56] dist=144.0 km | mag=DETECTED | temp=24.5 °C | hum=65.0 % | vib=YES"
        var m = line.match(/^\[(.+?)\]\s+dist=(.+?)\s*\|\s*mag=(.+?)\s*\|\s*temp=(.+?)\s*\|\s*hum=(.+?)\s*\|\s*vib=(.+)$/);
        if (!m) return null;
        return { time: m[1], dist: m[2], mag: m[3].trim(), temp: m[4], hum: m[5], vib: m[6].trim() };
    }

    function _renderTable(lines) {
        var tbody = document.getElementById('sensor-log-tbody');
        var empty = document.getElementById('sensor-log-empty');
        if (!tbody) return;

        var parsed = [];
        for (var i = lines.length - 1; i >= 0; i--) {
            var p = _parseLogLine(lines[i]);
            if (p) parsed.push(p);
            if (parsed.length >= 100) break;
        }

        if (parsed.length === 0) {
            tbody.innerHTML = '';
            if (empty) empty.style.display = '';
            return;
        }
        if (empty) empty.style.display = 'none';

        var html = '';
        for (var j = 0; j < parsed.length; j++) {
            var r = parsed[j];
            var magClass = r.mag === 'DETECTED' ? 'log-mag-yes' : 'log-mag-no';
            var vibClass = r.vib === 'YES' ? 'log-vib-yes' : 'log-vib-no';
            html += '<tr>' +
                '<td>' + r.time + '</td>' +
                '<td>' + r.dist + '</td>' +
                '<td class="' + magClass + '">' + r.mag + '</td>' +
                '<td>' + r.temp + '</td>' +
                '<td>' + r.hum + '</td>' +
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
                if (!data || !data.lines) return;
                _renderTable(data.lines);
            })
            .catch(function () {});
    }

    function _loadLatest() {
        var el = document.getElementById('sensor-log-latest');
        if (!el) return;
        fetch('/api/arduino/logs/latest')
            .then(function (r) { return r.ok ? r.text() : null; })
            .then(function (text) {
                if (!text) { el.style.display = 'none'; return; }
                el.textContent = text;
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
                    sel.innerHTML = '<option value="">— нет файлов —</option>';
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

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', function () {
        initCover();
        startArduinoAutoConnect();
        startArduinoPolling();
    });
} else {
    initCover();
    startArduinoAutoConnect();
    startArduinoPolling();
}

